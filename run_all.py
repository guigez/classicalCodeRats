#!/usr/bin/env python3
"""
ORQUESTRADOR
============

Substitui o run_all.sh. Percorre as etapas na ordem e dispara cada uma como
SUBPROCESSO separado.

POR QUE SUBPROCESSO, E NAO IMPORTAR AS FUNCOES
----------------------------------------------
A tentacao ao trocar bash por Python e rodar tudo dentro do mesmo interpretador.
Nao faca. Com dezenas de treinos numa sessao:

  - o alocador de cache do PyTorch acumula fragmentacao de memoria de GPU. O
    padrao tipico e rodar 40 modelos e estourar OOM no 41o — nao porque ele
    seja maior, mas porque a memoria virou queijo suico. empty_cache() ajuda e
    nao resolve;
  - qualquer vazamento (grafo retido, lista que cresce) se soma por todos os
    treinos em vez de morrer com o processo;
  - uma excecao nao tratada no candidato 100 leva junto os 99 anteriores.

Processo separado por (fold, familia) limita cada um desses estragos ao seu
proprio par. O ganho de trocar bash por Python nao e elegancia — e poder
capturar a falha de um fold, registrar, e CONTINUAR, coisa que `set -e` nao
permite.

Uso:
    python run_all.py
    python run_all.py --task binary --device cuda
    python run_all.py --force --no-shap
    python run_all.py --only-folds 0 1 --only-families bernini_cnn
    python run_all.py --task binary --split_mode fixed_paired \\
        --fixed_test_subjects 1 9 10 11 --seeds 1 2 3 4 5

REUSO DE ETAPAS
---------------
Cada etapa grava um `params.json` com o que a determinou (hash do manifesto,
do fold e do JSON do grid, semente, modo de particao...). Na execucao
seguinte a etapa so e pulada se a saida existir E os parametros baterem.
Antes, bastava o arquivo existir: mudar --split_mode ou o grid e rodar de
novo reaproveitava em silencio o resultado antigo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

FAMILIES = ("bernini_cnn", "resnet50_rf")

# Scripts e configs sao localizados a partir deste arquivo, entao o
# orquestrador roda de qualquer diretorio. Caminhos de DADOS (--metadata,
# --images_dir, --results_dir) continuam relativos ao diretorio atual.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"


def setup_logging(log_path: Path) -> logging.Logger:
    """Log simultaneo em arquivo e no terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")
    log = logging.getLogger("pipeline")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(log_path, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def run_step(cmd: list[str], log: logging.Logger, nome: str) -> tuple[bool, float]:
    """Roda um subprocesso e retransmite a saida linha a linha.

    PYTHONUNBUFFERED=1 no ambiente do filho e o que faz a saida aparecer em
    tempo real. Sem isso, Python bufferiza stdout quando nao esta num terminal
    e o log fica vazio por horas — voce nao sabe se travou ou esta trabalhando.
    """
    log.info(f"--- {nome} ---")
    log.info("  $ " + " ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=env)
        for linha in proc.stdout:
            linha = linha.rstrip()
            if linha:
                log.info("  | " + linha)
        codigo = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    dt = time.time() - t0
    if codigo != 0:
        log.error(f"  {nome} FALHOU (codigo {codigo}) apos {timedelta(seconds=int(dt))}")
        return False, dt
    log.info(f"  {nome} ok em {timedelta(seconds=int(dt))}")
    return True, dt


def file_sha(path: Path) -> str:
    """Hash curto do conteudo de um arquivo, para detectar mudanca."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def precisa_rodar(stamp: Path, saida: Path, params: dict, force: bool,
                  log: logging.Logger, nome: str) -> bool:
    """Decide se a etapa roda, comparando `params` com o registro anterior.

    Saida sem registro vem de uma execucao anterior a este mecanismo: e
    reaproveitada com aviso, para nao obrigar a refazer resultados validos.
    """
    if force or not saida.exists():
        return True
    if not stamp.exists():
        log.warning(f"  {nome}: saida sem params.json (execucao antiga); reusando. "
                    f"Use --force se os parametros mudaram.")
        return False
    antigo = json.loads(stamp.read_text())
    if antigo != params:
        mudou = sorted(k for k in set(antigo) | set(params) if antigo.get(k) != params.get(k))
        log.warning(f"  {nome}: parametros mudaram {mudou}; refazendo.")
        return True
    return False


def grava_stamp(stamp: Path, params: dict) -> None:
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(params, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Orquestrador do pipeline.")
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--results_dir", default="resultados")
    ap.add_argument("--task", default="multiclass",
                    choices=["multiclass", "three_bernini", "three_mlss", "binary"],
                    help="Cenario de agrupamento de classes. Os quatro da secao "
                         "4.2 da dissertacao.")
    ap.add_argument("--split_mode", default="group_kfold",
                    choices=["group_kfold", "complete_subject_paired",
                             "complete_subject_loso", "fixed_paired", "loso"],
                    help="Politica de particao. fixed_paired: usa exatamente os "
                         "sujeitos de --fixed_test_subjects, para manter o mesmo "
                         "split entre cenarios de agrupamento.")
    ap.add_argument("--fixed_test_subjects", type=int, nargs="*", default=None,
                    help="Sujeitos de teste para fixed_paired. Ex.: 1 9 10 11")
    ap.add_argument("--k", type=int, default=4, help="Numero de folds (group_kfold).")
    ap.add_argument("--n_val_subjects", type=int, default=2)
    ap.add_argument("--val_strategy", default="coverage", choices=["random", "coverage"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Retreina o vencedor de cada fold com estas sementes "
                         "(replicas para medir o ruido de treino). Ex.: 1 2 3 4 5")
    ap.add_argument("--keep_seed_models", action="store_true",
                    help="Mantem model.pt das replicas (por padrao sao apagados).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force", action="store_true", help="Refaz tudo do zero.")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--with-background", action="store_true",
                    help="Imagens ainda tem a esteira (afeta so a mascara do SHAP).")
    ap.add_argument("--only-folds", type=int, nargs="*", default=None)
    ap.add_argument("--only-families", nargs="*", default=None, choices=list(FAMILIES))
    args = ap.parse_args()

    py = sys.executable
    # Cada cenario de agrupamento vai para uma subpasta propria, senao os
    # quatro sobrescreveriam os mesmos folds e resultados: os resultados de
    # cada tarefa ficam sempre em <results_dir>/<task>/. Antes isso so valia
    # para o --results_dir padrao, e um diretorio explicito misturava as
    # quatro tarefas na mesma pasta.
    res = Path(args.results_dir) / args.task
    splits = res / "splits"
    manifest = res / f"manifest_{args.task}.csv"
    familias = args.only_families or list(FAMILIES)

    log = setup_logging(res / "run.log")
    log.info("=" * 70)
    log.info(f"tarefa={args.task}  modo={args.split_mode}  k={args.k}  "
             f"val={args.n_val_subjects} sujeitos  seed={args.seed}  device={args.device}")
    log.info(f"saida: {res}/    familias: {familias}    replicas: {args.seeds or '-'}")
    log.info("=" * 70)

    t_inicio = time.time()
    falhas: list[str] = []

    # --- Etapa 1: manifesto ------------------------------------------
    p_manifest = {"metadata_sha": file_sha(args.metadata) if Path(args.metadata).exists() else None,
                  "images_dir": args.images_dir, "task": args.task}
    stamp_manifest = manifest.with_suffix(".params.json")
    if precisa_rodar(stamp_manifest, manifest, p_manifest, args.force, log, "manifesto"):
        cmd = [py, str(SRC / "manifest.py"), "--metadata", args.metadata,
               "--images_dir", args.images_dir, "--task", args.task,
               "--out", str(manifest), "--check_files"]
        ok, _ = run_step(cmd, log, "1/6 manifesto")
        if not ok:
            log.error("manifesto invalido. Nada mais pode rodar.")
            return 1
        grava_stamp(stamp_manifest, p_manifest)
    else:
        log.info(f"1/6 reusando {manifest}")

    # --- Etapa 2: particao -------------------------------------------
    if args.split_mode == "fixed_paired" and not args.fixed_test_subjects:
        log.error("fixed_paired exige --fixed_test_subjects (ex.: 1 9 10 11).")
        return 2
    p_splits = {"manifest_sha": file_sha(manifest), "mode": args.split_mode,
                "k": args.k, "n_val_subjects": args.n_val_subjects,
                "val_strategy": args.val_strategy, "seed": args.seed,
                "fixed_test_subjects": args.fixed_test_subjects
                if args.split_mode == "fixed_paired" else None}
    stamp_splits = splits / "params.json"
    if precisa_rodar(stamp_splits, splits / "fold_00.csv", p_splits, args.force, log, "particao"):
        cmd = [py, str(SRC / "splits.py"), "--manifest", str(manifest),
               "--out_dir", str(splits), "--mode", args.split_mode,
               "--k", str(args.k), "--n_val_subjects", str(args.n_val_subjects),
               "--val_strategy", args.val_strategy, "--seed", str(args.seed)]
        if args.split_mode == "fixed_paired":
            cmd += ["--fixed_test_subjects"] + [str(s) for s in args.fixed_test_subjects]
        ok, _ = run_step(cmd, log, "2/6 particao por sujeito")
        if not ok:
            log.error("particao invalida (provavelmente fold degenerado). "
                      "Veja splits_summary.csv.")
            return 2
        grava_stamp(stamp_splits, p_splits)
    else:
        log.info(f"2/6 reusando folds em {splits}")

    todos_folds = sorted(splits.glob("fold_*.csv"))
    folds = todos_folds
    if args.only_folds is not None:
        folds = [f for f in folds if int(f.stem.split("_")[1]) in args.only_folds]
    log.info(f"folds a processar: {[f.stem for f in folds]}")

    # Pastas de resultado de folds que nao existem mais na particao atual
    # (sobra de uma particao anterior). Nao sao apagadas — podem ter modelos
    # que voce quer guardar — mas ficam fora da agregacao.
    atuais = {f.stem for f in todos_folds}
    orfas = sorted(d.name for d in res.glob("fold_*") if d.is_dir() and d.name not in atuais)
    if orfas:
        log.warning(f"pastas de folds fora da particao atual (ignoradas na agregacao): {orfas}")

    # --- Etapa 3: grid search ----------------------------------------
    # Cada par (fold, familia) roda em processo proprio. Se um falhar, os
    # outros seguem — e essa a razao de existir deste arquivo.
    log.info("3/6 grid search")
    for fold_csv in folds:
        fold = fold_csv.stem
        for fam in familias:
            out = res / fold / fam
            grid_json = CONFIGS / f"grid_{fam}.json"
            p_grid = {"fold_sha": file_sha(fold_csv), "grid_sha": file_sha(grid_json),
                      "seed": args.seed, "task": args.task}
            stamp_grid = out / "params.json"
            if not precisa_rodar(stamp_grid, out / "best_summary.csv", p_grid,
                                 args.force, log, f"{fold}/{fam}"):
                log.info(f"  {fold}/{fam} ja concluido, pulando")
                continue
            # Tira o vencedor antigo antes de refazer: se o grid novo falhar,
            # a agregacao nao pode pegar o resultado da configuracao anterior.
            for velho in ("best_summary.csv", "best_selection.json", "params.json"):
                (out / velho).unlink(missing_ok=True)
            cmd = [py, str(SRC / "grid_search.py"), "--fold", str(fold_csv),
                   "--grid", str(grid_json),
                   "--images_dir", args.images_dir, "--out_dir", str(out),
                   "--device", args.device, "--task", args.task,
                   "--seed", str(args.seed)]
            ok, _ = run_step(cmd, log, f"grid {fold}/{fam}")
            if ok:
                grava_stamp(stamp_grid, p_grid)
            else:
                falhas.append(f"grid {fold}/{fam}")

    # --- Etapa 4: replicas por semente -------------------------------
    # Retreina SO a configuracao vencedora de cada fold, com outras
    # sementes, mesmo protocolo (early stopping na validacao, teste uma vez).
    # Mede quanto do resultado por animal e ruido de treino.
    if not args.seeds:
        log.info("4/6 replicas por semente desligadas (use --seeds)")
    else:
        log.info(f"4/6 replicas por semente {args.seeds}")
        for fold_csv in folds:
            fold = fold_csv.stem
            for fam in familias:
                out = res / fold / fam
                if not (out / "best_selection.json").exists():
                    log.warning(f"  {fold}/{fam} sem vencedor, pulando replicas")
                    continue
                cmd = [py, str(SRC / "seed_replicates.py"), "--family_dir", str(out),
                       "--fold", str(fold_csv), "--images_dir", args.images_dir,
                       "--device", args.device, "--task", args.task,
                       "--seeds"] + [str(s) for s in args.seeds]
                if args.force:
                    cmd.append("--force")
                if args.keep_seed_models:
                    cmd.append("--keep_models")
                ok, _ = run_step(cmd, log, f"replicas {fold}/{fam}")
                if not ok:
                    falhas.append(f"replicas {fold}/{fam}")

    # --- Etapa 5: SHAP -----------------------------------------------
    if args.no_shap:
        log.info("5/6 SHAP desligado")
    else:
        log.info("5/6 SHAP")
        for fold_csv in folds:
            fold = fold_csv.stem
            for fam in familias:
                out = res / fold / fam
                sel = out / "best_selection.json"
                if not sel.exists():
                    log.warning(f"  {fold}/{fam} sem vencedor, pulando SHAP")
                    continue
                cand = json.loads(sel.read_text())["cand_dir"]
                cmd = [py, str(SRC / "explain_shap.py"), "--candidate", cand,
                       "--fold", str(fold_csv), "--images_dir", args.images_dir,
                       "--out_dir", str(out / "shap"), "--task", args.task,
                       "--device", args.device, "--seed", str(args.seed)]
                if args.with_background:
                    cmd.append("--with_background")
                ok, _ = run_step(cmd, log, f"shap {fold}/{fam}")
                if not ok:
                    falhas.append(f"shap {fold}/{fam}")

    # --- Etapa 6: agregacao ------------------------------------------
    # Sempre sobre TODOS os folds da particao atual (nao so os de
    # --only-folds), e nunca sobre pastas orfas.
    log.info("6/6 agregacao")
    ok, _ = run_step([py, str(SRC / "aggregate.py"), "--results_dir", str(res),
                      "--out_dir", str(res / "_tabelas"),
                      "--folds"] + sorted(atuais),
                     log, "agregacao")
    if not ok:
        falhas.append("agregacao")

    # --- Balanco -----------------------------------------------------
    total = timedelta(seconds=int(time.time() - t_inicio))
    log.info("=" * 70)
    if falhas:
        log.warning(f"CONCLUIDO COM {len(falhas)} FALHA(S) em {total}")
        for f in falhas:
            log.warning(f"  - {f}")
        log.warning("Rode de novo sem --force: as etapas que deram certo sao puladas.")
    else:
        log.info(f"CONCLUIDO SEM FALHAS em {total}")
    log.info(f"tabelas finais em {res}/_tabelas/    log completo em {res}/run.log")
    log.info("=" * 70)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
