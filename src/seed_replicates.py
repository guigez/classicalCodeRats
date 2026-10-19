"""
ETAPA 7b — REPLICAS POR SEMENTE
===============================

Retreina a configuracao VENCEDORA de um (fold, familia) com outras sementes,
pelo mesmo protocolo do grid: early stopping e floresta ajustados na
validacao, teste avaliado uma vez por replica.

POR QUE
-------
No desenho fixed_paired cada numero por animal vem de UM treino. Uma
diferenca de 0,2 em macro-F1 entre familias pode ser sistematica ou sorte de
inicializacao e de ordem dos lotes. Repetir o treino com R sementes, no
mesmo split, separa as duas coisas: se a diferenca entre familias e muito
maior que a variacao entre sementes do mesmo animal, ela nao e sorte. Isso
nao aumenta o numero de animais — a generalizacao continua apoiada em n=4 —
mas tira o ruido de treino da conta.

O que NAO muda entre replicas: hiperparametros (os do vencedor), split,
sujeitos de validacao e teste. A selecao do vencedor nao e refeita: as
replicas medem a estabilidade da configuracao escolhida, nao do grid.

Saida: <family_dir>/seeds/seed_<s>/ com os mesmos arquivos de teste que o
grid grava (metrics_test*.csv, predictions_test*.csv, ...), lidos por
aggregate.py. model.pt e rf_head.joblib sao apagados ao fim de cada replica,
a menos de --keep_models: R sementes x 4 folds x 2 familias x 4 tarefas de
ResNet50 passam facil de 10 GB.

Uso:
    python src/seed_replicates.py --family_dir resultados/binary/fold_00/resnet50_rf \\
        --fold resultados/binary/splits/fold_00.csv --images_dir imagens \\
        --task binary --seeds 1 2 3 4 5
"""

from __future__ import annotations

from common import resolve_device, silence_known_warnings
silence_known_warnings()

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from grid_search import evaluate_on_test
from manifest import TASKS, class_names_for
from train import run_candidate


def _resolve(registrado: str, family_dir: Path, relativo: Path) -> Path:
    """Caminho gravado em best_selection.json, ou o equivalente sob family_dir.

    best_selection.json guarda caminhos relativos ao diretorio de onde o grid
    rodou. Se o projeto foi movido ou o comando roda de outro lugar, cai para
    a mesma estrutura dentro de family_dir.
    """
    p = Path(registrado)
    return p if p.exists() else family_dir / relativo


def load_winner(family_dir: Path) -> tuple[dict, Path, Path, dict]:
    """(selecao, net_dir, cand_dir, rf_config) do vencedor do grid."""
    sel = json.loads((family_dir / "best_selection.json").read_text())
    net_reg, cand_reg = Path(sel["net_dir"]), Path(sel["cand_dir"])
    net_dir = _resolve(sel["net_dir"], family_dir, Path(net_reg.name))
    rel_cand = Path(net_reg.name) if cand_reg == net_reg else Path(net_reg.name) / cand_reg.name
    cand_dir = _resolve(sel["cand_dir"], family_dir, rel_cand)
    rf_cfg_path = cand_dir / "rf_config.json"
    rf_cfg = json.loads(rf_cfg_path.read_text()) if rf_cfg_path.exists() else {}
    return sel, net_dir, cand_dir, rf_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Etapa 7b: replicas do vencedor por semente.")
    ap.add_argument("--family_dir", required=True,
                    help="Pasta fold_XX/<familia> com best_selection.json.")
    ap.add_argument("--fold", required=True, help="CSV do fold.")
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--task", default="multiclass", choices=list(TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--force", action="store_true", help="Refaz replicas ja concluidas.")
    ap.add_argument("--keep_models", action="store_true",
                    help="Mantem model.pt e rf_head.joblib de cada replica.")
    args = ap.parse_args()

    device = resolve_device(args.device)
    class_names = class_names_for(args.task)
    n_classes = len(class_names)
    family_dir = Path(args.family_dir)

    sel, net_dir, _, rf_cfg = load_winner(family_dir)
    config = json.loads((net_dir / "config_used.json").read_text())
    config.pop("_env", None)
    eh_floresta = config.get("model_name") == "resnet50_rf"

    # O que define uma replica, alem da semente. Se mudar (novo vencedor,
    # novo split), as replicas antigas deixam de valer e sao refeitas.
    base_params = {
        "winner": sel.get("best_config_id"),
        "config": {k: v for k, v in config.items() if k not in ("seed", "config_id")},
        "rf_config": rf_cfg,
        "fold_sha": hashlib.sha1(Path(args.fold).read_bytes()).hexdigest()[:16],
    }

    print(f"[{family_dir}] vencedor={sel.get('best_config_id')}  sementes={args.seeds}  "
          f"dispositivo={device}")
    seeds_dir = family_dir / "seeds"
    resumos, falhas = [], []
    for s in args.seeds:
        out = seeds_dir / f"seed_{s}"
        stamp = out / "params.json"
        params = {**base_params, "seed": s}
        if (not args.force and (out / "metrics_test.csv").exists() and stamp.exists()
                and json.loads(stamp.read_text()) == params):
            print(f"  seed {s}: ja concluida, pulando")
            continue
        if out.exists():
            shutil.rmtree(out)          # replica antiga ou incompleta, gerada por este script

        c = {**config, "seed": s, "config_id": f"seed_{s}"}
        print(f"  seed {s}: treinando...")
        try:
            r = run_candidate(args.fold, args.images_dir, c, out, device, n_classes,
                              class_names, [rf_cfg] if eh_floresta else None)[0]
            m = evaluate_on_test(out, Path(r["cand_dir"]), args.fold, args.images_dir,
                                 c, device, n_classes, class_names, out)
        except Exception as e:                       # uma semente nao derruba as outras
            print(f"    ERRO: {e!r}")
            falhas.append(s)
            continue

        if not args.keep_models:
            (out / "model.pt").unlink(missing_ok=True)
            for j in out.glob("rf_*/rf_head.joblib"):
                j.unlink()
        stamp.write_text(json.dumps(params, indent=2))
        resumos.append({"seed": s, "best_epoch": r.get("best_epoch"),
                        "val_f1_macro": r.get("val_f1_macro"),
                        "test_f1_macro": m.get("f1_macro"),
                        "test_balanced_accuracy": m.get("balanced_accuracy")})
        print(f"    best_epoch={r.get('best_epoch')}  val_f1={r.get('val_f1_macro', float('nan')):.4f}"
              f"  teste_f1={m.get('f1_macro', float('nan')):.4f}")

    if resumos:
        novo = pd.DataFrame(resumos)
        resumo_csv = seeds_dir / "seeds_summary.csv"
        if resumo_csv.exists():
            antigo = pd.read_csv(resumo_csv)
            novo = pd.concat([antigo[~antigo.seed.isin(novo.seed)], novo]).sort_values("seed")
        novo.to_csv(resumo_csv, index=False)

    if falhas:
        print(f"[ERRO] sementes com falha: {falhas}", file=sys.stderr)
        return 1
    print("[ok] replicas concluidas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
