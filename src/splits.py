"""
ETAPA 2 — PARTICAO POR SUJEITO
==============================

Responsabilidade unica: transformar o manifesto em N arquivos de particao,
onde a unidade de agrupamento e o INDIVIDUO. Um sujeito que aparece no teste
nao aparece em treino nem em validacao, com nenhum de seus quadros.

Duas garantias, ambas verificadas em codigo e nao por confianca:

  1. Nenhum sujeito atravessa conjuntos. Violacao interrompe a execucao.
  2. Fold degenerado (rotulo ausente do teste ou do treino) e detectado e
     reportado. Por padrao interrompe a execucao.

A garantia 2 existe porque a falha silenciosa e a perigosa: se o teste nao tem
nenhum quadro de uma classe, o recall dessa classe fica indefinido, o macro-F1
vira media de k-1 numeros se passando por k, e o resultado agregado sai
inflado sem nenhum sinal de erro.

IMPORTANTE: degenerescencia e julgada sobre `label` (o espaco de rotulos da
tarefa), nao sobre `class_name`. No agrupamento binario, um fold sem quadros
`severe` continua valido desde que tenha algum positivo, porque severe e heavy
compartilham o mesmo rotulo. Julgar por class_name rejeitaria folds sadios.
As contagens por class_name continuam no resumo, como informacao: um fold cujo
positivo e 100% heavy e 0% severe e valido, mas voce quer saber disso.

Modos de particao:

  loso              cada sujeito e teste exatamente uma vez.
  group_kfold       sujeitos divididos em k grupos disjuntos (k folds).
  repeated_holdout  R sorteios independentes de sujeitos de teste. E a
                    traducao direta das "10 replicas" do pipeline de
                    referencia para o nivel de sujeito. Atencao: as replicas
                    se sobrepoem, entao NAO sao observacoes independentes
                    para teste pareado.

Uso:
    python src/splits.py --manifest manifest.csv --mode loso \
        --n_val_subjects 2 --out_dir splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from manifest import CLASS_ORDER

MODES = ("loso", "group_kfold", "repeated_holdout", "complete_subject_loso",
         "complete_subject_paired", "fixed_paired")


class SplitError(RuntimeError):
    """Erro que invalida a particao. Sempre interrompe a execucao."""


# --------------------------------------------------------------------------
# Perfil dos sujeitos
# --------------------------------------------------------------------------
def label_display_names(manifest: pd.DataFrame) -> dict[int, str]:
    """Nome legivel de cada rotulo, ex.: {0: 'warm-up+moderate', 1: 'heavy+severe'}."""
    nomes: dict[int, str] = {}
    for lab, grp in manifest.groupby("label"):
        classes = [c for c in CLASS_ORDER if c in set(grp["class_name"])]
        nomes[int(lab)] = "+".join(classes)
    return nomes


def subject_label_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Sujeito x rotulo. E esta tabela que governa as decisoes de particao."""
    return pd.crosstab(manifest["subject_id"], manifest["label"])


def subject_class_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Sujeito x classe original. Serve para relatorio, nao para decisao."""
    presentes = [c for c in CLASS_ORDER if c in set(manifest["class_name"])]
    return (
        pd.crosstab(manifest["subject_id"], manifest["class_name"])
        .reindex(columns=presentes, fill_value=0)
    )


def rarest_column(table: pd.DataFrame):
    """Coluna presente no menor numero de sujeitos.

    Nao e a coluna com menos quadros: o que restringe a particao e em quantos
    INDIVIDUOS a classe aparece, porque o sujeito e a unidade indivisivel.
    """
    return min(table.columns, key=lambda c: (table[c] > 0).sum())


# --------------------------------------------------------------------------
# Geracao dos conjuntos de teste
# --------------------------------------------------------------------------
def max_feasible_k(label_table: pd.DataFrame) -> tuple[int, int]:
    """Maior k possivel, e o rotulo que impoe esse limite.

    Um fold de teste so e valido se contiver ao menos um sujeito portador de
    cada rotulo. Como o sujeito e indivisivel, o rotulo com menos portadores
    define o teto: com 4 sujeitos carregando `severe`, nao existem 5 folds
    validos, faca o que fizer com a semente.
    """
    limitante = min(label_table.columns, key=lambda l: int((label_table[l] > 0).sum()))
    return int((label_table[limitante] > 0).sum()), limitante


def stratified_group_folds(label_table: pd.DataFrame, k: int, seed: int) -> list[list[int]]:
    """Divide sujeitos em k folds garantindo cobertura de rotulos em cada um.

    Embaralhar e cortar em k nao funciona: nada impede que dois portadores da
    classe rara caiam no mesmo fold, deixando outro fold sem nenhum. O
    procedimento aqui e deliberado:

      1. percorre os rotulos do mais raro para o mais comum;
      2. para cada fold ainda sem portador daquele rotulo, aloca um portador
         livre;
      3. distribui os sujeitos restantes no fold com menos quadros, o que
         aproxima os tamanhos dos conjuntos de teste.

    O passo 1 e o que importa: alocar o rotulo raro primeiro consome os
    sujeitos mais restritos enquanto ainda ha liberdade de escolha.
    """
    rng = np.random.default_rng(seed)
    labels = list(label_table.columns)
    portadores = {l: label_table.index[label_table[l] > 0].tolist() for l in labels}

    folds: list[list[int]] = [[] for _ in range(k)]
    atribuido: set[int] = set()

    for l in sorted(labels, key=lambda x: len(portadores[x])):
        livres = [int(s) for s in rng.permutation(
            [s for s in portadores[l] if s not in atribuido]
        )]
        sem_cobertura = [
            i for i in range(k)
            if not any(label_table.loc[s, l] > 0 for s in folds[i])
        ]
        for i in sem_cobertura:
            if not livres:
                break
            s = livres.pop()
            folds[i].append(s)
            atribuido.add(s)

    quadros = label_table.sum(axis=1)
    sobras = [int(s) for s in rng.permutation(
        [s for s in label_table.index if s not in atribuido]
    )]
    for s in sobras:
        i = min(range(k), key=lambda j: sum(quadros[x] for x in folds[j]))
        folds[i].append(s)
        atribuido.add(s)

    return [sorted(f) for f in folds]


def complete_subjects(label_table: pd.DataFrame) -> list[int]:
    """Sujeitos que carregam TODOS os rotulos da tarefa.

    Sao os unicos em que a tarefa e bem-posta como teste: um teste sem alguma
    classe torna o recall dela indefinido. No dataset de 4 classes esses sao
    justamente os portadores de severe (1, 9, 10, 11), porque severe e a
    classe mais escassa e quem a tem tem as demais.
    """
    return [int(s) for s in label_table.index if (label_table.loc[s] > 0).all()]


def paired_from_fixed(test_subjects: list[int]) -> list[tuple[list[int], list[int]]]:
    """Emparelhamento rotativo a partir de uma lista EXPLICITA de sujeitos.

    Ignora quem e ou nao 'completo' — usa exatamente os sujeitos passados, na
    ordem dada, com teste=s_i e validacao=s_{(i+1) mod n}. Existe para o caso
    em que voce quer o MESMO split em varios cenarios de agrupamento, cujos
    rotulos mudariam a nocao de 'completo' e portanto a particao automatica.

    E responsabilidade de quem chama garantir que esses sujeitos facam sentido
    para a tarefa (ex.: que o teste nao fique sem alguma classe). O diagnostico
    de degenerescencia ainda roda e vai acusar se ficar.
    """
    n = len(test_subjects)
    if n < 2:
        raise SplitError("fixed_paired precisa de ao menos 2 sujeitos de teste.")
    return [([test_subjects[i]], [test_subjects[(i + 1) % n]]) for i in range(n)]


def paired_complete_folds(label_table: pd.DataFrame) -> list[tuple[list[int], list[int]]]:
    """Emparelhamento rotativo de sujeitos completos: teste e validacao juntos.

    Com os completos ordenados [c0, c1, ..., c_{n-1}], o fold i tem
    teste=c_i e validacao=c_{(i+1) mod n}. Assim cada completo e teste
    exatamente uma vez e validacao exatamente uma vez, e a validacao SEMPRE
    contem todas as classes (por ser um sujeito completo) — o que elimina o
    caso de validacao sem severe.

    Custo: em cada fold, 2 dos portadores de severe ficam fora do treino (um
    no teste, outro na validacao). Sobram 2 portadores treinando a classe
    rara, contra 3 no modo complete_subject_loso.
    """
    completos = complete_subjects(label_table)
    if len(completos) < 2:
        raise SplitError(
            f"complete_subject_paired precisa de ao menos 2 sujeitos completos; "
            f"ha {len(completos)}: {completos}."
        )
    n = len(completos)
    return [([completos[i]], [completos[(i + 1) % n]]) for i in range(n)]


def make_test_groups(
    label_table: pd.DataFrame,
    mode: str,
    k: int = 4,
    n_replicates: int = 10,
    n_test_subjects: int = 1,
    seed: int = 42,
) -> list[list[int]]:
    """Devolve, para cada fold, a lista de sujeitos que compoem o teste.

    Excecao: complete_subject_paired e tratado antes de chegar aqui, em
    make_splits, porque fixa teste E validacao juntos.
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(int(s) for s in label_table.index)

    if mode == "loso":
        return [[s] for s in subjects]

    if mode == "complete_subject_loso":
        # Um sujeito COMPLETO por fold, sozinho no teste. Garante que todo
        # teste contenha as 4 classes — nenhum fold degenera. Custo: os
        # animais que pararam cedo (sem severe/heavy) nunca sao testados, e a
        # estimativa passa a valer para animais de alto desempenho, nao para a
        # populacao inteira.
        completos = complete_subjects(label_table)
        if not completos:
            raise SplitError(
                "nenhum sujeito carrega todas as classes; "
                "complete_subject_loso e impossivel nesta tarefa."
            )
        return [[s] for s in completos]

    if mode == "group_kfold":
        if k > len(subjects):
            raise SplitError(f"k={k} maior que o numero de sujeitos ({len(subjects)}).")
        k_max, limitante = max_feasible_k(label_table)
        if k > k_max:
            raise SplitError(
                f"k={k} e inviavel: o rotulo {limitante} tem apenas {k_max} "
                f"sujeito(s) portador(es), entao no maximo {k_max} folds podem "
                f"conter esse rotulo no teste. Use --k {k_max} ou reagrupe as "
                f"classes (--task binary no manifesto)."
            )
        return stratified_group_folds(label_table, k, seed)

    if mode == "repeated_holdout":
        if n_test_subjects >= len(subjects):
            raise SplitError("n_test_subjects deve ser menor que o total de sujeitos.")
        return [
            sorted(int(x) for x in rng.choice(subjects, size=n_test_subjects, replace=False))
            for _ in range(n_replicates)
        ]

    raise SplitError(f"modo '{mode}' desconhecido. Opcoes: {MODES}")


def choose_validation(
    pool: list[int],
    n_val: int,
    label_table: pd.DataFrame,
    strategy: str,
    rng: np.random.Generator,
) -> list[int]:
    """Escolhe os sujeitos de validacao dentro do pool de treino.

    strategy='random'    sorteio simples.
    strategy='coverage'  garante que a validacao contenha o rotulo mais raro,
                         se algum sujeito do pool o carregar. Sem isso, a
                         validacao pode nao ter a classe minoritaria e a
                         selecao de hiperparametro passa a otimizar uma
                         metrica que ignora justamente a classe dificil.

    Nota: n_val=1 significa ajustar hiperparametro em um unico animal. E
    estatisticamente ruidoso. O modulo permite, mas avisa.
    """
    if n_val <= 0:
        return []
    if n_val >= len(pool):
        raise SplitError(f"n_val_subjects={n_val} nao cabe no pool de treino ({len(pool)}).")

    if strategy == "random":
        return sorted(int(x) for x in rng.choice(pool, size=n_val, replace=False))

    if strategy == "coverage":
        sub = label_table.loc[pool]
        raro = rarest_column(sub)
        portadores = [int(s) for s in sub.index if sub.loc[s, raro] > 0]
        nao_portadores = [int(s) for s in pool if s not in portadores]

        escolhidos: list[int] = []
        if portadores:
            escolhidos.append(int(rng.choice(portadores)))

        # Completa preferindo NAO-portadores. Levar varios portadores para a
        # validacao esvazia o treino da classe rara: com 4 portadores de
        # `severe` repartidos entre teste, validacao e treino, cada portador a
        # mais na validacao e um a menos com que o modelo pode aprender a
        # classe dificil. Um basta para tornar a metrica de validacao definida.
        reserva = ([s for s in rng.permutation(nao_portadores)]
                   + [s for s in rng.permutation([p for p in portadores if p not in escolhidos])])
        faltam = n_val - len(escolhidos)
        escolhidos += [int(x) for x in reserva[:faltam]]
        return sorted(escolhidos)

    raise SplitError(f"estrategia de validacao '{strategy}' desconhecida.")


# --------------------------------------------------------------------------
# Montagem e verificacao de um fold
# --------------------------------------------------------------------------
def build_fold(manifest: pd.DataFrame, train: list[int], val: list[int], test: list[int]) -> pd.DataFrame:
    """Marca cada quadro com seu conjunto, a partir da atribuicao de sujeitos."""
    destino = {s: "train" for s in train}
    destino.update({s: "val" for s in val})
    destino.update({s: "test" for s in test})

    fold = manifest.copy()
    fold["split"] = fold["subject_id"].map(destino)
    if fold["split"].isna().any():
        orfaos = sorted(fold.loc[fold["split"].isna(), "subject_id"].unique())
        raise SplitError(f"sujeitos sem conjunto atribuido: {orfaos}")
    return fold


def assert_no_subject_leakage(fold: pd.DataFrame) -> None:
    """A garantia central deste modulo. Falha alto se for violada."""
    conjuntos = {
        nome: set(fold.loc[fold["split"] == nome, "subject_id"])
        for nome in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = conjuntos[a] & conjuntos[b]
        if inter:
            raise SplitError(
                f"VAZAMENTO: sujeitos {sorted(inter)} aparecem em '{a}' e em '{b}'."
            )


def fold_diagnosis(fold: pd.DataFrame, labels: list[int], label_names: dict[int, str]) -> dict:
    """Descreve o fold e decide se ele e utilizavel.

    Degenerado = algum ROTULO ausente do teste (metrica indefinida) ou do
    treino (o modelo nunca pode aprende-lo). As contagens por class_name
    entram como informacao, nao como criterio.
    """
    labs = {
        nome: set(fold.loc[fold["split"] == nome, "label"])
        for nome in ("train", "val", "test")
    }
    falta_teste = [label_names[l] for l in labels if l not in labs["test"]]
    falta_treino = [label_names[l] for l in labels if l not in labs["train"]]
    falta_val = [label_names[l] for l in labels if l not in labs["val"]] if labs["val"] else []

    d = {
        "train_subjects": sorted(set(fold.loc[fold["split"] == "train", "subject_id"])),
        "val_subjects": sorted(set(fold.loc[fold["split"] == "val", "subject_id"])),
        "test_subjects": sorted(set(fold.loc[fold["split"] == "test", "subject_id"])),
        "n_train": int((fold["split"] == "train").sum()),
        "n_val": int((fold["split"] == "val").sum()),
        "n_test": int((fold["split"] == "test").sum()),
        "missing_in_test": falta_teste,
        "missing_in_train": falta_treino,
        "missing_in_val": falta_val,
        "degenerate": bool(falta_teste or falta_treino),
    }
    # Contagem de SUJEITOS portadores por rotulo, alem da contagem de quadros.
    # E o numero que revela treino faminto: 700 quadros de `severe` vindos de
    # um unico animal nao ensinam a classe, ensinam aquele animal.
    d["min_train_carriers"] = min(
        int(fold.loc[(fold["split"] == "train") & (fold["label"] == l), "subject_id"].nunique())
        for l in labels
    )

    # contagens por rotulo (criterio) e por classe original (informacao)
    for l in labels:
        for nome in ("train", "val", "test"):
            sel = (fold["split"] == nome) & (fold["label"] == l)
            d[f"{nome}_label{l}"] = int(sel.sum())
            d[f"{nome}_subj_label{l}"] = int(fold.loc[sel, "subject_id"].nunique())
    for c in [c for c in CLASS_ORDER if c in set(fold["class_name"])]:
        for nome in ("train", "val", "test"):
            sel = (fold["split"] == nome) & (fold["class_name"] == c)
            d[f"{nome}_{c}"] = int(sel.sum())
    return d


# --------------------------------------------------------------------------
# Orquestracao
# --------------------------------------------------------------------------
def make_splits(
    manifest: pd.DataFrame,
    mode: str = "loso",
    k: int = 4,
    n_replicates: int = 10,
    n_test_subjects: int = 1,
    n_val_subjects: int = 1,
    val_strategy: str = "coverage",
    fixed_test_subjects: list[int] | None = None,
    seed: int = 42,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Gera todos os folds e o resumo diagnostico."""
    label_table = subject_label_table(manifest)
    subjects = label_table.index.tolist()
    labels = sorted(int(l) for l in label_table.columns)
    nomes = label_display_names(manifest)

    # complete_subject_paired e fixed_paired fixam teste E validacao juntos;
    # os demais modos escolhem o teste aqui e sorteiam a validacao no loop.
    if mode == "complete_subject_paired":
        pares = paired_complete_folds(label_table)
        grupos_teste = [t for t, _ in pares]
        vals_fixas = [v for _, v in pares]
    elif mode == "fixed_paired":
        if not fixed_test_subjects:
            raise SplitError("fixed_paired exige fixed_test_subjects (lista de sujeitos).")
        pares = paired_from_fixed(list(fixed_test_subjects))
        grupos_teste = [t for t, _ in pares]
        vals_fixas = [v for _, v in pares]
    else:
        grupos_teste = make_test_groups(
            label_table, mode, k=k, n_replicates=n_replicates,
            n_test_subjects=n_test_subjects, seed=seed,
        )
        vals_fixas = None

    folds, linhas = [], []
    for i, teste in enumerate(grupos_teste):
        # Semente propria por fold: torna a escolha de validacao reprodutivel
        # e independente da ordem em que os folds sao gerados.
        rng = np.random.default_rng(seed + 1000 * (i + 1))
        pool = [s for s in subjects if s not in teste]
        if vals_fixas is not None:
            val = vals_fixas[i]
        else:
            val = choose_validation(pool, n_val_subjects, label_table, val_strategy, rng)
        treino = [s for s in pool if s not in val]

        fold = build_fold(manifest, treino, val, teste)
        assert_no_subject_leakage(fold)               # garantia 1
        fold["fold"] = i

        diag = fold_diagnosis(fold, labels, nomes)    # garantia 2
        diag["fold"] = i
        folds.append(fold)
        linhas.append(diag)

    resumo = pd.DataFrame(linhas)
    colunas = ["fold", "degenerate", "test_subjects", "val_subjects",
               "n_train", "n_val", "n_test", "missing_in_test",
               "missing_in_train", "missing_in_val", "train_subjects"]
    resto = [c for c in resumo.columns if c not in colunas]
    return folds, resumo[colunas + resto]


def print_report(resumo: pd.DataFrame, n_val_subjects: int, label_names: dict[int, str]) -> None:
    print("=" * 78)
    print("PARTICAO POR SUJEITO")
    print("=" * 78)
    print("rotulos: " + ",  ".join(f"{l}={n}" for l, n in sorted(label_names.items())))
    print("-" * 78)
    for _, r in resumo.iterrows():
        marca = "DEGENERADO" if r["degenerate"] else "ok"
        print(f"fold {int(r['fold']):>2d} [{marca:^10s}] "
              f"teste={r['test_subjects']} val={r['val_subjects']}  "
              f"quadros {r['n_train']}/{r['n_val']}/{r['n_test']}")
        if r["missing_in_test"]:
            print(f"          rotulos ausentes do TESTE:  {r['missing_in_test']}")
        if r["missing_in_train"]:
            print(f"          rotulos ausentes do TREINO: {r['missing_in_train']}")
        if r["missing_in_val"]:
            print(f"          rotulos ausentes da VAL:    {r['missing_in_val']}")
        if r.get("min_train_carriers", 99) == 1:
            print(f"          [aviso] algum rotulo aparece no treino em apenas "
                  f"1 sujeito")
    print("-" * 78)

    n_bad = int(resumo["degenerate"].sum())
    print(f"folds gerados: {len(resumo)}   utilizaveis: {len(resumo) - n_bad}   "
          f"degenerados: {n_bad}")
    famintos = int((resumo.get("min_train_carriers", pd.Series(dtype=int)) == 1).sum())
    if famintos:
        print(f"[aviso] {famintos} fold(s) treinam algum rotulo a partir de um unico "
              f"sujeito. O modelo aprende o animal, nao a classe.")
    if n_val_subjects == 1:
        print("[aviso] validacao com 1 sujeito: a selecao de hiperparametro fica "
              "ruidosa. Considere n_val_subjects >= 2 ou CV interna.")
    print("=" * 78)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Etapa 2: particao por sujeito.")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out_dir", default="splits")
    ap.add_argument("--mode", default="group_kfold", choices=MODES)
    ap.add_argument("--k", type=int, default=4, help="Numero de folds em group_kfold.")
    ap.add_argument("--n_replicates", type=int, default=10, help="Replicas em repeated_holdout.")
    ap.add_argument("--n_test_subjects", type=int, default=1, help="Sujeitos de teste em repeated_holdout.")
    ap.add_argument("--n_val_subjects", type=int, default=2)
    ap.add_argument("--val_strategy", default="coverage", choices=["random", "coverage"])
    ap.add_argument("--fixed_test_subjects", type=int, nargs="*", default=None,
                    help="Lista de sujeitos de teste para o modo fixed_paired. "
                         "Ex.: --fixed_test_subjects 1 9 10 11")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_degenerate", action="store_true",
                    help="Grava folds degenerados em vez de interromper.")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    obrigatorias = {"filename", "subject_id", "class_name", "label"}
    faltando = obrigatorias.difference(manifest.columns)
    if faltando:
        print(f"[ERRO] manifesto sem colunas {sorted(faltando)}. "
              f"Rode src/manifest.py antes.", file=sys.stderr)
        return 1

    try:
        folds, resumo = make_splits(
            manifest, mode=args.mode, k=args.k, n_replicates=args.n_replicates,
            n_test_subjects=args.n_test_subjects, n_val_subjects=args.n_val_subjects,
            val_strategy=args.val_strategy, fixed_test_subjects=args.fixed_test_subjects,
            seed=args.seed,
        )
    except SplitError as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        return 1

    print_report(resumo, args.n_val_subjects, label_display_names(manifest))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(out_dir / "splits_summary.csv", index=False)

    n_bad = int(resumo["degenerate"].sum())
    if n_bad and not args.allow_degenerate:
        print(f"\n[ERRO] {n_bad} fold(s) degenerado(s). Nada foi gravado.\n"
              f"       Resolva a montagem da tarefa (ex.: --task binary no manifesto,\n"
              f"       ou --mode group_kfold --k 4) ou passe --allow_degenerate\n"
              f"       para gravar assim mesmo. Diagnostico em "
              f"{out_dir / 'splits_summary.csv'}", file=sys.stderr)
        return 2

    # Apaga folds de uma particao anterior antes de gravar. Sem isso, trocar
    # de um modo com 11 folds para um com 4 deixava fold_04..fold_10 antigos
    # na pasta, e o run_all.py (que faz glob em fold_*.csv) os treinava junto.
    antigos = sorted(out_dir.glob("fold_*.csv"))
    for f in antigos:
        f.unlink()
    if len(antigos) > len(folds):
        print(f"[aviso] {len(antigos) - len(folds)} fold(s) de uma particao "
              f"anterior removido(s) de {out_dir}/")
    for i, fold in enumerate(folds):
        fold.to_csv(out_dir / f"fold_{i:02d}.csv", index=False)
    print(f"[ok] {len(folds)} folds gravados em {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
