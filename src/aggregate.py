"""
ETAPA 9 — AGREGACAO E INFERENCIA
================================

Junta os resultados de teste de todos os folds e bracos, resume, e compara os
bracos ANIMAL A ANIMAL.

Bracos: `bernini_cnn`, `resnet50_rf` e `resnet50_rf_linear_head` (a cabeca
linear da ResNet, avaliada no mesmo teste — o baseline contra o qual a
floresta precisa se justificar).

A UNIDADE DE INFERENCIA E O ANIMAL
----------------------------------
No desenho fixed_paired cada fold testa um unico animal, e os bracos sao
testados nos MESMOS animais. A comparacao certa e, portanto, pareada por
animal: para cada animal de teste, a metrica de cada braco e a diferenca.
`paired_by_subject.csv` e essa tabela; e ela que vai para a dissertacao.

O PISO DO P-VALOR
-----------------
Com n animais, o menor p-valor bilateral que o teste de sinal (ou o de
Wilcoxon) exato pode produzir e 2/2^n. Com n=4 isso da 0,125: mesmo que um
braco venca nos QUATRO animais, por qualquer margem, nada e significativo a
5%. Isso nao e um resultado do experimento — e aritmetica.

Por isso o que se reporta e: a direcao (em quantos animais cada braco
vence), o tamanho do efeito (diferenca media, minima e maxima) e, com
replicas por semente, se a diferenca supera o ruido de treino. O p do teste
de sinal vai na tabela como a probabilidade, sob H0, de uma consistencia tao
grande quanto a observada — com a marcacao `p_reportavel=False` quando o
piso passa de 0,05.

REPLICAS POR SEMENTE
--------------------
Se existir `fold_XX/<familia>/seeds/seed_*/` (gerado por
seed_replicates.py), as mesmas metricas sao lidas por replica e resumidas por
animal: media, desvio entre sementes, e a comparacao pareada entre bracos
usando as replicas. Uma diferenca entre bracos muito maior que o desvio
entre sementes e sistematica, nao sorte de inicializacao.

Media entre animais da peso IGUAL a cada animal, o certo para a pergunta
"funciona em animais novos?". Juntar todos os quadros num conjunto so
pesaria os animais pela quantidade de quadros.

Uso:
    python src/aggregate.py --results_dir resultados/binary \\
        --out_dir resultados/binary/_tabelas
"""


from __future__ import annotations

from common import silence_known_warnings
silence_known_warnings()

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

LINEAR_SUFFIX = "_linear_head"


def p_floor(n: int) -> float:
    """Menor p-valor bilateral alcancavel pelo teste de sinal / Wilcoxon exato."""
    return 2.0 / (2 ** n) if n > 0 else float("nan")


# --------------------------------------------------------------------------
# Coleta
# --------------------------------------------------------------------------
def list_folds(results_dir: Path, folds: list[str] | None) -> list[str]:
    if folds:
        return sorted(folds)
    return sorted(d.name for d in results_dir.glob("fold_*") if d.is_dir())


def collect_selection(results_dir: Path, folds: list[str]) -> pd.DataFrame:
    """Um `best_summary.csv` por (fold, familia): o que foi selecionado e como."""
    linhas = []
    for fold in folds:
        for f in sorted((results_dir / fold).glob("*/best_summary.csv")):
            df = pd.read_csv(f)
            if df.empty:
                continue
            r = df.iloc[0].to_dict()
            r["fold"] = fold
            r["family"] = f.parent.name
            linhas.append(r)
    if not linhas:
        raise FileNotFoundError(f"nenhum best_summary.csv sob {results_dir} para {folds}")
    return pd.DataFrame(linhas)


def _arm_files(base: Path, family: str):
    """(braco, prefixo) disponiveis numa pasta de resultado."""
    yield family, "test"
    if (base / "metrics_test_linear_head.csv").exists():
        yield family + LINEAR_SUFFIX, "test_linear_head"


def collect_arms(results_dir: Path, folds: list[str], seeds: bool = False):
    """Metricas de teste por fold e por animal, de cada braco.

    seeds=False le o vencedor do grid (fold/familia/). seeds=True le as
    replicas (fold/familia/seeds/seed_*/) e acrescenta a coluna `seed`.
    Devolve (por_fold, por_sujeito).
    """
    por_fold, por_sujeito = [], []
    for fold in folds:
        for fam_dir in sorted(p for p in (results_dir / fold).iterdir() if p.is_dir()):
            if seeds:
                bases = [(d, int(d.name.split("_", 1)[1]))
                         for d in sorted((fam_dir / "seeds").glob("seed_*")) if d.is_dir()]
            else:
                bases = [(fam_dir, None)]
            for base, seed in bases:
                for arm, prefix in _arm_files(base, fam_dir.name):
                    f_fold = base / f"metrics_{prefix}.csv"
                    f_subj = base / f"metrics_{prefix}_by_subject.csv"
                    if not f_fold.exists():
                        continue
                    chave = {"fold": fold, "arm": arm}
                    if seed is not None:
                        chave["seed"] = seed
                    por_fold.append({**chave, **pd.read_csv(f_fold).iloc[0].to_dict()})
                    if f_subj.exists():
                        for r in pd.read_csv(f_subj).to_dict("records"):
                            por_sujeito.append({**chave, **r})
    return pd.DataFrame(por_fold), pd.DataFrame(por_sujeito)


# --------------------------------------------------------------------------
# Resumos
# --------------------------------------------------------------------------
def aggregate_metrics(por_fold: pd.DataFrame) -> pd.DataFrame:
    """Media, desvio e extremos de cada metrica entre folds, por braco."""
    ignorar = {"fold", "arm", "seed", "n"}
    cols = [c for c in por_fold.columns
            if c not in ignorar and pd.api.types.is_numeric_dtype(por_fold[c])]
    linhas = []
    for arm, g in por_fold.groupby("arm"):
        for c in cols:
            v = pd.to_numeric(g[c], errors="coerce")
            linhas.append({"arm": arm, "metric": f"test_{c}", "mean": v.mean(),
                           "std": v.std(ddof=1), "min": v.min(),
                           "median": v.median(), "max": v.max(),
                           "n_folds": int(v.notna().sum())})
    return pd.DataFrame(linhas)


def paired_by_subject(por_sujeito: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Tabela animal x braco, com as diferencas pareadas. Formato longo em `metric`."""
    blocos = []
    fold_de = por_sujeito.groupby("subject_id")["fold"].first()
    for metric in metrics:
        if metric not in por_sujeito.columns:
            continue
        tab = por_sujeito.pivot_table(index="subject_id", columns="arm", values=metric)
        bracos = sorted(tab.columns)
        for a, b in itertools.combinations(bracos, 2):
            tab[f"diff_{a}__vs__{b}"] = tab[a] - tab[b]
        tab.insert(0, "fold", fold_de.reindex(tab.index))
        tab.insert(0, "metric", metric)
        blocos.append(tab.reset_index())
    return pd.concat(blocos, ignore_index=True) if blocos else pd.DataFrame()


def paired_comparison(por_sujeito: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Compara bracos animal a animal: direcao, tamanho do efeito, teste de sinal."""
    from scipy.stats import binomtest, wilcoxon

    linhas = []
    for metric in metrics:
        if metric not in por_sujeito.columns:
            continue
        tab = por_sujeito.pivot_table(index="subject_id", columns="arm", values=metric)
        for a, b in itertools.combinations(sorted(tab.columns), 2):
            par = tab[[a, b]].dropna()
            n = len(par)
            if n < 2:
                continue
            d = (par[a] - par[b]).to_numpy()
            wins_a, wins_b = int((d > 0).sum()), int((d < 0).sum())
            n_eff = wins_a + wins_b                     # empates saem do teste de sinal
            piso = p_floor(n_eff)
            linha = {
                "metric": metric, "arm_a": a, "arm_b": b, "n_subjects": n,
                "mean_a": float(par[a].mean()), "mean_b": float(par[b].mean()),
                "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                "min_diff": float(d.min()), "max_diff": float(d.max()),
                "wins_a": wins_a, "wins_b": wins_b, "ties": n - n_eff,
                "sign_test_p": float(binomtest(wins_a, n_eff, 0.5).pvalue)
                if n_eff else float("nan"),
                "p_floor": piso,
                "p_reportavel": bool(piso <= 0.05),
            }
            if piso <= 0.05:
                try:
                    linha["wilcoxon_p"] = float(wilcoxon(d).pvalue)
                except ValueError:
                    linha["wilcoxon_p"] = float("nan")
            else:
                linha["wilcoxon_p"] = float("nan")
                linha["nota"] = (f"com {n_eff} animais o menor p possivel e {piso:.3f}; "
                                 f"reporte a direcao ({wins_a}x{wins_b}) e a diferenca "
                                 f"por animal, nao significancia.")
            linhas.append(linha)
    return pd.DataFrame(linhas)


def seed_summary(rep: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Por animal e braco: media e dispersao da metrica entre sementes."""
    blocos = []
    for metric in metrics:
        if metric not in rep.columns:
            continue
        g = rep.groupby(["fold", "subject_id", "arm"])[metric]
        s = g.agg(n_seeds="count", mean="mean", std="std", min="min", max="max").reset_index()
        s.insert(0, "metric", metric)
        blocos.append(s)
    return pd.concat(blocos, ignore_index=True) if blocos else pd.DataFrame()


def seed_paired(rep: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Por animal: diferenca entre bracos comparada ao ruido entre sementes.

    p_superior  fracao dos pares (replica de a, replica de b) em que a supera
                b (empate conta meio). 1,0 = toda replica de a bate toda
                replica de b naquele animal.
    separados   True quando as faixas [min, max] dos dois bracos nao se
                sobrepoem.
    diff_sobre_ruido  |diferenca das medias| / desvio combinado entre
                sementes. Bem acima de 2: a diferenca nao e sorte de
                inicializacao.
    """
    linhas = []
    for metric in metrics:
        if metric not in rep.columns:
            continue
        for (fold, sid), g in rep.groupby(["fold", "subject_id"]):
            por_braco = {arm: gg[metric].dropna().to_numpy() for arm, gg in g.groupby("arm")}
            for a, b in itertools.combinations(sorted(por_braco), 2):
                va, vb = por_braco[a], por_braco[b]
                if not len(va) or not len(vb):
                    continue
                diff = float(va.mean() - vb.mean())
                sa = float(va.std(ddof=1)) if len(va) > 1 else float("nan")
                sb = float(vb.std(ddof=1)) if len(vb) > 1 else float("nan")
                ruido = float(np.sqrt((sa ** 2 + sb ** 2) / 2))
                comp = va[:, None] - vb[None, :]
                linhas.append({
                    "metric": metric, "fold": fold, "subject_id": int(sid),
                    "arm_a": a, "arm_b": b, "n_seeds_a": len(va), "n_seeds_b": len(vb),
                    "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                    "std_a": sa, "std_b": sb, "mean_diff": diff,
                    "diff_sobre_ruido": abs(diff) / ruido if ruido > 0 else float("nan"),
                    "p_superior": float((comp > 0).mean() + 0.5 * (comp == 0).mean()),
                    "separados": bool(va.min() > vb.max() or vb.min() > va.max()),
                })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Etapa 9: agregacao entre folds.")
    ap.add_argument("--results_dir", default="resultados")
    ap.add_argument("--out_dir", default="resultados/_tabelas")
    ap.add_argument("--folds", nargs="*", default=None,
                    help="Folds a agregar (ex.: fold_00 fold_01). Padrao: todas as "
                         "pastas fold_*. O run_all.py passa os folds da particao atual.")
    ap.add_argument("--metrics", nargs="*", default=["f1_macro", "balanced_accuracy"],
                    help="Metricas da comparacao pareada; a primeira vai para o terminal.")
    args = ap.parse_args()

    res = Path(args.results_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    folds = list_folds(res, args.folds)

    collect_selection(res, folds).to_csv(out / "best_by_fold.csv", index=False)

    por_fold, por_sujeito = collect_arms(res, folds)
    if por_fold.empty or por_sujeito.empty:
        print(f"[ERRO] nenhum metrics_test*.csv sob {res} para {folds}")
        return 1
    ag = aggregate_metrics(por_fold)
    ag.to_csv(out / "aggregate_metrics.csv", index=False)
    por_sujeito.to_csv(out / "metrics_by_subject_all_arms.csv", index=False)

    tabela = paired_by_subject(por_sujeito, args.metrics)
    tabela.to_csv(out / "paired_by_subject.csv", index=False)
    cmp = paired_comparison(por_sujeito, args.metrics)
    cmp.to_csv(out / "paired_comparison.csv", index=False)

    principal = args.metrics[0]
    n_animais = por_sujeito.subject_id.nunique()
    print("=" * 78)
    print(f"AGREGACAO  —  {len(folds)} folds, {n_animais} animais de teste, "
          f"bracos: {sorted(por_sujeito.arm.unique())}")
    print("=" * 78)
    print(f"{principal} por animal de teste:")
    t = tabela[tabela.metric == principal].drop(columns="metric")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    if not cmp.empty:
        c = cmp[cmp.metric == principal]
        print(c[["arm_a", "arm_b", "mean_diff", "min_diff", "max_diff",
                 "wins_a", "wins_b", "sign_test_p", "p_reportavel"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\npiso do p com {n_animais} animais: {p_floor(n_animais):.4f}", end="")
        if p_floor(n_animais) > 0.05:
            print("  -> nenhum resultado pode ser significativo a 5%. Reporte a\n"
                  "   direcao e a diferenca por animal (e o ruido entre sementes).")
        else:
            print()

    # --- replicas por semente -----------------------------------------
    _, rep = collect_arms(res, folds, seeds=True)
    if rep.empty:
        print("\n(sem replicas por semente; rode run_all.py com --seeds para medir o "
              "ruido de treino)")
    else:
        rep.to_csv(out / "seed_replicates_by_subject.csv", index=False)
        resumo = seed_summary(rep, args.metrics)
        resumo.to_csv(out / "seed_summary_by_subject.csv", index=False)
        sp = seed_paired(rep, args.metrics)
        sp.to_csv(out / "seed_paired_by_subject.csv", index=False)

        print()
        print("-" * 78)
        print(f"REPLICAS POR SEMENTE — {principal} (media ± desvio entre sementes)")
        print("-" * 78)
        r = resumo[resumo.metric == principal]
        for (fold, sid), g in r.groupby(["fold", "subject_id"]):
            celulas = "   ".join(f"{a}: {m:.3f}±{s:.3f} (n={k})" for a, m, s, k in
                                 zip(g.arm, g["mean"], g["std"], g.n_seeds))
            print(f"  animal {sid:>2} ({fold})  {celulas}")
        if not sp.empty:
            s = sp[sp.metric == principal]
            print()
            print(s[["subject_id", "arm_a", "arm_b", "mean_diff", "diff_sobre_ruido",
                     "p_superior", "separados"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 78)
    print(f"[ok] tabelas em {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
