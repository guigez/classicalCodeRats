"""
ETAPA 5 — METRICAS
==================

Metricas multiclasse, agregacao por sujeito e intervalo de confianca.

A REGRA QUE GOVERNA ESTE MODULO
-------------------------------
A unidade de reamostragem e o SUJEITO, nunca o quadro. Quadros consecutivos a
~7 fps sao quase duplicatas: reamostra-los produz intervalos de confianca
absurdamente estreitos, porque o bootstrap acredita ter milhares de
observacoes independentes quando tem onze animais. `bootstrap_ci` so aceita
reamostragem por sujeito, de proposito.

Metricas por quadro continuam sendo reportadas — sao o desempenho operacional
do classificador. O que nao se pode fazer e construir inferencia sobre elas.
"""


from __future__ import annotations

from common import silence_known_warnings
silence_known_warnings()

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list[str] | None = None) -> dict:
    """Metricas por quadro.

    Macro (nao weighted) e o padrao de leitura aqui: com `severe` em 4,4% dos
    quadros, a media ponderada e dominada pelas classes faceis e esconde
    exatamente o que interessa.
    """
    labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    n = len(class_names) if class_names else len(labels)
    todos = list(range(n))

    out: dict = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=todos, average=avg, zero_division=0)
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=todos, average=None, zero_division=0)
    for i in todos:
        nome = class_names[i] if class_names else f"class{i}"
        out[f"precision_{nome}"] = float(p[i])
        out[f"recall_{nome}"] = float(r[i])
        out[f"f1_{nome}"] = float(f[i])
        out[f"support_{nome}"] = int(sup[i])
        # Suporte zero significa metrica indefinida contada como 0 pelo
        # zero_division. Marcar explicitamente evita que uma media macro
        # calculada sobre classe ausente passe por resultado legitimo.
        if sup[i] == 0:
            out[f"undefined_{nome}"] = True
    return out


def confusion_dataframe(y_true, y_pred, class_names: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm,
                        index=[f"real_{c}" for c in class_names],
                        columns=[f"pred_{c}" for c in class_names])


def predictions_dataframe(filenames, subjects, y_true, y_pred, probs) -> pd.DataFrame:
    """Predicoes por quadro, com o sujeito preservado.

    O subject_id tem de sobreviver ate aqui: sem ele nao ha como agregar por
    animal nem reamostrar corretamente.
    """
    df = pd.DataFrame({
        "filename": list(filenames),
        "subject_id": np.asarray(subjects).astype(int),
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": np.asarray(y_pred).astype(int),
    })
    probs = np.asarray(probs)
    for k in range(probs.shape[1]):
        df[f"prob_{k}"] = probs[:, k]
    df["correct"] = (df.y_pred == df.y_true).astype(int)
    return df


def per_subject_metrics(preds: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Desempenho de cada animal do teste, separadamente.

    Util porque a media entre sujeitos esconde o caso em que um animal
    sustenta o resultado inteiro e os outros vao mal.
    """
    linhas = []
    for sid, g in preds.groupby("subject_id"):
        m = compute_metrics(g.y_true.values, g.y_pred.values, class_names)
        linhas.append({"subject_id": int(sid), **m})
    return pd.DataFrame(linhas)


def bootstrap_ci(preds: pd.DataFrame, metric_fn, class_names: list[str],
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """IC percentil reamostrando SUJEITOS com reposicao.

    Reamostra a lista de animais, nao a lista de quadros. Com poucos sujeitos
    por fold o intervalo sai largo — e isso e o resultado honesto, nao um
    defeito do metodo. Um intervalo estreito aqui seria sinal de que a
    reamostragem esta errada.
    """
    rng = np.random.default_rng(seed)
    sujeitos = preds.subject_id.unique()
    if len(sujeitos) < 2:
        return {"point": float(metric_fn(preds.y_true.values, preds.y_pred.values,
                                         class_names)),
                "lo": np.nan, "hi": np.nan, "n_subjects": int(len(sujeitos)),
                "note": "IC indefinido com menos de 2 sujeitos"}

    ponto = float(metric_fn(preds.y_true.values, preds.y_pred.values, class_names))
    amostras = []
    por_sujeito = {s: g for s, g in preds.groupby("subject_id")}
    for _ in range(n_boot):
        escolhidos = rng.choice(sujeitos, size=len(sujeitos), replace=True)
        g = pd.concat([por_sujeito[s] for s in escolhidos], ignore_index=True)
        amostras.append(metric_fn(g.y_true.values, g.y_pred.values, class_names))
    amostras = np.asarray(amostras, dtype=float)
    return {
        "point": ponto,
        "lo": float(np.percentile(amostras, 100 * alpha / 2)),
        "hi": float(np.percentile(amostras, 100 * (1 - alpha / 2))),
        "n_subjects": int(len(sujeitos)),
    }


def macro_f1(y_true, y_pred, class_names) -> float:
    _, _, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))),
        average="macro", zero_division=0)
    return float(f)


def balanced_acc(y_true, y_pred, class_names) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def save_all(preds: pd.DataFrame, class_names: list[str], out_dir: Path,
             prefix: str = "test") -> dict:
    """Grava predicoes, matriz de confusao, metricas e IC por sujeito."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / f"predictions_{prefix}.csv", index=False)

    m = compute_metrics(preds.y_true.values, preds.y_pred.values, class_names)
    pd.DataFrame([m]).to_csv(out_dir / f"metrics_{prefix}.csv", index=False)
    confusion_dataframe(preds.y_true.values, preds.y_pred.values,
                        class_names).to_csv(out_dir / f"confusion_{prefix}.csv")
    per_subject_metrics(preds, class_names).to_csv(
        out_dir / f"metrics_{prefix}_by_subject.csv", index=False)

    # Com um unico sujeito de teste (fixed_paired, LOSO) o IC por sujeito nao
    # existe. Gravar um CSV so com NaN dava a impressao de que havia um IC a
    # reportar; nesse desenho a inferencia e feita entre folds, na tabela
    # pareada por animal de aggregate.py.
    ci_csv = out_dir / f"ci_{prefix}_subject_bootstrap.csv"
    if preds.subject_id.nunique() < 2:
        ci_csv.unlink(missing_ok=True)
        return m
    ic = {f"macro_f1_{k}": v for k, v in
          bootstrap_ci(preds, macro_f1, class_names).items()}
    ic.update({f"balanced_acc_{k}": v for k, v in
               bootstrap_ci(preds, balanced_acc, class_names).items()})
    pd.DataFrame([ic]).to_csv(ci_csv, index=False)
    return {**m, **ic}
