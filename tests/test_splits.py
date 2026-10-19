from __future__ import annotations

import sys

import pandas as pd
import pytest

import splits
from splits import SplitError, assert_no_subject_leakage, make_splits


def _conjuntos(fold: pd.DataFrame) -> dict[str, set[int]]:
    return {n: set(fold.loc[fold.split == n, "subject_id"]) for n in ("train", "val", "test")}


def test_fixed_paired_rotaciona_teste_e_validacao(manifest_binary):
    folds, resumo = make_splits(manifest_binary, mode="fixed_paired",
                                fixed_test_subjects=[1, 9, 10, 11])
    assert len(folds) == 4
    esperado = [(1, 9), (9, 10), (10, 11), (11, 1)]
    for fold, (teste, val) in zip(folds, esperado):
        c = _conjuntos(fold)
        assert c["test"] == {teste}
        assert c["val"] == {val}
        assert not (c["train"] & (c["test"] | c["val"]))
    assert not resumo.degenerate.any()


@pytest.mark.parametrize("modo", ["loso", "group_kfold", "complete_subject_paired"])
def test_nenhum_sujeito_atravessa_conjuntos(manifest_multiclass, modo):
    folds, _ = make_splits(manifest_multiclass, mode=modo, k=4, n_val_subjects=2)
    for fold in folds:
        c = _conjuntos(fold)
        assert not (c["train"] & c["val"])
        assert not (c["train"] & c["test"])
        assert not (c["val"] & c["test"])
        assert c["train"] | c["val"] | c["test"] == set(range(1, 12))


def test_group_kfold_cobre_todos_os_rotulos_no_teste(manifest_multiclass):
    folds, resumo = make_splits(manifest_multiclass, mode="group_kfold", k=4, seed=7)
    assert not resumo.degenerate.any()
    for fold in folds:
        assert set(fold.loc[fold.split == "test", "label"]) == {0, 1, 2, 3}


def test_group_kfold_inviavel_acima_do_numero_de_portadores(manifest_multiclass):
    # severe tem 4 portadores: 5 folds com severe no teste sao impossiveis.
    with pytest.raises(SplitError):
        make_splits(manifest_multiclass, mode="group_kfold", k=5)


def test_vazamento_e_detectado(manifest_binary):
    fold = manifest_binary.copy()
    fold["split"] = "train"
    fold.loc[fold.index[0], "split"] = "test"      # mesmo sujeito em treino e teste
    with pytest.raises(SplitError, match="VAZAMENTO"):
        assert_no_subject_leakage(fold)


def test_cli_apaga_folds_de_particao_anterior(manifest_binary, tmp_path, monkeypatch):
    man = tmp_path / "manifest.csv"
    manifest_binary.to_csv(man, index=False)
    out = tmp_path / "splits"
    out.mkdir()
    for i in range(11):                               # sobra de um LOSO antigo
        (out / f"fold_{i:02d}.csv").write_text("velho")

    monkeypatch.setattr(sys, "argv", [
        "splits.py", "--manifest", str(man), "--out_dir", str(out),
        "--mode", "fixed_paired", "--fixed_test_subjects", "1", "9", "10", "11"])
    assert splits.main() == 0
    assert sorted(f.name for f in out.glob("fold_*.csv")) == \
        [f"fold_{i:02d}.csv" for i in range(4)]
