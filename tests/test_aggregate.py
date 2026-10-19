from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aggregate import collect_arms, paired_by_subject, paired_comparison, seed_paired

# animal de teste de cada fold e o macro-F1 de cada braco nesse animal
ANIMAIS = {"fold_00": 1, "fold_01": 9, "fold_02": 10, "fold_03": 11}
F1 = {
    "bernini_cnn":             [0.50, 0.55, 0.60, 0.52],
    "resnet50_rf":             [0.70, 0.80, 0.75, 0.90],
    "resnet50_rf_linear_head": [0.72, 0.78, 0.74, 0.88],
}


def _grava(base: Path, prefix: str, sid: int, f1: float) -> None:
    base.mkdir(parents=True, exist_ok=True)
    linha = {"n": 100, "f1_macro": f1, "balanced_accuracy": f1 + 0.01}
    pd.DataFrame([linha]).to_csv(base / f"metrics_{prefix}.csv", index=False)
    pd.DataFrame([{"subject_id": sid, **linha}]).to_csv(
        base / f"metrics_{prefix}_by_subject.csv", index=False)


@pytest.fixture
def resultados(tmp_path: Path) -> Path:
    for i, (fold, sid) in enumerate(ANIMAIS.items()):
        _grava(tmp_path / fold / "bernini_cnn", "test", sid, F1["bernini_cnn"][i])
        rn = tmp_path / fold / "resnet50_rf"
        _grava(rn, "test", sid, F1["resnet50_rf"][i])
        _grava(rn, "test_linear_head", sid, F1["resnet50_rf_linear_head"][i])
        for s, ruido in ((1, -0.01), (2, 0.0), (3, 0.01)):
            _grava(tmp_path / fold / "bernini_cnn" / "seeds" / f"seed_{s}", "test", sid,
                   F1["bernini_cnn"][i] + ruido)
            _grava(rn / "seeds" / f"seed_{s}", "test", sid, F1["resnet50_rf"][i] + ruido)
    return tmp_path


def test_coleta_inclui_cabeca_linear(resultados):
    por_fold, por_sujeito = collect_arms(resultados, list(ANIMAIS))
    assert set(por_sujeito.arm) == set(F1)
    assert len(por_fold) == 4 * 3


def test_tabela_pareada_por_animal(resultados):
    _, por_sujeito = collect_arms(resultados, list(ANIMAIS))
    t = paired_by_subject(por_sujeito, ["f1_macro"])
    assert list(t.subject_id) == [1, 9, 10, 11]
    d = t["diff_bernini_cnn__vs__resnet50_rf"].tolist()
    esperado = [b - r for b, r in zip(F1["bernini_cnn"], F1["resnet50_rf"])]
    assert d == pytest.approx(esperado)


def test_comparacao_respeita_o_piso_do_p(resultados):
    _, por_sujeito = collect_arms(resultados, list(ANIMAIS))
    cmp = paired_comparison(por_sujeito, ["f1_macro"])
    r = cmp[(cmp.arm_a == "bernini_cnn") & (cmp.arm_b == "resnet50_rf")].iloc[0]
    assert (r.wins_a, r.wins_b, r.ties) == (0, 4, 0)
    assert r.p_floor == pytest.approx(0.125)
    assert r.sign_test_p == pytest.approx(0.125)     # 4x0 e exatamente o piso
    assert not r.p_reportavel


def test_folds_fora_da_lista_sao_ignorados(resultados):
    _grava(resultados / "fold_09" / "bernini_cnn", "test", 5, 0.99)   # orfa
    _, por_sujeito = collect_arms(resultados, list(ANIMAIS))
    assert 5 not in set(por_sujeito.subject_id)


def test_replicas_separam_diferenca_do_ruido(resultados):
    _, rep = collect_arms(resultados, list(ANIMAIS), seeds=True)
    assert sorted(rep.seed.unique()) == [1, 2, 3]
    sp = seed_paired(rep, ["f1_macro"])
    par = sp[(sp.arm_a == "bernini_cnn") & (sp.arm_b == "resnet50_rf")]
    assert len(par) == 4
    assert par.separados.all()
    assert (par.p_superior == 0.0).all()
    assert (par.diff_sobre_ruido > 10).all()
