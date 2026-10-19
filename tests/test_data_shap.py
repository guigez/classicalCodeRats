"""Testes que dependem de torch/torchvision (dataset e fundo do SHAP)."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from dataset import ThermalFrameDataset  # noqa: E402
from metrics import predictions_dataframe, save_all  # noqa: E402


def test_dataset_usa_filepath_resolvido(tmp_path):
    # imagem numa subpasta e com extensao diferente da declarada no metadata
    (tmp_path / "imgs" / "rato_01").mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(tmp_path / "imgs" / "rato_01" / "a.png")
    fold = tmp_path / "fold.csv"
    pd.DataFrame([{"filename": "a.jpg", "filepath": "rato_01/a.png", "subject_id": 1,
                   "label": 0, "split": "train"}]).to_csv(fold, index=False)
    ds = ThermalFrameDataset(fold, tmp_path / "imgs", "train")
    img, label, sid, nome = ds[0]
    assert img.size == (8, 8) and (label, sid, nome) == (0, 1, "a.jpg")


class _FakeDataset:
    """Ordenado por sujeito e classe, como o fold real."""

    def __init__(self):
        self.df = pd.DataFrame([{"subject_id": s, "label": l}
                                for s in (1, 2, 3) for l in (0, 1) for _ in range(50)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return torch.tensor([float(r.subject_id), float(r.label)]), int(r.label)


def test_fundo_do_shap_cobre_todas_as_celulas():
    pytest.importorskip("matplotlib")
    from explain_shap import sample_background
    bg = sample_background(_FakeDataset(), k=6, seed=0, device="cpu")
    celulas = {(int(x[0]), int(x[1])) for x in bg}
    assert celulas == {(s, l) for s in (1, 2, 3) for l in (0, 1)}


def test_sem_ic_com_um_unico_sujeito(tmp_path):
    preds = predictions_dataframe(["a", "b", "c"], [1, 1, 1], [0, 1, 1], [0, 1, 0],
                                  [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    (tmp_path / "ci_test_subject_bootstrap.csv").write_text("antigo")
    m = save_all(preds, ["a", "b"], tmp_path)
    assert "macro_f1_lo" not in m
    assert not (tmp_path / "ci_test_subject_bootstrap.csv").exists()
    assert (tmp_path / "metrics_test_by_subject.csv").exists()
