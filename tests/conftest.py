"""Fixtures compartilhadas. Os modulos de src/ se importam pelo nome curto
(`from manifest import ...`), entao src/ entra no sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from manifest import CLASS_ORDER, TASKS  # noqa: E402


def _manifest(task: str) -> pd.DataFrame:
    """Manifesto sintetico com a mesma estrutura de cobertura do dataset real.

    11 sujeitos; 1, 9, 10 e 11 tem as 4 classes; 4 para antes de heavy; os
    demais nao chegam a severe.
    """
    linhas = []
    for s in range(1, 12):
        if s in (1, 9, 10, 11):
            classes = CLASS_ORDER
        elif s == 4:
            classes = ["warm-up", "moderate"]
        else:
            classes = ["warm-up", "moderate", "heavy"]
        for c in classes:
            for i in range(3):
                linhas.append({"filename": f"s{s}_{c}_{i}.jpg",
                               "filepath": f"s{s}_{c}_{i}.jpg",
                               "subject_id": s, "class_name": c,
                               "label": TASKS[task][c]})
    return pd.DataFrame(linhas)


@pytest.fixture
def manifest_multiclass() -> pd.DataFrame:
    return _manifest("multiclass")


@pytest.fixture
def manifest_binary() -> pd.DataFrame:
    return _manifest("binary")
