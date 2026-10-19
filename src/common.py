"""
UTILITARIOS COMPARTILHADOS
==========================

O que antes estava copiado em varios modulos: silenciar avisos conhecidos,
resolver o dispositivo, fixar sementes e registrar o ambiente de execucao.
"""

from __future__ import annotations

import platform
import random
import subprocess
import warnings
from pathlib import Path


def silence_known_warnings() -> None:
    """Silencia dois avisos ruidosos e inofensivos.

    sklearn "y_pred contains classes not in y_true" e torch "non-writable
    tensors". Nenhum indica bug; warnings legitimos de outra origem continuam
    aparecendo.
    """
    warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
    warnings.filterwarnings("ignore", message=".*non-writable tensors.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")


def resolve_device(name: str):
    """'auto' vira cuda quando houver GPU, senao cpu."""
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Fixa as sementes e desliga o autotuning nao deterministico do cuDNN.

    `cudnn.benchmark=True` escolhe o algoritmo de convolucao mais rapido
    medindo em tempo de execucao, e a escolha pode mudar entre execucoes —
    o mesmo treino com a mesma semente da numeros diferentes. Custa um pouco
    de velocidade. Nao liga `torch.use_deterministic_algorithms`, que aborta
    em operacoes sem versao deterministica na GPU (backward do
    AdaptiveAvgPool2d da ResNet, por exemplo).
    """
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit(repo_dir: Path | None = None) -> str | None:
    """Hash do commit atual, com sufixo '-dirty' se houver alteracao local."""
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True).stdout.strip()
        sujo = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                              cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
        return sha + ("-dirty" if sujo else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info() -> dict:
    """Versoes que determinam o resultado, gravadas junto de cada treino."""
    import numpy as np
    import sklearn
    import torch
    import torchvision
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "git_commit": git_commit(),
    }
