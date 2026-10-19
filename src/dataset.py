"""
ETAPA 3 — ACESSO AOS DADOS (PyTorch)
====================================

Responsabilidade unica: dado um CSV de fold e a raiz das imagens, entregar
tensores. Nao decide particao, nao conhece modelo.

Nenhuma imagem e copiada ou movida. O CSV de fold e a unica fonte de verdade
sobre quem esta em treino, validacao e teste — a mesma escolha de arquitetura
do pipeline do Furlanetti, e a razao pela qual trocar a politica de particao
nao exige mexer em nada aqui.

AUGMENTATION EM IMAGEM TERMICA
------------------------------
As imagens sao pseudo-coloridas: a camera FLIR renderiza um unico canal de
temperatura atraves de um colormap. Os tres canais RGB nao sao cores, sao uma
codificacao da temperatura. Consequencia pratica: qualquer transformacao que
mexa em cor (jitter, hue, saturacao, grayscale) destroi a variavel de
interesse. So transformacao geometrica e admissivel, e mesmo assim com
cuidado — flip vertical inverte a orientacao anatomica do animal na esteira.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ThermalFrameDataset(Dataset):
    """Quadros termograficos de um dos conjuntos de um fold.

    Devolve (imagem, rotulo, subject_id, filename). O subject_id acompanha
    cada amostra de proposito: e ele que permite agregar predicoes por animal
    na etapa de inferencia, que e onde a unidade de reamostragem tem de ser o
    sujeito e nao o quadro.
    """

    def __init__(self, fold_csv: str | Path, images_dir: str | Path,
                 split: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_csv(fold_csv)
        faltando = {"filename", "subject_id", "label", "split"}.difference(df.columns)
        if faltando:
            raise ValueError(f"CSV de fold sem colunas {sorted(faltando)}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"nenhuma linha com split='{split}' em {fold_csv}")

        self.labels = self.df["label"].astype(int).to_numpy()
        self.subjects = self.df["subject_id"].astype(int).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # `filepath` e o caminho que o manifesto resolveu contra o disco
        # (subpasta, extensao trocada). `filename` e so o nome declarado no
        # metadata; usa-lo quebraria com imagens organizadas em subpastas ou
        # com a versao sem fundo salva em .png. Fold antigo sem a coluna cai
        # para filename.
        rel = row["filepath"] if "filepath" in row.index and pd.notna(row["filepath"]) \
            else row["filename"]
        path = self.images_dir / str(rel)
        if not path.exists():
            raise FileNotFoundError(f"imagem nao encontrada: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"]), int(row["subject_id"]), str(row["filename"])


def build_transforms(image_size: tuple[int, int],
                     imagenet_norm: bool,
                     augment: bool,
                     hflip: bool = False):
    """Devolve (transform_treino, transform_avaliacao).

    image_size e (altura, largura).

    O QUE FOI DELIBERADAMENTE EXCLUIDO
    ----------------------------------
    `transforms.Grayscale`: destroi a informacao. Num colormap tipo jet, a
    luma ITU-R 601 (0.299R + 0.587G + 0.114B) NAO e monotonica na temperatura.
    Medindo sobre o colormap: 3 inversoes de direcao, 70 niveis de cinza
    recebendo mais de uma temperatura, e ate 6 temperaturas distintas
    colapsando no mesmo valor. Amarelo-esverdeado (T=0.63) vira cinza 0.863;
    vermelho (T=0.88, bem mais quente) vira 0.368. O animal quente le mais
    escuro que o morno.

    `transforms.ColorJitter`: brilho e contraste sobre um colormap sao a
    temperatura aparente. Perturba-los e perturbar a variavel de interesse.

    Flip vertical: inverte a orientacao anatomica do animal na baia.

    hflip fica opcional e desligado por padrao: a baia e aproximadamente
    simetrica e espelhar ajuda a descorrelacionar posicao, mas inverte o
    sentido da corrida, situacao que nunca ocorre no dado real.
    """
    treino = [transforms.Resize(image_size)]
    if augment:
        if hflip:
            treino.append(transforms.RandomHorizontalFlip())
        treino += [
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03),
                                    scale=(0.97, 1.03)),
        ]
    treino.append(transforms.ToTensor())

    avaliacao = [transforms.Resize(image_size), transforms.ToTensor()]

    if imagenet_norm:
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        treino.append(norm)
        avaliacao.append(norm)

    return transforms.Compose(treino), transforms.Compose(avaliacao)


def class_weights_from_split(dataset: ThermalFrameDataset,
                             n_classes: int,
                             scheme: str = "max_over_n") -> torch.Tensor:
    """Pesos de classe calculados SO no conjunto recebido.

    Tem de ser recalculado por fold. O Bernini reporta 1.0, 1.06, 1.77 e 8.72,
    que sao N_maior/N_classe sobre a base inteira — mas cada fold tem uma
    composicao diferente, entao reaproveitar aqueles numeros usa informacao de
    sujeitos que estao no teste daquele fold.

    scheme='max_over_n'  N_maior/N_classe        (esquema do Bernini)
    scheme='balanced'    N_total/(k*N_classe)    (esquema do sklearn)
    """
    # Contagem em numpy e so depois converte. torch.as_tensor sobre o array
    # do pandas compartilha memoria; quando o array vem marcado como
    # somente-leitura o PyTorch emite aviso de tensor nao gravavel. Nao ha
    # bug (nunca escrevemos aqui), mas o aviso polui o log de cada candidato.
    counts = torch.tensor(
        np.bincount(np.asarray(dataset.labels), minlength=n_classes),
        dtype=torch.float32,
    ).clamp(min=1.0)
    if scheme == "max_over_n":
        return counts.max() / counts
    if scheme == "balanced":
        return counts.sum() / (n_classes * counts)
    raise ValueError(f"esquema de peso '{scheme}' desconhecido")