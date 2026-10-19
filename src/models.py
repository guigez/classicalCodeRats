"""
ETAPA 4 — MODELOS (PyTorch)
===========================

Duas familias:

  bernini_cnn    CNN do artigo do Bernini, reimplementada em PyTorch.
  resnet50_rf    ResNet50 pre-treinada com a layer4 destravada, seguida de
                 random forest sobre as features extraidas.

O modulo constroi e treina; nao decide particao nem calcula metrica final.

SOBRE A ARQUITETURA DO BERNINI
------------------------------
A Figura 8 do artigo tem legenda e tabela em contradicao. A legenda fala em
5 convolucoes (32, 64, 128, 256, 512 filtros) e densa de 1024; a tabela fala
em 4 convolucoes (2, 4, 8, 16) e densa de 128. A tabela e a correta — a
aritmetica dos parametros fecha nela e nao na legenda:

    conv 3->2   3*3*3*2  + 2  =     56
    conv 2->4   3*3*2*4  + 4  =     76
    conv 4->8   3*3*4*8  + 8  =    296
    conv 8->16  3*3*8*16 + 16 =  1.168
    bn2d(16)    2*16          =     32
    densa       6688*128 +128 = 856.192      6688 = 16 * 22 * 19
    bn1d(128)   2*128         =    256
    saida       128*4    + 4  =    516
                               --------
                                858.592     = total declarado no artigo

A versao da legenda passaria de 1,5 milhao so nas convolucoes. As larguras
ficam configuraveis via `widths` para permitir testar a variante da legenda
como ablacao.

SOBRE A COMBINACAO COM RANDOM FOREST
------------------------------------
Random forest nao e diferenciavel: nao ha gradiente que volte dela para a
layer4. Treinar a layer4 E usar floresta como classificador so faz sentido em
dois estagios, e e assim que esta implementado:

  1. treina a ResNet50 (layer4 + fc destravadas) com cross-entropy;
  2. congela, extrai o vetor de 2048 dimensoes do avgpool para cada quadro;
  3. ajusta a floresta sobre esses vetores.

Consequencia que vale registrar: o passo 1 molda as features para serem
linearmente separaveis, porque e isso que a cabeca linear otimiza. A floresta
entra num espaco ja preparado para separacao linear e costuma empatar com a
propria cabeca linear. Ela se justifica como braco de comparacao com viés
indutivo distinto, nao como ganho presumido. Por isso `predict_linear_head`
existe: a cabeca linear sai de graca e serve de terceiro braco.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models as tvm

BERNINI_INPUT_SIZE = (360, 306)   # (altura, largura), resolucao nativa
RESNET_INPUT_SIZE = (224, 224)


# ==========================================================================
# Modelo 1 — CNN do Bernini
# ==========================================================================
class BerniniCNN(nn.Module):
    """CNN do artigo, em PyTorch.

    Blocos: [conv 3x3 'same' -> ReLU -> avgpool 2x2] x4, BatchNorm2d, flatten,
    densa -> ReLU -> BatchNorm1d, densa de saida.

    O flatten e calculado dinamicamente a partir do tamanho de entrada, em vez
    de fixado em 6688. Fixar amarra o modelo a 360x306 e quebra em silencio se
    alguem mudar a resolucao.
    """

    def __init__(self, n_classes: int = 4,
                 input_size: tuple[int, int] = BERNINI_INPUT_SIZE,
                 widths: Sequence[int] = (2, 4, 8, 16),
                 hidden_dim: int = 128,
                 dropout: float = 0.0):
        super().__init__()

        blocos: list[nn.Module] = []
        canais_entrada = 3
        for w in widths:
            blocos += [
                nn.Conv2d(canais_entrada, w, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ]
            canais_entrada = w
        blocos.append(nn.BatchNorm2d(canais_entrada))
        self.features = nn.Sequential(*blocos)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size)
            n_flat = int(np.prod(self.features(dummy).shape[1:]))
        self.n_flat = n_flat

        cabeca: list[nn.Module] = [
            nn.Linear(n_flat, hidden_dim),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(hidden_dim),
        ]
        if dropout > 0:
            cabeca.append(nn.Dropout(dropout))
        cabeca.append(nn.Linear(hidden_dim, n_classes))
        self.classifier = nn.Sequential(*cabeca)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def elastic_net_penalty(model: nn.Module, l1: float = 1e-5, l2: float = 1e-4) -> torch.Tensor:
    """Penalidade L1+L2 sobre os pesos convolucionais, como no artigo.

    O artigo aplica L1 (1e-5) e L2 (1e-4) as matrizes de peso das camadas
    convolucionais. Nao da para usar `weight_decay` do otimizador para isso:
    weight_decay e L2 e incide sobre TODOS os parametros, inclusive vieses e
    ganhos de BatchNorm. Somar a penalidade a loss explicitamente e o
    equivalente fiel.
    """
    p1 = torch.zeros((), device=next(model.parameters()).device)
    p2 = torch.zeros((), device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            p1 = p1 + m.weight.abs().sum()
            p2 = p2 + m.weight.pow(2).sum()
    return l1 * p1 + l2 * p2


# ==========================================================================
# Modelo 2 — ResNet50 (layer4 treinavel) + random forest
# ==========================================================================
class ResNet50Layer4(nn.Module):
    """ResNet50 pre-treinada com apenas layer4 e fc destravadas.

    Expoe `forward` (logits, para o treino supervisionado) e `features`
    (vetor de 2048 dimensoes do avgpool, para alimentar a floresta).
    """

    FEATURE_DIM = 2048

    def __init__(self, n_classes: int = 4, pretrained: bool = True,
                 train_layer4: bool = True):
        super().__init__()
        pesos = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = tvm.resnet50(weights=pesos)

        for p in self.backbone.parameters():
            p.requires_grad = False
        if train_layer4:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        # Substituida DEPOIS do congelamento: a Linear nova ja nasce treinavel.
        self.backbone.fc = nn.Linear(self.FEATURE_DIM, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(self.features(x))

    def freeze_all(self) -> None:
        """Congela tudo. Chamar antes da extracao de features."""
        for p in self.backbone.parameters():
            p.requires_grad = False


@torch.no_grad()
def extract_features(model: ResNet50Layer4, loader, device: torch.device):
    """Extrai features de um DataLoader.

    Devolve (X, y, subjects, filenames). O modelo entra em eval() e o loader
    TEM de usar a transformacao deterministica — extrair feature de imagem
    aumentada faz a floresta aprender o ruido da augmentation.
    """
    model.eval()
    X, y, s, f = [], [], [], []
    for imgs, labels, subs, names in loader:
        X.append(model.features(imgs.to(device)).cpu().numpy())
        y.append(labels.numpy())
        s.append(subs.numpy())
        f.extend(list(names))
    return np.concatenate(X), np.concatenate(y), np.concatenate(s), f


def build_random_forest(n_estimators: int = 100,
                        max_depth: int | None = None,
                        min_samples_leaf: int = 1,
                        max_features: str | float = "sqrt",
                        class_weight: str | None = "balanced",
                        seed: int = 42,
                        n_jobs: int = -1) -> RandomForestClassifier:
    """Floresta que recebe as features da ResNet50.

    ATENCAO — o erro out-of-bag NAO e valido neste dataset. O OOB reamostra
    QUADROS, e quadros consecutivos a ~7 fps sao quase duplicatas do mesmo
    animal: o bootstrap e a amostra OOB compartilham o mesmo sujeito, e o erro
    OOB fica absurdamente otimista pelo mesmo motivo que o split por quadro
    ficava. `oob_score` fica desligado de proposito. Selecione hiperparametro
    nos SUJEITOS de validacao do fold.

    Padroes conforme a Tabela 4.6 da dissertacao: 100 arvores,
    class_weight="balanced", sem reducao de dimensionalidade.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        oob_score=False,
        bootstrap=True,
        random_state=seed,
        n_jobs=n_jobs,
    )


# ==========================================================================
# Fabrica
# ==========================================================================
MODEL_REGISTRY = ("bernini_cnn", "resnet50_rf")


def create_model(config: dict, n_classes: int = 4) -> nn.Module:
    """Constroi o modelo torch a partir de um dicionario de configuracao."""
    nome = config.get("model_name", "bernini_cnn")

    if nome == "bernini_cnn":
        return BerniniCNN(
            n_classes=n_classes,
            input_size=tuple(config.get("input_size", BERNINI_INPUT_SIZE)),
            widths=tuple(config.get("widths", (2, 4, 8, 16))),
            hidden_dim=int(config.get("hidden_dim", 128)),
            dropout=float(config.get("dropout", 0.0)),
        )

    if nome == "resnet50_rf":
        return ResNet50Layer4(
            n_classes=n_classes,
            pretrained=bool(config.get("pretrained", True)),
            train_layer4=bool(config.get("train_layer4", True)),
        )

    raise ValueError(f"modelo '{nome}' desconhecido. Opcoes: {MODEL_REGISTRY}")


def input_size_for(config: dict) -> tuple[int, int]:
    """Resolucao de entrada exigida pelo modelo.

    Os dois modelos rodam em resolucoes diferentes por natureza — 360x306 na
    resolucao nativa do Bernini, 224x224 no que a ResNet espera. Isso confunde
    arquitetura com resolucao na comparacao entre os dois. Se a comparacao
    precisar ser limpa, force a mesma resolucao nos dois e registre a escolha.
    """
    if config.get("model_name") == "resnet50_rf":
        return tuple(config.get("input_size", RESNET_INPUT_SIZE))
    return tuple(config.get("input_size", BERNINI_INPUT_SIZE))


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_rf_pipeline(n_estimators: int = 100,
                     class_weight: str | None = "balanced",
                     seed: int = 42,
                     **rf_kwargs) -> Pipeline:
    """StandardScaler + RandomForest, como na Tabela 4.6 da dissertacao.

    O escalonamento e irrelevante para arvores — elas so olham para a ordem
    dos valores, e StandardScaler e uma transformacao monotonica. Fica aqui
    por fidelidade a especificacao, nao por efeito.

    O que NAO e irrelevante: o scaler tem de ser ajustado apenas nas features
    de TREINO e depois aplicado a validacao e teste. Encapsular no Pipeline
    garante isso — `fit` no pipeline inteiro nunca ve os outros conjuntos.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", build_random_forest(n_estimators=n_estimators,
                                   class_weight=class_weight,
                                   seed=seed, **rf_kwargs)),
    ])
