"""
ETAPA 6 — TREINO DE UM CANDIDATO
================================

Opera um unico par (fold, configuracao). Nao percorre grid nem folds — isso e
responsabilidade de grid_search.py.

Dois caminhos, conforme o modelo:

  bernini_cnn   treino supervisionado -> logits -> argmax.

  resnet50_rf   dois estagios, porque random forest nao e diferenciavel:
                (1) treina ResNet50 com layer4 destravada e cabeca linear,
                    por cross-entropy;
                (2) congela, extrai o vetor de 2048 dimensoes do avgpool e
                    ajusta StandardScaler + RandomForest sobre ele.
                A cabeca linear do estagio 1 fica registrada como terceiro
                braco de comparacao: ela sai de graca e e o baseline honesto
                contra o qual a floresta precisa se justificar.

O conjunto de TESTE nunca e tocado aqui. Epoca, hiperparametro e floresta sao
todos escolhidos na validacao.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common import environment_info, seed_everything
from dataset import ThermalFrameDataset, build_transforms, class_weights_from_split
from metrics import compute_metrics, predictions_dataframe
from models import (create_model, elastic_net_penalty, extract_features,
                    input_size_for, make_rf_pipeline, trainable_parameters)


def build_loaders(fold_csv, images_dir, config, n_classes):
    """Loaders dos tres conjuntos. Treino com augmentation, resto sem."""
    size = input_size_for(config)
    imagenet = config.get("model_name") == "resnet50_rf"
    tf_treino, tf_eval = build_transforms(
        size, imagenet_norm=imagenet,
        augment=bool(config.get("augment", True)),
        hflip=bool(config.get("hflip", False)),
    )

    ds = {
        "train": ThermalFrameDataset(fold_csv, images_dir, "train", tf_treino),
        # Copia do treino SEM augmentation: e desta que saem as features para
        # a floresta. Extrair feature de imagem aumentada faz a floresta
        # aprender o ruido da augmentation.
        "train_eval": ThermalFrameDataset(fold_csv, images_dir, "train", tf_eval),
        "val": ThermalFrameDataset(fold_csv, images_dir, "val", tf_eval),
        "test": ThermalFrameDataset(fold_csv, images_dir, "test", tf_eval),
    }

    bs = int(config.get("batch_size", 32))
    nw = int(config.get("num_workers", 4))
    pin = torch.cuda.is_available()
    g = torch.Generator(); g.manual_seed(int(config.get("seed", 42)))

    loaders = {
        "train": DataLoader(ds["train"], batch_size=bs, shuffle=True, generator=g,
                            num_workers=nw, pin_memory=pin, drop_last=True),
    }
    for k in ("train_eval", "val", "test"):
        loaders[k] = DataLoader(ds[k], batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=pin)
    return ds, loaders


@torch.no_grad()
def infer(model, loader, device):
    """Probabilidades, rotulos, sujeitos e nomes de arquivo."""
    model.eval()
    P, Y, S, F = [], [], [], []
    for x, y, s, f in loader:
        logits = model(x.to(device))
        P.append(torch.softmax(logits, 1).cpu().numpy())
        Y.append(y.numpy()); S.append(s.numpy()); F.extend(list(f))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(S), F


def train_torch(model, loaders, config, device, n_classes, class_names, out_dir):
    """Laco de treino com early stopping na validacao."""
    lr = float(config.get("learning_rate", 1e-3))
    epochs = int(config.get("epochs", 60))
    patience = int(config.get("patience", 12))
    metrica = str(config.get("selection_metric", "f1_macro"))

    criterio = nn.CrossEntropyLoss(
        weight=class_weights_from_split(loaders["train"].dataset, n_classes,
                                        config.get("class_weight_scheme", "max_over_n")).to(device)
        if config.get("use_class_weights", True) else None
    )
    otimizador = optim.Adam(trainable_parameters(model), lr=lr,
                            weight_decay=float(config.get("weight_decay", 0.0)))
    reduz = optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.1,
        patience=int(config.get("lr_patience", 6)))

    usa_penalidade = bool(config.get("elastic_net", False))
    l1 = float(config.get("l1", 1e-5)); l2 = float(config.get("l2", 1e-4))

    melhor, melhor_estado, melhor_epoca, espera = -np.inf, None, -1, 0
    log = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perda_acc, n = 0.0, 0
        for x, y, _, _ in loaders["train"]:
            x, y = x.to(device), y.to(device)
            otimizador.zero_grad(set_to_none=True)
            perda = criterio(model(x), y)
            if usa_penalidade:
                perda = perda + elastic_net_penalty(model, l1, l2)
            perda.backward()
            otimizador.step()
            perda_acc += float(perda.item()) * x.size(0); n += x.size(0)

        probs, y_true, _, _ = infer(model, loaders["val"], device)
        m = compute_metrics(y_true, probs.argmax(1), class_names)
        score = float(m.get(metrica, m["f1_macro"]))
        reduz.step(score)
        log.append({"epoch": ep, "train_loss": perda_acc / max(n, 1),
                    "lr": otimizador.param_groups[0]["lr"],
                    **{f"val_{k}": v for k, v in m.items()}})

        if score > melhor:
            melhor, melhor_epoca, espera = score, ep, 0
            melhor_estado = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        else:
            espera += 1
            if patience > 0 and espera >= patience:
                break

    if melhor_estado is not None:
        model.load_state_dict(melhor_estado)
    pd.DataFrame(log).to_csv(Path(out_dir) / "train_log.csv", index=False)
    return {"best_epoch": melhor_epoca, "best_val_score": melhor,
            "epochs_run": len(log), "seconds": time.time() - t0}


def extract_once(model, loaders, device):
    """Extrai as features de treino e validacao UMA vez.

    Este e o ponto do desenho: a extracao custa uma passada completa pelo
    dataset, enquanto ajustar uma floresta sobre features prontas custa
    segundos. Separar o grid da rede do grid da floresta significa pagar a
    passada cara uma vez por configuracao de REDE, nao uma vez por
    combinacao rede x floresta.
    """
    model.freeze_all()
    return {k: extract_features(model, loaders[k], device)
            for k in ("train_eval", "val")}


def fit_forest(feats, rf_config, seed, out_dir):
    """Ajusta StandardScaler + RandomForest sobre features ja extraidas.

    O Pipeline e ajustado SO nas features de treino. O scaler encapsulado
    garante que estatisticas de validacao e teste nunca vazam.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rf = make_rf_pipeline(
        n_estimators=int(rf_config.get("rf_n_estimators", 100)),
        class_weight=rf_config.get("rf_class_weight", "balanced"),
        max_depth=rf_config.get("rf_max_depth"),
        min_samples_leaf=int(rf_config.get("rf_min_samples_leaf", 1)),
        seed=int(seed),
    )
    Xtr, ytr, _, _ = feats["train_eval"]
    rf.fit(Xtr, ytr)

    imp = rf.named_steps["rf"].feature_importances_
    pd.DataFrame({"feature": np.arange(len(imp)), "importance": imp}) \
        .sort_values("importance", ascending=False) \
        .to_csv(out_dir / "rf_feature_importance.csv", index=False)

    Xv, yv, sv, fv = feats["val"]
    return rf, (rf.predict_proba(Xv), yv, sv, fv)


def run_candidate(fold_csv, images_dir, config, out_dir, device,
                  n_classes, class_names, rf_grid=None):
    """Treina UMA configuracao de rede e devolve uma lista de resultados.

    Para bernini_cnn a lista tem um elemento. Para resnet50_rf ela tem um
    elemento por configuracao de floresta em `rf_grid`, todas ajustadas sobre
    a mesma extracao de features — a rede treina uma vez so.

    O teste nao e avaliado aqui. Quem avalia e grid_search.py, uma unica vez,
    depois de escolher o vencedor.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config.get("seed", 42)))

    _, loaders = build_loaders(fold_csv, images_dir, config, n_classes)
    model = create_model(config, n_classes).to(device)
    resumo = train_torch(model, loaders, config, device, n_classes,
                         class_names, out_dir)

    torch.save({"model_state_dict": model.state_dict(), "config": config},
               out_dir / "model.pt")
    with open(out_dir / "config_used.json", "w") as fh:
        json.dump({**config, "_env": environment_info()}, fh, indent=2, default=str)

    if config.get("model_name") != "resnet50_rf":
        probs, y, s, f = infer(model, loaders["val"], device)
        predictions_dataframe(f, s, y, probs.argmax(1), probs) \
            .to_csv(out_dir / "predictions_val.csv", index=False)
        m = compute_metrics(y, probs.argmax(1), class_names)
        return [{"net_dir": str(out_dir), "cand_dir": str(out_dir),
                 **resumo, **{f"val_{k}": v for k, v in m.items()}}]

    # Cabeca linear: baseline gratuito contra o qual a floresta se compara.
    probs, y, s, f = infer(model, loaders["val"], device)
    m_lin = compute_metrics(y, probs.argmax(1), class_names)
    resumo.update({f"linear_val_{k}": v for k, v in m_lin.items()})

    feats = extract_once(model, loaders, device)          # passada cara, uma vez
    import joblib
    resultados = []
    for j, rf_cfg in enumerate(rf_grid or [{}]):
        rf_dir = out_dir / f"rf_{j:03d}"
        rf, (probs, y, s, f) = fit_forest(feats, {**config, **rf_cfg},
                                          config.get("seed", 42), rf_dir)
        joblib.dump(rf, rf_dir / "rf_head.joblib")
        predictions_dataframe(f, s, y, probs.argmax(1), probs) \
            .to_csv(rf_dir / "predictions_val.csv", index=False)
        with open(rf_dir / "rf_config.json", "w") as fh:
            json.dump(rf_cfg, fh, indent=2, default=str)
        m = compute_metrics(y, probs.argmax(1), class_names)
        resultados.append({"net_dir": str(out_dir), "cand_dir": str(rf_dir),
                           "rf_id": f"rf_{j:03d}", **rf_cfg, **resumo,
                           **{f"val_{k}": v for k, v in m.items()}})
    return resultados
