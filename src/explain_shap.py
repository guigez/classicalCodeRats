"""
ETAPA 8 — INTERPRETACAO (SHAP)
==============================

Duas analises distintas, porque os dois modelos expoem superficies diferentes:

  bernini_cnn / cabeca linear da ResNet50
      GradientExplainer produz atribuicao POR PIXEL. E o que responde a
      pergunta que interessa neste projeto: o modelo esta olhando para o
      animal ou para a esteira?

  floresta
      A importancia por impureza opera sobre as 2048 dimensoes do avgpool,
      nao sobre pixels. Isso diz QUAL feature abstrata pesou, nao ONDE na imagem. Nao
      ha como converter de volta para pixel sem um substituto diferenciavel.
      A analise da floresta e, portanto, um complemento — a evidencia
      espacial vem da atribuicao em pixel da cabeca linear, que compartilha o
      mesmo backbone.

CONTEXTO QUE MOTIVA ESTA ETAPA
------------------------------
O proprio Bernini reportou que os valores de Shapley se concentravam na
esteira, e listou "remover o fundo" como trabalho futuro. Combinado com o
confundimento temporal — a classe e determinada pelo tempo decorrido, e a
deriva termica do fundo acompanha o tempo — a pergunta "o modelo le o animal
ou o ambiente?" deixa de ser curiosidade interpretativa e vira teste de
validade. Por isso `background_ratio` e calculado: e a fracao da atribuicao
absoluta que cai fora da regiao do animal.

Uso:
    python src/explain_shap.py --candidate resultados/fold_00/bernini_cnn/cfg_003 \\
        --fold splits/fold_00.csv --images_dir imagens \\
        --out_dir resultados/fold_00/bernini_cnn/shap
"""


from __future__ import annotations

from common import resolve_device, silence_known_warnings
silence_known_warnings()

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from manifest import TASKS, class_names_for
from models import create_model
from train import build_loaders



def to_display(x: torch.Tensor, imagenet_norm: bool) -> np.ndarray:
    """Tensor CHW normalizado -> array HWC em [0,1] para exibicao."""
    a = x.detach().cpu().numpy().transpose(1, 2, 0)
    if imagenet_norm:
        a = a * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(a, 0, 1)


def sample_background(dataset, k: int, seed: int, device):
    """Amostra de referencia para o explicador, estratificada por sujeito x rotulo.

    A versao anterior lia os primeiros lotes de um DataLoader sem shuffle.
    Como o fold vem ordenado por sujeito e depois por classe, as k imagens
    saiam todas do warm-up do primeiro sujeito do treino — e o
    GradientExplainer atribui em relacao a essa referencia. Aqui cada celula
    (sujeito, rotulo) do treino contribui em rodizio, ate completar k.
    """
    rng = np.random.default_rng(seed)
    celulas = [rng.permutation(np.asarray(idx)).tolist()
               for _, idx in sorted(dataset.df.groupby(["subject_id", "label"]).groups.items())]
    ordem = rng.permutation(len(celulas))
    alvo = min(k, len(dataset))
    escolhidos: list[int] = []
    while len(escolhidos) < alvo:
        for j in ordem:
            if celulas[j] and len(escolhidos) < alvo:
                escolhidos.append(int(celulas[j].pop()))
    return torch.stack([dataset[i][0] for i in escolhidos]).to(device)


def pick_examples(preds_csv: Path, class_names: list[str], seed: int = 42):
    """Um acerto e um erro por classe, quando existirem.

    Escolher so acertos produz um painel que confirma o que se quer ver. Os
    erros sao onde a interpretacao ensina alguma coisa.
    """
    df = pd.read_csv(preds_csv)
    rng = np.random.default_rng(seed)
    escolhidos = []
    for c in range(len(class_names)):
        for tipo, sel in (("acerto", (df.y_true == c) & (df.correct == 1)),
                          ("erro", (df.y_true == c) & (df.correct == 0))):
            sub = df[sel]
            if len(sub):
                r = sub.iloc[int(rng.integers(len(sub)))]
                escolhidos.append({"filename": r.filename, "tipo": tipo,
                                   "y_true": int(r.y_true), "y_pred": int(r.y_pred),
                                   "subject_id": int(r.subject_id)})
    return escolhidos


def animal_mask(img: np.ndarray, background_removed: bool = True,
                tol: float = 0.02) -> np.ndarray:
    """Mascara do animal.

    background_removed=True: o fundo ja foi retirado no pre-processamento; a
    mascara e a regiao nao vazia. Exato.

    background_removed=False: imagem COM esteira. O animal e a regiao quente do
    colormap jet — alta componente vermelha e baixa azul. Usa limiar de Otsu
    sobre (R-B) em vez de um quantil fixo, porque a fracao da imagem ocupada
    pelo rato varia muito entre quadros (o rato 4 e minusculo, o rato 1 e
    grande) e um quantil fixo ora corta o animal ora inclui esteira.

    ATENCAO: mesmo assim isto e uma aproximacao. A esteira tem regioes mornas
    (barras verdes/amarelas) que podem entrar na mascara. Para o
    background_ratio ser confiavel de verdade seria preciso a mascara de
    segmentacao real usada na remocao de fundo — se voce a tem, passe-a.
    """
    if background_removed:
        return img.max(axis=2) > tol

    calor = img[:, :, 0] - img[:, :, 2]          # R - B: alto = quente
    calor = (calor - calor.min()) / (np.ptp(calor) + 1e-9)
    # Otsu simples: varre limiares e maximiza variancia entre classes.
    hist, bordas = np.histogram(calor.ravel(), bins=64)
    centros = (bordas[:-1] + bordas[1:]) / 2
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centros)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-9
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    limiar = centros[np.argmax(sigma_b)]
    return calor >= limiar


def background_ratio(shap_abs: np.ndarray, mask: np.ndarray) -> float:
    """Fracao da atribuicao absoluta que cai FORA do animal.

    ATENCAO: este numero sozinho e ININTERPRETAVEL. Se o animal ocupa 8% da
    imagem e a atribuicao fosse uniforme (o modelo nao olhando nada em
    especial), background_ratio ja daria 92% so pela area. Um valor de 75%
    pode significar 'olha o fundo' OU 'atribuicao difusa num quadro onde o
    animal e pequeno'. Use `attribution_enrichment` para desambiguar.
    """
    total = shap_abs.sum()
    return float(shap_abs[~mask].sum() / total) if total > 0 else float("nan")


def attribution_enrichment(shap_abs: np.ndarray, mask: np.ndarray) -> dict:
    """Densidade de atribuicao no animal versus no fundo.

    Este e o numero que de fato responde 'o modelo olha o animal ou o
    ambiente?', porque corrige pela area. Definicoes:

      dens_animal = (soma da atribuicao no animal) / (area do animal)
      dens_fundo  = (soma da atribuicao no fundo)  / (area do fundo)
      enrichment  = dens_animal / dens_fundo

    Interpretacao:
      enrichment > 1  -> cada pixel do animal recebe MAIS atribuicao que cada
                         pixel do fundo. O modelo privilegia o animal.
      enrichment ~ 1  -> atribuicao uniforme; o modelo nao distingue animal de
                         fundo. E o caso do colapso.
      enrichment < 1  -> o modelo olha o fundo desproporcionalmente.

    Diferente do background_ratio, enrichment nao depende do tamanho do
    animal: e uma razao de densidades, nao de massas.
    """
    area_animal = int(mask.sum())
    area_fundo = int((~mask).sum())
    if area_animal == 0 or area_fundo == 0:
        return {"enrichment": float("nan"), "animal_area_frac": float("nan"),
                "dens_animal": float("nan"), "dens_fundo": float("nan")}
    dens_animal = shap_abs[mask].sum() / area_animal
    dens_fundo = shap_abs[~mask].sum() / area_fundo
    enr = float(dens_animal / dens_fundo) if dens_fundo > 0 else float("inf")
    return {
        "enrichment": enr,
        "animal_area_frac": area_animal / (area_animal + area_fundo),
        "dens_animal": float(dens_animal),
        "dens_fundo": float(dens_fundo),
    }


def explain_pixels(model, background, X, device, n_classes):
    """Atribuicao por pixel via GradientExplainer."""
    import shap
    explainer = shap.GradientExplainer(model, background)
    valores = explainer.shap_values(X.to(device))
    if isinstance(valores, list):                      # lista por classe
        valores = np.stack([np.asarray(v) for v in valores], axis=-1)
    return np.asarray(valores)


def painel(exemplos, imgs, shap_vals, class_names, out_png: Path,
           imagenet_norm: bool, background_removed: bool = True):
    """Grade: imagem, atribuicao para a classe real, sobreposicao."""
    n = len(exemplos)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.0 * n), squeeze=False)
    for i, ex in enumerate(exemplos):
        img = to_display(imgs[i], imagenet_norm)
        heat = np.abs(shap_vals[i, ..., ex["y_true"]]).sum(0)
        heat = heat / (heat.max() + 1e-12)
        mask = animal_mask(img, background_removed)
        br = background_ratio(heat, mask)

        axes[i][0].imshow(img)
        axes[i][0].set_title(
            f"{class_names[ex['y_true']]} -> {class_names[ex['y_pred']]}"
            f"  (rato {ex['subject_id']})", fontsize=8)
        axes[i][1].imshow(heat, cmap="magma")
        axes[i][1].set_title(f"|SHAP|  fundo={br:.0%}", fontsize=8)
        axes[i][2].imshow(img); axes[i][2].imshow(heat, cmap="magma", alpha=0.5)
        axes[i][2].set_title("sobreposicao", fontsize=8)
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def explain_forest(cand_dir: Path, out_dir: Path, top_k: int = 30):
    """Importancia por impureza das 2048 features. Nao e espacial — ver docstring.

    Nao roda TreeExplainer: com 2048 features e centenas de arvores ele e caro,
    e continuaria sem dizer ONDE na imagem. A importancia da floresta basta
    como complemento da atribuicao em pixel da cabeca linear.
    """
    import joblib
    rf = joblib.load(cand_dir / "rf_head.joblib")
    floresta = rf.named_steps["rf"]
    imp = floresta.feature_importances_
    ordem = np.argsort(imp)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(7, 0.22 * top_k + 1.5))
    ax.barh([f"f{j}" for j in ordem][::-1], imp[ordem][::-1])
    ax.set_xlabel("importancia (impureza)")
    ax.set_title("Floresta: top features do avgpool (nao espacial)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "rf_top_features.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"feature": np.arange(len(imp)), "importance": imp}) \
        .sort_values("importance", ascending=False) \
        .to_csv(out_dir / "rf_feature_importance.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Etapa 8: interpretacao com SHAP.")
    ap.add_argument("--candidate", required=True, help="Pasta do candidato vencedor.")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task", default="multiclass", choices=list(TASKS))
    ap.add_argument("--background_size", type=int, default=32)
    ap.add_argument("--with_background", action="store_true",
                    help="Imagens AINDA tem a esteira. Por padrao assume fundo removido.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = resolve_device(args.device)
    class_names = class_names_for(args.task)
    n_classes = len(class_names)

    cand = Path(args.candidate)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # O candidato pode ser um diretorio de rede (net_XXX/) ou de floresta
    # (net_XXX/rf_YYY/). O model.pt e o config_used.json vivem sempre no
    # diretorio da REDE — a floresta so guarda rf_head.joblib. Sobe um nivel
    # quando o config nao esta onde --candidate aponta. Foi este descasamento
    # que fez as 4 execucoes de SHAP da ResNet falharem: --candidate recebia
    # o diretorio da floresta e o modulo procurava config la.
    net_dir = cand
    if not (net_dir / "config_used.json").exists() and \
       (net_dir.parent / "config_used.json").exists():
        net_dir = net_dir.parent
    config = json.loads((net_dir / "config_used.json").read_text())
    imagenet_norm = config.get("model_name") == "resnet50_rf"

    model = create_model(config, n_classes).to(device)
    model.load_state_dict(torch.load(net_dir / "model.pt",
                                     map_location=device)["model_state_dict"])
    model.eval()
    # GradientExplainer precisa de gradiente. O pipeline pode ter congelado
    # tudo com freeze_all() durante a extracao de features da floresta; aqui
    # reativamos, senao o explicador nao tem por onde propagar.
    for p in model.parameters():
        p.requires_grad_(True)

    ds, _ = build_loaders(args.fold, args.images_dir, config, n_classes)

    # As predicoes de teste ficam na raiz do resultado do fold/familia (dois
    # ou tres niveis acima), gravadas por grid_search. As de validacao ficam
    # junto do candidato. Preferimos teste; caimos para validacao se preciso.
    preds_csv = None
    for cand_path in (net_dir.parent / "predictions_test.csv",
                      net_dir.parent.parent / "predictions_test.csv",
                      cand / "predictions_val.csv",
                      net_dir / "predictions_val.csv"):
        if cand_path.exists():
            preds_csv = cand_path
            break
    if preds_csv is None:
        print("[aviso] nenhum arquivo de predicoes encontrado.")
        return 0
    exemplos = pick_examples(preds_csv, class_names, args.seed)
    if not exemplos:
        print("[aviso] nenhum exemplo selecionado.")
        return 0

    conjunto = "test" if "test" in preds_csv.name else "val"
    por_nome = {n: i for i, n in enumerate(ds[conjunto].df.filename)}
    idx = [por_nome[e["filename"]] for e in exemplos if e["filename"] in por_nome]
    exemplos = [e for e in exemplos if e["filename"] in por_nome]
    X = torch.stack([ds[conjunto][i][0] for i in idx])

    bg = sample_background(ds["train_eval"], args.background_size,
                           args.seed, device)
    print(f"explicando {len(exemplos)} exemplos com fundo de {len(bg)} imagens...")
    shap_vals = explain_pixels(model, bg, X, device, n_classes)

    bg_removido = not args.with_background
    painel(exemplos, X, shap_vals, class_names,
           out_dir / "shap_panel.png", imagenet_norm, bg_removido)

    linhas = []
    for i, e in enumerate(exemplos):
        img = to_display(X[i], imagenet_norm)
        heat = np.abs(shap_vals[i, ..., e["y_true"]]).sum(0)
        mask = animal_mask(img, bg_removido)
        enr = attribution_enrichment(heat, mask)
        linhas.append({**e, "background_ratio": background_ratio(heat, mask), **enr})
    df = pd.DataFrame(linhas)
    df.to_csv(out_dir / "shap_background_ratio.csv", index=False)

    media = df.background_ratio.mean()
    enr_med = df.enrichment.replace([np.inf, -np.inf], np.nan).mean()
    area_med = df.animal_area_frac.mean()
    print(f"fracao media da atribuicao fora do animal: {media:.1%}")
    print(f"area media do animal: {area_med:.1%} da imagem")
    print(f"enriquecimento (densidade animal / densidade fundo): {enr_med:.2f}")
    if enr_med > 1.3:
        print("  -> o modelo olha o animal DESPROPORCIONALMENTE. O background_ratio")
        print("     alto e efeito da area pequena do animal, nao de olhar a esteira.")
    elif enr_med < 0.8:
        print("  -> o modelo olha o FUNDO desproporcionalmente. Confirma leitura de")
        print("     ambiente, como o Bernini relatou.")
    else:
        print("  -> atribuicao aproximadamente UNIFORME: o modelo nao privilegia o")
        print("     animal (compativel com colapso ou leitura difusa).")

    if (cand / "rf_head.joblib").exists():
        explain_forest(cand, out_dir)

    print(f"[ok] resultados em {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
