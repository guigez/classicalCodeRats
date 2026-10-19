"""
ETAPA 7 — GRID SEARCH E AVALIACAO
=================================

Percorre o grid dentro de um fold, escolhe o vencedor pela VALIDACAO e avalia
no teste uma unica vez.

O invariante do protocolo esta concentrado aqui: entre `run_candidate` (que
so ve treino e validacao) e `evaluate_on_test` (que roda uma vez, com tudo ja
congelado) nao existe caminho de volta. Se voce olhar o teste e voltar para
ajustar qualquer coisa, a etapa de inferencia vira ficcao.

Uma diferenca deliberada em relacao ao pipeline do Furlanetti: todos os
candidatos de um fold recebem a MESMA semente. La, `seed = base + idx` fazia
os candidatos diferirem em hiperparametro E em inicializacao, e o argmax sobre
a validacao selecionava parcialmente sorte de semente. Aqui a semente nunca
varia por candidato (o run_all.py passa a mesma --seed a todos os folds).
Quanto da diferenca entre familias e sorte de semente e medido a parte, em
seed_replicates.py.

Uso:
    python src/grid_search.py --fold splits/fold_00.csv --grid configs/grid_bernini.json \\
        --images_dir imagens --out_dir resultados/fold_00/bernini_cnn
"""


from __future__ import annotations

from common import resolve_device, silence_known_warnings
silence_known_warnings()

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import ThermalFrameDataset
from metrics import compute_metrics, predictions_dataframe, save_all
from manifest import TASKS, class_names_for
from models import create_model
from train import build_loaders, infer, run_candidate



def expand_grid(grid: dict) -> list[dict]:
    """Produto cartesiano do bloco `grid`. Valor escalar vira lista de um."""
    chaves = list(grid)
    valores = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(chaves, c)) for c in itertools.product(*valores)]


def evaluate_on_test(net_dir: Path, cand_dir: Path, fold_csv, images_dir,
                     config, device, n_classes, class_names, out_dir: Path):
    """Avaliacao unica no teste, com o modelo vencedor ja congelado.

    net_dir  guarda model.pt (a rede treinada).
    cand_dir guarda rf_head.joblib — igual a net_dir quando nao ha floresta.
    """
    _, loaders = build_loaders(fold_csv, images_dir, config, n_classes)
    model = create_model(config, n_classes).to(device)
    model.load_state_dict(torch.load(net_dir / "model.pt",
                                     map_location=device)["model_state_dict"])

    if config.get("model_name") == "resnet50_rf":
        # Cabeca linear como terceiro braco, no mesmo teste.
        probs, y, s, f = infer(model, loaders["test"], device)
        save_all(predictions_dataframe(f, s, y, probs.argmax(1), probs),
                 class_names, out_dir, prefix="test_linear_head")

        import joblib
        rf = joblib.load(cand_dir / "rf_head.joblib")
        from models import extract_features
        model.freeze_all()
        X, y, s, f = extract_features(model, loaders["test"], device)
        probs = rf.predict_proba(X)
    else:
        probs, y, s, f = infer(model, loaders["test"], device)

    preds = predictions_dataframe(f, s, y, probs.argmax(1), probs)
    return save_all(preds, class_names, out_dir, prefix="test")


def main() -> int:
    ap = argparse.ArgumentParser(description="Etapa 7: grid search em um fold.")
    ap.add_argument("--fold", required=True, help="CSV de um fold.")
    ap.add_argument("--grid", required=True, help="JSON com blocos 'base' e 'grid'.")
    ap.add_argument("--images_dir", default="imagens")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--task", default="multiclass", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = resolve_device(args.device)
    class_names = class_names_for(args.task)
    n_classes = len(class_names)

    cfg = json.loads(Path(args.grid).read_text())
    base, grid = cfg.get("base", {}), cfg.get("grid", {})
    rf_grid = expand_grid(cfg.get("rf_grid", {})) or [{}]
    metrica = cfg.get("selection", {}).get("metric", "val_f1_macro")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    redes = expand_grid(grid) or [{}]
    n_folhas = len(redes) * (len(rf_grid) if base.get("model_name") == "resnet50_rf" else 1)
    print(f"[{out_dir.name}] dispositivo={device}  "
          f"treinos de rede={len(redes)}  candidatos avaliados={n_folhas}  "
          f"selecao por {metrica}")

    resumos = []
    for i, combo in enumerate(redes):
        c = {**base, **combo, "config_id": f"net_{i:03d}", "seed": args.seed}
        net_dir = out_dir / f"net_{i:03d}"
        print(f"  net_{i:03d}  {combo}")
        try:
            for r in run_candidate(args.fold, args.images_dir, c, net_dir, device,
                                   n_classes, class_names, rf_grid):
                resumos.append({"net_id": f"net_{i:03d}", **combo, **r,
                                "status": "ok"})
        except Exception as e:
            print(f"    ERRO: {e!r}")
            resumos.append({"net_id": f"net_{i:03d}", **combo, "status": "erro",
                            "error": repr(e), "net_dir": str(net_dir),
                            "cand_dir": str(net_dir)})
        pd.DataFrame(resumos).to_csv(out_dir / "grid_summary.csv", index=False)

    df = pd.DataFrame(resumos)
    ok = df[df.status == "ok"]
    if ok.empty:
        print("[ERRO] nenhum candidato concluiu.", file=sys.stderr)
        return 1
    if metrica not in ok.columns:
        print(f"[ERRO] metrica '{metrica}' ausente. Disponiveis: "
              f"{[c for c in ok.columns if c.startswith('val_')]}", file=sys.stderr)
        return 1

    melhor = ok.loc[ok[metrica].astype(float).idxmax()]
    rotulo = f"{melhor.net_id}" + (f"/{melhor.rf_id}" if "rf_id" in ok.columns
                                   and pd.notna(melhor.get("rf_id")) else "")
    print(f"  vencedor: {rotulo}  {metrica}={melhor[metrica]:.4f}")

    net_dir = Path(melhor["net_dir"]); cand_dir = Path(melhor["cand_dir"])
    config = json.loads((net_dir / "config_used.json").read_text())
    m_teste = evaluate_on_test(net_dir, cand_dir, args.fold, args.images_dir,
                               config, device, n_classes, class_names, out_dir)

    pd.DataFrame([{**melhor.to_dict(),
                   **{f"test_{k}": v for k, v in m_teste.items()}}]) \
        .to_csv(out_dir / "best_summary.csv", index=False)
    (out_dir / "best_selection.json").write_text(json.dumps(
        {"best_config_id": rotulo, "net_dir": str(net_dir),
         "cand_dir": str(cand_dir), "selected_by": metrica}, indent=2))
    linha = f"  teste: f1_macro={m_teste.get('f1_macro', float('nan')):.4f}"
    if "macro_f1_lo" in m_teste:
        linha += (f"  IC95%=[{m_teste['macro_f1_lo']:.3f}, {m_teste['macro_f1_hi']:.3f}]"
                  f"  (bootstrap sobre {m_teste.get('macro_f1_n_subjects')} sujeitos)")
    print(linha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
