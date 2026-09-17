"""Inference helpers and a small command-line evaluation report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_dataset
from .metrics import expected_calibration_error, summarise_performance
from .models import build_model
from .utils import load_config


def predict_probs(model, x: np.ndarray, batch_size: int = 256, device=None) -> np.ndarray:
    """Softmax probabilities for a batch of images shaped (N, 1, H, W)."""
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    xs = torch.as_tensor(np.asarray(x, dtype=np.float32))
    out = []
    with torch.no_grad():
        for start in range(0, len(xs), batch_size):
            batch = xs[start : start + batch_size].to(device)
            out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_checkpoint(path: str | Path, device=None):
    """Rebuild a model from a checkpoint written by :mod:`ctxaiqc.train`."""
    import torch

    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model = build_model(
        ckpt["model_name"],
        n_classes=ckpt["n_classes"],
        dropout=ckpt.get("dropout", 0.3),
        **ckpt.get("model_kwargs", {}),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data = load_dataset(**cfg["dataset"], seed=cfg["seed"])
    model, ckpt = load_checkpoint(cfg["train"]["checkpoint"])
    probs = predict_probs(model, data.x_test)

    scores = summarise_performance(data.y_test, probs)
    scores["ece"] = expected_calibration_error(data.y_test, probs)
    print(f"checkpoint: {cfg['train']['checkpoint']} (epoch {ckpt.get('epoch', '?')})")
    for key, value in scores.items():
        print(f"  {key:>18}: {value:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
