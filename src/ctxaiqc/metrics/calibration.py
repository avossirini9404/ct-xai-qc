"""Calibration and predictive uncertainty.

A model that keeps its accuracy under a perturbation but becomes overconfident
has still degraded, and the expected calibration error is the cheapest way to see
it. Monte-Carlo dropout gives a second, distribution-level view at the cost of a
few extra forward passes.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width binned gap between confidence and accuracy (Guo et al., 2017)."""
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15):
    """Bin centres, empirical accuracy and mean confidence, for the diagram."""
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, acc, mean_conf, weights = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        centres.append(0.5 * (lo + hi))
        acc.append(float(correct[mask].mean()))
        mean_conf.append(float(conf[mask].mean()))
        weights.append(float(mask.mean()))
    return np.array(centres), np.array(acc), np.array(mean_conf), np.array(weights)


def mc_dropout_probs(model, x, n_samples: int = 20, batch_size: int = 256, device=None):
    """Mean and standard deviation of the softmax over Monte-Carlo dropout passes."""
    import torch

    from ..models import enable_mc_dropout

    device = device or next(model.parameters()).device
    enable_mc_dropout(model)
    xs = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            probs = []
            for start in range(0, len(xs), batch_size):
                batch = xs[start : start + batch_size].to(device)
                probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
            samples.append(np.concatenate(probs, axis=0))
    stacked = np.stack(samples)
    model.eval()
    return stacked.mean(axis=0), stacked.std(axis=0)
