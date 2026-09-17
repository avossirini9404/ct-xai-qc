"""Agreement between two saliency maps.

This is the measurement the whole protocol turns on. Three complementary views
are reported, because each fails in a different way:

``spearman``
    Rank correlation over all pixels. Insensitive to the scale of the map,
    sensitive to any reordering, including reordering in the empty background.
``ssim``
    Structural similarity, which asks whether the map has the same spatial
    structure rather than the same ordering.
``top_k_jaccard``
    Overlap of the most salient k per cent of pixels, which is the part of the
    map a reader actually looks at.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


def saliency_agreement(a: np.ndarray, b: np.ndarray, top_k: float = 0.1) -> dict[str, float]:
    """Compare two saliency maps of the same shape, both scaled to [0, 1]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    if a.std() < 1e-12 or b.std() < 1e-12:
        rho = float("nan")
    else:
        rho = float(spearmanr(a.ravel(), b.ravel()).statistic)

    ssim = float(structural_similarity(a, b, data_range=1.0))

    k = max(1, int(round(top_k * a.size)))
    top_a = set(np.argsort(a.ravel())[::-1][:k].tolist())
    top_b = set(np.argsort(b.ravel())[::-1][:k].tolist())
    union = len(top_a | top_b)
    jaccard = float(len(top_a & top_b) / union) if union else float("nan")

    return {"spearman": rho, "ssim": ssim, "top_k_jaccard": jaccard}


def summarise_agreement(records: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation of each agreement measure over a set of images."""
    if not records:
        return {}
    keys = records[0].keys()
    out: dict[str, float] = {"n": float(len(records))}
    for key in keys:
        values = np.array([r[key] for r in records], dtype=np.float64)
        out[f"{key}_mean"] = float(np.nanmean(values))
        out[f"{key}_std"] = float(np.nanstd(values))
    return out
