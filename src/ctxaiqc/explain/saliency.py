"""Saliency methods, implemented from the original papers.

Grad-CAM (Selvaraju et al., 2017), Integrated Gradients (Sundararajan et al.,
2017) and occlusion (Zeiler and Fergus, 2014) are written out here rather than
imported from a library, so that every step that the quality-control protocol
measures is visible in this repository and can be checked against the papers.

All three return a single-channel map, resized to the input resolution and scaled
to ``[0, 1]``. The scaling matters: the metrics in :mod:`ctxaiqc.metrics.stability`
compare maps produced from different images, and an unnormalised map would make
them incomparable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def _as_batch(x: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x, dtype=torch.float32)
    if t.ndim == 2:
        t = t[None, None]
    elif t.ndim == 3:
        t = t[None]
    if t.shape[0] != 1:
        raise ValueError("explainers operate on one image at a time")
    return t.to(device)


def _normalise(sal: np.ndarray) -> np.ndarray:
    sal = np.asarray(sal, dtype=np.float32)
    lo, hi = float(sal.min()), float(sal.max())
    if hi - lo < 1e-12:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - lo) / (hi - lo)).astype(np.float32)


def _resize(sal: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    out = nn.functional.interpolate(sal[None, None], size=size, mode="bilinear", align_corners=False)
    return out[0, 0].detach().cpu().numpy()


def _target_of(model: nn.Module, x: torch.Tensor, target: int | None) -> int:
    if target is not None:
        return int(target)
    with torch.no_grad():
        return int(model(x).argmax(dim=1).item())


def grad_cam(model: nn.Module, x, target: int | None = None, layer: nn.Module | None = None) -> np.ndarray:
    """Gradient-weighted class activation mapping."""
    device = next(model.parameters()).device
    x = _as_batch(x, device)
    layer = layer if layer is not None else model.feature_layer
    model.eval()

    activations: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inp, out):
        activations["value"] = out
        out.retain_grad()

    handle = layer.register_forward_hook(forward_hook)
    try:
        logits = model(x)
        cls = int(target) if target is not None else int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, cls].backward()
        acts = activations["value"]
        grads = acts.grad
        if grads is None:
            raise RuntimeError("no gradients reached the feature layer")
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))[0]
    finally:
        handle.remove()
    return _normalise(_resize(cam, x.shape[-2:]))


def integrated_gradients(
    model: nn.Module,
    x,
    target: int | None = None,
    steps: int = 64,
    baseline: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Integrated gradients along a straight path from a baseline image.

    The baseline is a constant image by default. For CT the natural choice is the
    value that encodes air, which under the convention of
    :mod:`ctxaiqc.perturb.acquisition` is 0.
    """
    device = next(model.parameters()).device
    x = _as_batch(x, device)
    model.eval()
    base = torch.full_like(x, float(baseline)) if np.isscalar(baseline) else _as_batch(baseline, device)
    cls = _target_of(model, x, target)

    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)
    total = torch.zeros_like(x)
    for alpha in alphas:
        point = (base + alpha * (x - base)).clone().requires_grad_(True)
        logits = model(point)
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(logits[0, cls], point)[0]
        total += grad
    attributions = (x - base) * total / steps
    return _normalise(attributions.abs().sum(dim=1)[0].detach().cpu().numpy())


def occlusion(
    model: nn.Module,
    x,
    target: int | None = None,
    patch: int = 8,
    stride: int = 4,
    fill: float = 0.0,
) -> np.ndarray:
    """Drop in the class probability when a patch is replaced by ``fill``."""
    device = next(model.parameters()).device
    x = _as_batch(x, device)
    model.eval()
    cls = _target_of(model, x, target)
    with torch.no_grad():
        base_prob = torch.softmax(model(x), dim=1)[0, cls].item()

    h, w = x.shape[-2:]
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    patches, positions = [], []
    for top in range(0, max(h - patch + 1, 1), stride):
        for left in range(0, max(w - patch + 1, 1), stride):
            occluded = x.clone()
            occluded[..., top : top + patch, left : left + patch] = fill
            patches.append(occluded)
            positions.append((top, left))
    with torch.no_grad():
        probs = torch.softmax(model(torch.cat(patches, dim=0)), dim=1)[:, cls].cpu().numpy()
    for (top, left), prob in zip(positions, probs, strict=False):
        heat[top : top + patch, left : left + patch] += base_prob - float(prob)
        counts[top : top + patch, left : left + patch] += 1.0
    heat = heat / np.maximum(counts, 1.0)
    return _normalise(np.maximum(heat, 0.0))


EXPLAINERS: dict[str, Callable[..., np.ndarray]] = {
    "gradcam": grad_cam,
    "integrated_gradients": integrated_gradients,
    "occlusion": occlusion,
}


def explain(method: str, model: nn.Module, x, target: int | None = None, **kwargs) -> np.ndarray:
    """Dispatch to a saliency method by name."""
    if method not in EXPLAINERS:
        raise KeyError(f"unknown explainer {method!r}; available: {sorted(EXPLAINERS)}")
    return EXPLAINERS[method](model, x, target=target, **kwargs)
