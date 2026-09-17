"""Faithfulness of an explanation: deletion and insertion curves.

Petsiuk et al. (2018). Pixels are removed (deletion) or restored (insertion) in
the order the saliency map ranks them, and the probability of the predicted class
is tracked. A faithful map deletes the evidence quickly, giving a low deletion
area and a high insertion area. An explanation that does no better than random
ranking is not measuring what the model used.
"""

from __future__ import annotations

import numpy as np


def _ordered_indices(saliency: np.ndarray, descending: bool = True) -> np.ndarray:
    flat = np.asarray(saliency, dtype=np.float64).ravel()
    order = np.argsort(flat)
    return order[::-1] if descending else order


def deletion_insertion(
    model,
    x: np.ndarray,
    saliency: np.ndarray,
    target: int | None = None,
    steps: int = 32,
    fill: float = 0.0,
    batch_size: int = 64,
) -> dict[str, object]:
    """Deletion and insertion curves and their normalised areas.

    Returns a dictionary with the fraction of pixels perturbed, both probability
    curves, and the areas under them (trapezoidal, on a 0-1 scale).
    """
    import torch

    device = next(model.parameters()).device
    model.eval()
    img = np.asarray(x, dtype=np.float32)
    if img.ndim == 2:
        img = img[None]
    n_pixels = img.shape[-1] * img.shape[-2]

    with torch.no_grad():
        logits = model(torch.as_tensor(img[None], device=device))
        cls = int(target) if target is not None else int(logits.argmax(dim=1).item())

    order = _ordered_indices(saliency)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    counts = (fractions * n_pixels).astype(int)

    def _probs(images: list[np.ndarray]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = torch.as_tensor(np.stack(images[start : start + batch_size]), device=device)
                out.append(torch.softmax(model(batch), dim=1)[:, cls].cpu().numpy())
        return np.concatenate(out)

    deletion_imgs, insertion_imgs = [], []
    for k in counts:
        deleted = img.copy().reshape(img.shape[0], -1)
        deleted[:, order[:k]] = fill
        deletion_imgs.append(deleted.reshape(img.shape))

        inserted = np.full_like(img, fill).reshape(img.shape[0], -1)
        inserted[:, order[:k]] = img.reshape(img.shape[0], -1)[:, order[:k]]
        insertion_imgs.append(inserted.reshape(img.shape))

    del_curve = _probs(deletion_imgs)
    ins_curve = _probs(insertion_imgs)
    return {
        "target": cls,
        "fraction": fractions,
        "deletion": del_curve,
        "insertion": ins_curve,
        "deletion_auc": float(np.trapezoid(del_curve, fractions)),
        "insertion_auc": float(np.trapezoid(ins_curve, fractions)),
    }
