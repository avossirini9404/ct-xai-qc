"""Dataset loading.

Two public CT-derived benchmarks from MedMNIST v2 are supported, plus a small
synthetic phantom dataset that needs no download and is what continuous
integration runs on.

``organamnist``
    Axial abdominal CT slices, 11 organ classes, cropped from the Liver Tumor
    Segmentation Benchmark (LiTS). Used as the default because the classes are
    spatially localised, which is what makes a saliency map worth inspecting.
``nodulemnist3d``
    Lung nodules from LIDC-IDRI, binary (benign / malignant). The volumes are
    28x28x28; this repository takes the central axial slice so that a single 2-D
    architecture serves both datasets. That is a deliberate simplification and it
    discards information the 3-D task provides.
``synthetic``
    Circular phantom with a higher-attenuation insert placed either centrally or
    peripherally. Two classes, generated on the fly, no network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATASETS = ("organamnist", "nodulemnist3d", "synthetic")


@dataclass
class DatasetBundle:
    """Arrays for one dataset, images as float32 ``(N, 1, H, W)`` in [0, 1]."""

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    class_names: list[str]
    x_test_raw: np.ndarray | None = None  # test images before simulated acquisition

    @property
    def image_size(self) -> int:
        return int(self.x_train.shape[-1])

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_classes": self.n_classes,
            "image_size": self.image_size,
            "n_train": int(len(self.x_train)),
            "n_val": int(len(self.x_val)),
            "n_test": int(len(self.x_test)),
        }


def _stratified_subset(x: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n is None or n >= len(x):
        return x, y
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    classes, counts = np.unique(y, return_counts=True)
    per_class = max(1, n // len(classes))
    for c in classes:
        idx = np.flatnonzero(y == c)
        keep.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    keep_arr = np.array(sorted(keep))
    return x[keep_arr], y[keep_arr]


def _to_nchw(images: np.ndarray) -> np.ndarray:
    """Normalise MedMNIST arrays to float32 (N, 1, H, W) in [0, 1]."""
    arr = np.asarray(images)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):  # (N, H, W, C)
        arr = arr[..., 0]
    if arr.ndim == 5:  # (N, C, D, H, W) volumes
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[1] == arr.shape[2] == arr.shape[3]:  # (N, D, H, W)
        arr = arr[:, arr.shape[1] // 2]  # central axial slice
    if arr.ndim != 3:
        raise ValueError(f"cannot interpret image array of shape {images.shape}")
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr[:, None]


def _load_medmnist(name: str, root: str, size: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "medmnist is required for this dataset. Install it with `pip install medmnist`, "
            "or use dataset: synthetic to run offline."
        ) from exc

    info = INFO[name]
    cls = getattr(medmnist, info["python_class"])
    splits = {}
    for split in ("train", "val", "test"):
        try:
            ds = cls(split=split, download=True, root=root, size=size)
        except TypeError:
            # medmnist < 3 has no size argument and only ships the 28-pixel release
            ds = cls(split=split, download=True, root=root)
        splits[split] = (_to_nchw(ds.imgs), np.asarray(ds.labels).reshape(-1).astype(np.int64))
    return splits


def _make_synthetic(n: int, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Phantom with a bright insert, central (class 0) or peripheral (class 1)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    centre = (size - 1) / 2.0
    body = ((yy - centre) ** 2 + (xx - centre) ** 2) <= (0.42 * size) ** 2
    images = np.zeros((n, size, size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        img = np.where(body, 0.55, 0.05).astype(np.float32)
        label = i % 2
        radius = 0.06 * size + rng.uniform(0, 0.02 * size)
        if label == 0:
            cy, cx = centre + rng.uniform(-2, 2), centre + rng.uniform(-2, 2)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            offset = 0.25 * size
            cy, cx = centre + offset * np.sin(angle), centre + offset * np.cos(angle)
        insert = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        img[insert] = 0.8
        img += rng.normal(0, 0.01, img.shape).astype(np.float32)
        images[i] = np.clip(img, 0, 1)
        labels[i] = label
    return images[:, None], labels



def _simulate_reference(x: np.ndarray, seed: int = 0, cache: Path | None = None) -> np.ndarray:
    """Acquire every image once, at the reference settings.

    The benchmark images are treated as the object, not as an image that has
    already been through a scanner. If the classifier were trained on those raw
    arrays, the *reference* level of every perturbation family would already be
    out of distribution, and a drop in performance would be measuring the
    simulator rather than the perturbation. Training on reference acquisitions
    removes that confound: the model only ever sees the output of one pipeline,
    and a family moves exactly one setting of it.
    """
    from ..perturb import reference_acquisition

    if cache is not None and cache.exists():
        return np.load(cache)["x"].astype(np.float32)
    out = np.stack([reference_acquisition(x[i][0], seed=seed + i) for i in range(len(x))])[:, None]
    out = out.astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=out)
    return out


def load_dataset(
    name: str = "organamnist",
    root: str = "data",
    size: int = 64,
    subset: dict[str, int] | None = None,
    reconstruct: bool = True,
    seed: int = 0,
) -> DatasetBundle:
    """Load one dataset as NumPy arrays.

    Parameters
    ----------
    name
        One of :data:`DATASETS`.
    root
        Directory for the MedMNIST download cache.
    size
        Image side in pixels. MedMNIST v2 publishes 28, 64, 128 and 224.
    subset
        Optional ``{"train": n, "val": n, "test": n}`` stratified caps, which is
        how the smoke tests and the continuous-integration run stay fast.
    reconstruct
        Acquire every image once at the reference settings (forward projection,
        photon noise at the reference dose, filtered back-projection), so that
        the classifier is trained on the same kind of image the perturbation
        experiments produce. The untouched test images are kept in
        ``x_test_raw`` and are what :mod:`ctxaiqc.run_experiment` perturbs.
        MedMNIST results are cached under ``root``; the first call takes a few
        minutes.
    seed
        Seed for subsampling and for the synthetic generator.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {DATASETS}")

    if name == "synthetic":
        n_train = (subset or {}).get("train", 256)
        n_val = (subset or {}).get("val", 64)
        n_test = (subset or {}).get("test", 128)
        x_tr, y_tr = _make_synthetic(n_train, size, seed)
        x_va, y_va = _make_synthetic(n_val, size, seed + 1)
        x_te, y_te = _make_synthetic(n_test, size, seed + 2)
        x_te_raw = x_te
        if reconstruct:
            x_tr = _simulate_reference(x_tr, seed=seed)
            x_va = _simulate_reference(x_va, seed=seed + 10_000)
            x_te = _simulate_reference(x_te, seed=seed + 20_000)
        return DatasetBundle(
            name, x_tr, y_tr, x_va, y_va, x_te, y_te, 2,
            ["central insert", "peripheral insert"], x_test_raw=x_te_raw,
        )

    splits = _load_medmnist(name, root=root, size=size)
    subset = subset or {}
    out, raw = {}, {}
    for split, (x, y) in splits.items():
        raw[split] = _stratified_subset(x, y, subset.get(split), seed)[0]
        if reconstruct:
            cache = Path(root) / "cache" / f"{name}_{size}_{split}_reference.npz"
            x = _simulate_reference(x, seed=seed, cache=cache)
        out[split] = _stratified_subset(x, y, subset.get(split), seed)

    n_classes = int(max(int(y.max()) for _, y in splits.values()) + 1)
    from medmnist import INFO  # local import: only reached when medmnist is present

    labels = INFO[name]["label"]
    class_names = [labels[str(i)] for i in range(n_classes)]
    return DatasetBundle(
        name,
        out["train"][0],
        out["train"][1],
        out["val"][0],
        out["val"][1],
        out["test"][0],
        out["test"][1],
        n_classes,
        class_names,
        x_test_raw=raw["test"],
    )
