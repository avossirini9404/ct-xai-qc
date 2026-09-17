"""Physics-based simulation of acquisition and reconstruction changes.

The perturbations in this module are the reason the repository exists. A model
that is tested only against additive Gaussian noise is being asked a question no
scanner ever asks. What actually changes between CT examinations, and between
institutions, is the photon statistics (tube current-time product and therefore
dose), the reconstruction kernel, the tube voltage and contrast phase, and the
effective slice thickness. Those are the four families simulated here.

Image convention
----------------
Images are float32 arrays in ``[0, 1]``. The mapping to Hounsfield units is
fixed and explicit: ``0 -> HU_MIN`` (-1000 HU, air) and ``1 -> HU_MAX``
(+1000 HU, dense bone). MedMNIST distributes 8-bit images whose original HU
window is not recoverable, so this is a stated convention, not a recovered
calibration; see "Scope and limitations" in the README.

Fair comparison
---------------
Forward projection followed by filtered back-projection is not the identity: it
blurs the image and zeroes the corners outside the reconstruction circle. If a
perturbed image went through that path and the reference image did not, part of
any measured difference would be the round trip rather than the perturbation.
Every family therefore defines its own *baseline level* (``baseline_level``),
which is passed through exactly the same code path as the perturbed levels.
Comparisons are always made against that baseline, never against the raw input.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import iradon, radon

HU_MIN = -1000.0
HU_MAX = 1000.0
MU_WATER = 0.02  # mm^-1, roughly water at the effective energy of a 120 kVp beam

__all__ = [
    "HU_MIN",
    "HU_MAX",
    "to_hu",
    "from_hu",
    "hu_to_mu",
    "mu_to_hu",
    "ct_roundtrip",
    "simulate",
    "reference_acquisition",
    "hu_shift",
    "contrast_change",
    "slice_thickness",
    "PERTURBATIONS",
    "levels",
    "baseline_level",
    "apply_perturbation",
    "apply_to_batch",
]


# --------------------------------------------------------------------------- #
# Unit conversions
# --------------------------------------------------------------------------- #
def to_hu(x: np.ndarray) -> np.ndarray:
    """Map normalised intensities in [0, 1] to the stated HU window."""
    return HU_MIN + np.asarray(x, dtype=np.float64) * (HU_MAX - HU_MIN)


def from_hu(hu: np.ndarray) -> np.ndarray:
    """Map Hounsfield units back to [0, 1], clipping outside the window."""
    x = (np.asarray(hu, dtype=np.float64) - HU_MIN) / (HU_MAX - HU_MIN)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def hu_to_mu(hu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    """Linear attenuation coefficient from HU, by the definition of the HU scale."""
    return mu_water * (1.0 + np.asarray(hu, dtype=np.float64) / 1000.0)


def mu_to_hu(mu: np.ndarray, mu_water: float = MU_WATER) -> np.ndarray:
    return 1000.0 * (np.asarray(mu, dtype=np.float64) / mu_water - 1.0)


def _angles(n_angles: int) -> np.ndarray:
    return np.linspace(0.0, 180.0, int(n_angles), endpoint=False)


def _circular_mask(shape: tuple[int, int]) -> np.ndarray:
    """The reconstruction circle inscribed in the image.

    The centre and radius follow the convention of ``skimage.transform.radon``
    exactly, so that nothing is left outside the circle for the forward
    projection to discard silently.
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 2
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #
def ct_roundtrip(
    img: np.ndarray,
    dose_fraction: float | None = None,
    filter_name: str = "ramp",
    n_angles: int = 180,
    i0: float = 1.0e5,
    pixel_mm: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Forward project, optionally add photon noise, and reconstruct.

    The image is converted to linear attenuation coefficients, projected with the
    Radon transform, and turned into detected photon counts through the
    Beer-Lambert law ``N = I0 exp(-p)``. Counts are drawn from a Poisson
    distribution, which is the physical noise model for photon counting in the
    quantum-limited regime, and converted back to line integrals before filtered
    back-projection. Lowering ``dose_fraction`` lowers ``I0`` and therefore
    raises the relative noise, exactly as reducing the tube current-time product
    does on a scanner.

    Parameters
    ----------
    img
        2-D array in [0, 1].
    dose_fraction
        Fraction of the reference photon flux. ``None`` performs a noiseless
        round trip, which is the baseline used for kernel comparisons.
    filter_name
        Reconstruction filter passed to ``skimage.transform.iradon``
        ("ramp", "shepp-logan", "cosine", "hamming", "hann").
    n_angles
        Number of projection angles over 180 degrees.
    i0
        Reference photon fluence per ray at full dose.
    pixel_mm
        Pixel size, used to convert attenuation per mm into attenuation per pixel.
    seed
        Seed for the Poisson draw. Runs with the same seed are bit-identical.

    Returns
    -------
    np.ndarray
        Reconstructed image in [0, 1], same shape as the input, masked to the
        reconstruction circle.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {img.shape}")
    if dose_fraction is not None and dose_fraction <= 0:
        raise ValueError("dose_fraction must be positive or None")

    theta = _angles(n_angles)
    mask = _circular_mask(img.shape)
    mu = hu_to_mu(to_hu(img)) * pixel_mm  # attenuation per pixel
    mu = np.maximum(mu, 0.0) * mask  # nothing is scanned outside the reconstruction circle
    sino = radon(mu, theta=theta, circle=True)

    if dose_fraction is not None:
        rng = np.random.default_rng(seed)
        flux = i0 * float(dose_fraction)
        counts = rng.poisson(flux * np.exp(-sino))
        counts = np.maximum(counts, 1.0)  # guard the logarithm against zero counts
        sino = -np.log(counts / flux)

    recon_mu = iradon(sino, theta=theta, filter_name=filter_name, circle=True)
    out = from_hu(mu_to_hu(recon_mu / pixel_mm))
    out[~mask] = 0.0
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Perturbation families
# --------------------------------------------------------------------------- #
# Every image the pipeline produces, including the reference, is the output of a
# single simulated acquisition. A family changes one setting of that acquisition
# and leaves the others at their reference value. ``mode`` says where in the
# chain the change acts:
#
#   dose    -> photon flux, before reconstruction
#   kernel  -> reconstruction filter
#   pre     -> the object itself, before projection (contrast, material change)
#   post    -> the reconstructed image (calibration offset, smoothing)
#
# Modelling a calibration offset after reconstruction and a contrast change
# before projection is not cosmetic: a beam-hardening-driven contrast change
# alters the projections and is therefore inseparable from the reconstruction,
# while a scanner calibration offset is added to the reconstructed HU values.


def hu_shift(img: np.ndarray, delta_hu: float) -> np.ndarray:
    """Add a constant offset in Hounsfield units (scanner calibration drift)."""
    return from_hu(to_hu(img) + float(delta_hu))


def contrast_change(img: np.ndarray, gain: float, center_hu: float = 0.0) -> np.ndarray:
    """Scale HU contrast about a centre, a coarse surrogate for a change of kVp.

    Raising the tube voltage hardens the beam and compresses soft-tissue
    contrast; iodinated contrast medium does the opposite in enhancing
    structures. Both are modelled here as a single affine gain in HU applied to
    the object before projection, which reproduces the direction of the effect
    but not its material dependence.
    """
    hu = to_hu(img)
    return from_hu(center_hu + float(gain) * (hu - center_hu))


def slice_thickness(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Blur the reconstructed image, as a surrogate for a thicker slice.

    Partial-volume averaging over a thicker slice acts along the axis a 2-D
    benchmark does not have, so this is an in-plane stand-in for it, not a
    simulation of it.
    """
    if sigma_px <= 0:
        return np.asarray(img, dtype=np.float32).copy()
    return gaussian_filter(np.asarray(img, dtype=np.float64), sigma=float(sigma_px)).astype(np.float32)


PERTURBATIONS: dict[str, dict[str, Any]] = {
    "dose": {
        "mode": "dose",
        "fn": None,
        "param": "dose_fraction",
        "levels": [1.0, 0.5, 0.2, 0.05, 0.02],
        "unit": "fraction of reference dose",
        "label": "Dose reduction (Poisson photon noise)",
    },
    "kernel": {
        "mode": "kernel",
        "fn": None,
        "param": "filter_name",
        "levels": ["ramp", "shepp-logan", "cosine", "hamming", "hann"],
        "unit": "reconstruction filter",
        "label": "Reconstruction kernel (sharp to smooth)",
    },
    "hu_shift": {
        "mode": "post",
        "fn": hu_shift,
        "param": "delta_hu",
        "levels": [0.0, 25.0, 50.0, 100.0, 200.0],
        "unit": "HU",
        "label": "Calibration offset",
    },
    "contrast": {
        "mode": "pre",
        "fn": contrast_change,
        "param": "gain",
        "levels": [1.0, 1.1, 1.2, 0.9, 0.8],
        "unit": "HU contrast gain",
        "label": "Contrast / tube voltage change",
    },
    "blur": {
        "mode": "post",
        "fn": slice_thickness,
        "param": "sigma_px",
        "levels": [0.0, 0.5, 1.0, 1.5, 2.0],
        "unit": "Gaussian sigma (pixels)",
        "label": "Effective slice thickness (surrogate)",
    },
}


def _check(kind: str) -> None:
    if kind not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {kind!r}; available: {sorted(PERTURBATIONS)}")


def levels(kind: str) -> Sequence[Any]:
    """Levels of a perturbation family, reference value first."""
    _check(kind)
    return PERTURBATIONS[kind]["levels"]


def baseline_level(kind: str) -> Any:
    """The level at which this family reproduces the reference acquisition."""
    return levels(kind)[0]


def simulate(
    img: np.ndarray,
    kind: str | None = None,
    level: Any = None,
    seed: int | None = None,
    reference_dose: float = 1.0,
    reference_filter: str = "ramp",
    **kw: Any,
) -> np.ndarray:
    """Simulate one acquisition of ``img``, optionally with one setting changed.

    ``img`` is the underlying object, not an image that has already been through
    the scanner. Called without a ``kind`` it produces the reference acquisition,
    which is what the classifier is trained on; called with one, it produces the
    same acquisition with a single setting moved away from its reference value.
    Keeping both on the same code path is what makes the comparison fair.
    """
    dose: float | None = reference_dose
    filter_name = reference_filter
    pre_fn = post_fn = None

    if kind is not None:
        _check(kind)
        spec = PERTURBATIONS[kind]
        mode = spec["mode"]
        if mode == "dose":
            dose = float(level)
        elif mode == "kernel":
            filter_name = str(level)
        elif mode == "pre":
            pre_fn = (spec["fn"], level)
        elif mode == "post":
            post_fn = (spec["fn"], level)
        else:  # pragma: no cover - guarded by the registry
            raise ValueError(f"unknown perturbation mode {mode!r}")

    x = np.asarray(img, dtype=np.float64)
    if pre_fn is not None:
        x = pre_fn[0](x, pre_fn[1])
    x = ct_roundtrip(x, dose_fraction=dose, filter_name=filter_name, seed=seed, **kw)
    if post_fn is not None:
        x = post_fn[0](x, post_fn[1])
    return np.asarray(x, dtype=np.float32)


def reference_acquisition(img: np.ndarray, seed: int | None = None, **kw: Any) -> np.ndarray:
    """The reference acquisition: every setting at its reference value."""
    return simulate(img, seed=seed, **kw)


def apply_perturbation(
    img: np.ndarray, kind: str, level: Any, seed: int | None = None, **kw: Any
) -> np.ndarray:
    """Simulate one acquisition with family ``kind`` set to ``level``."""
    return simulate(img, kind=kind, level=level, seed=seed, **kw)


def apply_to_batch(
    x: np.ndarray, kind: str | None, level: Any = None, seed: int = 0, **kw: Any
) -> np.ndarray:
    """Apply an acquisition to a batch shaped ``(N, 1, H, W)`` or ``(N, H, W)``.

    Each image gets its own noise realisation, derived deterministically from
    ``seed`` and the image index.
    """
    x = np.asarray(x)
    squeezed = x[:, 0] if x.ndim == 4 else x
    out = np.stack(
        [simulate(squeezed[i], kind=kind, level=level, seed=seed + i, **kw) for i in range(squeezed.shape[0])]
    )
    return out[:, None] if x.ndim == 4 else out
