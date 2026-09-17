"""The perturbation module is the part of this repository that makes a physical
claim, so it is the part with the most tests."""

import numpy as np
import pytest

from ctxaiqc.perturb import PERTURBATIONS, apply_perturbation, apply_to_batch, ct_roundtrip, levels
from ctxaiqc.perturb.acquisition import from_hu, hu_to_mu, to_hu


def test_hu_conversion_roundtrip():
    x = np.linspace(0, 1, 11)
    assert np.allclose(from_hu(to_hu(x)), x, atol=1e-6)


def test_hu_to_mu_matches_the_hu_definition():
    # 0 HU is water by definition, -1000 HU is vacuum/air.
    assert hu_to_mu(np.array([0.0]))[0] == pytest.approx(0.02, rel=1e-6)
    assert hu_to_mu(np.array([-1000.0]))[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kind", sorted(PERTURBATIONS))
def test_shape_dtype_and_range_are_preserved(phantom, kind):
    for level in levels(kind):
        out = apply_perturbation(phantom, kind, level, seed=0)
        assert out.shape == phantom.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("kind", sorted(PERTURBATIONS))
def test_same_seed_gives_identical_output(phantom, kind):
    level = levels(kind)[-1]
    a = apply_perturbation(phantom, kind, level, seed=7)
    b = apply_perturbation(phantom, kind, level, seed=7)
    assert np.array_equal(a, b)


def test_noise_increases_monotonically_as_dose_falls(phantom):
    reference = ct_roundtrip(phantom, dose_fraction=None)
    errors = [
        float(np.sqrt(np.mean((ct_roundtrip(phantom, dose_fraction=d, seed=3) - reference) ** 2)))
        for d in (1.0, 0.25, 0.05)
    ]
    assert errors[0] < errors[1] < errors[2], errors


def test_dose_noise_is_random_across_seeds(phantom):
    a = ct_roundtrip(phantom, dose_fraction=0.1, seed=1)
    b = ct_roundtrip(phantom, dose_fraction=0.1, seed=2)
    assert not np.array_equal(a, b)


def test_smoother_kernels_reduce_high_frequency_content(phantom):
    def roughness(img):
        return float(np.mean(np.abs(np.diff(img, axis=0))) + np.mean(np.abs(np.diff(img, axis=1))))

    sharp = roughness(ct_roundtrip(phantom, dose_fraction=None, filter_name="ramp"))
    smooth = roughness(ct_roundtrip(phantom, dose_fraction=None, filter_name="hann"))
    assert smooth < sharp


def test_hu_shift_moves_the_mean_in_the_right_direction(phantom):
    shifted = apply_perturbation(phantom, "hu_shift", 100.0)
    assert shifted[phantom > 0.1].mean() > phantom[phantom > 0.1].mean()


def test_batch_helper_keeps_layout_and_varies_noise_per_image(phantom):
    batch = np.repeat(phantom[None, None], 3, axis=0)
    out = apply_to_batch(batch, "dose", 0.05, seed=0)
    assert out.shape == batch.shape
    assert not np.array_equal(out[0, 0], out[1, 0])
