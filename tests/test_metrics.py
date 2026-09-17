import numpy as np
import pytest

from ctxaiqc.metrics import (
    balanced_accuracy,
    expected_calibration_error,
    macro_auc,
    reliability_curve,
    saliency_agreement,
    summarise_agreement,
)


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 1, 0])
    probs = np.eye(2)[y] * 0.999 + 0.0005
    assert balanced_accuracy(y, probs) == 1.0
    assert macro_auc(y, probs) == 1.0


def test_ece_is_small_for_a_calibrated_model():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=p1.size) < p1).astype(int)
    probs = np.stack([1 - p1, p1], axis=1)
    assert expected_calibration_error(y, probs) < 0.03


def test_ece_is_large_for_an_overconfident_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    probs = np.full((2000, 2), 0.01)
    probs[np.arange(2000), rng.integers(0, 2, 2000)] = 0.99
    assert expected_calibration_error(y, probs) > 0.3


def test_reliability_curve_shapes_agree():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 2], size=500)
    y = (rng.uniform(size=500) < probs[:, 1]).astype(int)
    centres, acc, conf, weights = reliability_curve(y, probs, n_bins=10)
    assert centres.shape == acc.shape == conf.shape == weights.shape
    assert 0.99 < weights.sum() <= 1.0001


def test_identical_saliency_maps_agree_perfectly():
    rng = np.random.default_rng(0)
    a = rng.random((24, 24))
    scores = saliency_agreement(a, a)
    assert scores["spearman"] == 1.0
    assert scores["ssim"] > 0.999
    assert scores["top_k_jaccard"] == 1.0


def test_unrelated_saliency_maps_do_not_agree():
    rng = np.random.default_rng(0)
    scores = saliency_agreement(rng.random((24, 24)), rng.random((24, 24)))
    assert abs(scores["spearman"]) < 0.2
    assert scores["top_k_jaccard"] < 0.3


def test_summarise_agreement_reports_n_and_means():
    rng = np.random.default_rng(0)
    a = rng.random((16, 16))
    out = summarise_agreement([saliency_agreement(a, a), saliency_agreement(a, a)])
    assert out["n"] == 2
    assert out["spearman_mean"] == pytest.approx(1.0)
