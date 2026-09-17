"""End-to-end smoke tests. Everything here runs on synthetic phantoms, on CPU,
with no network access, in a few seconds."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ctxaiqc.data import load_dataset  # noqa: E402
from ctxaiqc.explain import EXPLAINERS, explain  # noqa: E402
from ctxaiqc.metrics import cascading_randomization, deletion_insertion, mc_dropout_probs  # noqa: E402
from ctxaiqc.models import build_model  # noqa: E402
from ctxaiqc.utils import load_config, seed_everything  # noqa: E402

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from ctxaiqc.train import train_model

    cfg = load_config(SMOKE)
    out = tmp_path_factory.mktemp("smoke")
    cfg["train"]["checkpoint"] = str(out / "model.pt")
    cfg["experiment"]["out_dir"] = str(out)
    train_model(cfg)
    return cfg


def test_synthetic_dataset_is_well_formed():
    data = load_dataset("synthetic", size=32, subset={"train": 8, "val": 4, "test": 4})
    assert data.x_train.shape[1:] == (1, 32, 32)
    assert data.x_train.dtype == np.float32
    assert set(np.unique(data.y_train)) <= {0, 1}


@pytest.mark.parametrize("method", sorted(EXPLAINERS))
def test_explainers_return_a_normalised_map(method):
    seed_everything(0)
    data = load_dataset("synthetic", size=32, subset={"train": 4, "val": 2, "test": 2})
    model = build_model("smallcnn", n_classes=2, width=8)
    sal = explain(method, model, data.x_test[0], **({"steps": 4} if method == "integrated_gradients" else {}))
    assert sal.shape == (32, 32)
    assert sal.min() >= 0.0 and sal.max() <= 1.0


def test_training_writes_a_loadable_checkpoint(trained):
    from ctxaiqc.evaluate import load_checkpoint, predict_probs

    model, ckpt = load_checkpoint(trained["train"]["checkpoint"])
    assert ckpt["n_classes"] == 2
    data = load_dataset(**trained["dataset"], seed=trained["seed"])
    probs = predict_probs(model, data.x_test)
    assert probs.shape == (len(data.x_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_mc_dropout_produces_non_zero_spread(trained):
    from ctxaiqc.evaluate import load_checkpoint

    model, _ = load_checkpoint(trained["train"]["checkpoint"])
    data = load_dataset(**trained["dataset"], seed=trained["seed"])
    mean, std = mc_dropout_probs(model, data.x_test[:8], n_samples=5)
    assert mean.shape == std.shape == (8, 2)
    assert std.sum() > 0


def test_deletion_insertion_curves_are_bounded(trained):
    from ctxaiqc.evaluate import load_checkpoint

    model, _ = load_checkpoint(trained["train"]["checkpoint"])
    data = load_dataset(**trained["dataset"], seed=trained["seed"])
    sal = explain("gradcam", model, data.x_test[0])
    out = deletion_insertion(model, data.x_test[0], sal, steps=4)
    assert 0.0 <= out["deletion_auc"] <= 1.0
    assert 0.0 <= out["insertion_auc"] <= 1.0
    assert len(out["deletion"]) == len(out["fraction"]) == 5


def test_cascading_randomisation_visits_every_parameterised_layer(trained):
    from ctxaiqc.evaluate import load_checkpoint

    model, _ = load_checkpoint(trained["train"]["checkpoint"])
    data = load_dataset(**trained["dataset"], seed=trained["seed"])
    before = [p.detach().clone() for p in model.parameters()]
    records = cascading_randomization(model, data.x_test[0], EXPLAINERS["gradcam"], seed=0)
    assert len(records) >= 6
    assert {"layer", "spearman", "ssim", "top_k_jaccard"} <= set(records[0])
    # the model handed in must not be modified
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=False))


def test_full_protocol_writes_all_tables(trained, tmp_path):
    import pandas as pd

    from ctxaiqc.report import family_figure, main_figure, summary_table
    from ctxaiqc.run_experiment import run

    cfg = dict(trained)
    cfg["experiment"] = dict(cfg["experiment"], out_dir=str(tmp_path))
    paths = run(cfg)
    for name, path in paths.items():
        assert path.exists(), name
        assert len(pd.read_csv(path)) > 0, name

    tables = tmp_path / "tables"
    assert main_figure(tables, tmp_path / "figures" / "main.png").exists()
    assert family_figure(tables, tmp_path / "figures" / "family.png").exists()
    assert summary_table(tables, tables / "summary.md").exists()
