"""The quality-control protocol.

For every perturbation family the protocol answers two questions and keeps them
apart, which is the whole point:

1. Does the *prediction* change? Balanced accuracy, macro AUC, expected
   calibration error and the fraction of predictions that agree with the
   baseline, over the evaluation subset.
2. Does the *explanation* change when the prediction does not? Saliency maps are
   compared only on the images whose predicted class is unchanged, so that any
   drop in agreement cannot be attributed to the model having changed its mind.

Two further tables are written: the faithfulness of each explainer at baseline
(deletion and insertion areas) and the cascading-randomisation sanity check.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset
from .evaluate import load_checkpoint, predict_probs
from .explain import EXPLAINERS
from .metrics import (
    cascading_randomization,
    deletion_insertion,
    expected_calibration_error,
    saliency_agreement,
    summarise_performance,
)
from .perturb import PERTURBATIONS, apply_to_batch, baseline_level, levels
from .utils import ensure_dir, load_config, seed_everything


def _explainer_kwargs(cfg: dict, method: str) -> dict:
    return dict(cfg["experiment"].get("explainer_kwargs", {}).get(method, {}))


def _saliency_batch(
    model, method: str, images: np.ndarray, targets: np.ndarray, **kwargs
) -> list[np.ndarray]:
    fn = EXPLAINERS[method]
    return [fn(model, images[i], target=int(targets[i]), **kwargs) for i in range(len(images))]


def run(cfg: dict) -> dict[str, Path]:
    """Run the full protocol and write the result tables. Returns their paths."""
    seed_everything(cfg["seed"])
    exp = cfg["experiment"]
    out_dir = ensure_dir(Path(exp["out_dir"]) / "tables")

    data = load_dataset(**cfg["dataset"], seed=cfg["seed"])
    model, _ = load_checkpoint(cfg["train"]["checkpoint"])

    # Perturbations are applied to the object, not to an image that has already
    # been through the simulator, so the evaluation set is the raw test split.
    x_source = data.x_test_raw if data.x_test_raw is not None else data.x_test
    n_eval = min(int(exp["n_eval"]), len(x_source))
    x_eval, y_eval = x_source[:n_eval], data.y_test[:n_eval]
    methods = list(exp["explainers"])
    families = list(exp["perturbations"])

    started = time.time()
    perf_rows: list[dict] = []
    stability_rows: list[dict] = []

    for kind in families:
        base_level = baseline_level(kind)
        x_base = apply_to_batch(x_eval, kind, base_level, seed=cfg["seed"])
        probs_base = predict_probs(model, x_base)
        pred_base = probs_base.argmax(axis=1)

        # Explain only images the baseline model gets right: an explanation of a
        # wrong prediction is a different object of study.
        correct = np.flatnonzero(pred_base == y_eval)
        explain_idx = correct[: int(exp["n_explain"])]
        saliency_base = {
            m: _saliency_batch(
                model, m, x_base[explain_idx], pred_base[explain_idx], **_explainer_kwargs(cfg, m)
            )
            for m in methods
        }

        for level in levels(kind):
            x_pert = apply_to_batch(x_eval, kind, level, seed=cfg["seed"] + 1)
            probs = predict_probs(model, x_pert)
            pred = probs.argmax(axis=1)

            row = {
                "perturbation": kind,
                "level": level,
                "unit": PERTURBATIONS[kind]["unit"],
                "n_eval": int(n_eval),
                "prediction_agreement": float((pred == pred_base).mean()),
                "ece": expected_calibration_error(y_eval, probs),
                "mean_confidence": float(probs.max(axis=1).mean()),
            }
            row.update(summarise_performance(y_eval, probs))
            perf_rows.append(row)

            unchanged = [i for i in explain_idx if pred[i] == pred_base[i]]
            keep = [k for k, i in enumerate(explain_idx) if pred[i] == pred_base[i]]
            for method in methods:
                if not unchanged:
                    stability_rows.append(
                        {"perturbation": kind, "level": level, "explainer": method, "n_images": 0}
                    )
                    continue
                maps = _saliency_batch(
                    model,
                    method,
                    x_pert[unchanged],
                    pred_base[unchanged],
                    **_explainer_kwargs(cfg, method),
                )
                records = [
                    saliency_agreement(saliency_base[method][k], m)
                    for k, m in zip(keep, maps, strict=False)
                ]
                values = pd.DataFrame(records)
                stability_rows.append(
                    {
                        "perturbation": kind,
                        "level": level,
                        "explainer": method,
                        "n_images": len(records),
                        "spearman_mean": float(values["spearman"].mean()),
                        "spearman_std": float(values["spearman"].std(ddof=1)) if len(values) > 1 else 0.0,
                        "ssim_mean": float(values["ssim"].mean()),
                        "top_k_jaccard_mean": float(values["top_k_jaccard"].mean()),
                    }
                )
            print(f"  {kind:>9} = {str(level):>12}  agreement {row['prediction_agreement']:.3f}  "
                  f"balanced accuracy {row['balanced_accuracy']:.3f}")

    # ---------------------------------------------------------------- faithfulness
    faith_rows: list[dict] = []
    x_ref = apply_to_batch(x_eval, families[0], baseline_level(families[0]), seed=cfg["seed"])
    probs_ref = predict_probs(model, x_ref)
    pred_ref = probs_ref.argmax(axis=1)
    idx = np.flatnonzero(pred_ref == y_eval)[: int(exp.get("n_faithfulness", 32))]
    rng = np.random.default_rng(cfg["seed"])
    for method in methods:
        for i in idx:
            sal = EXPLAINERS[method](
                model, x_ref[i], target=int(pred_ref[i]), **_explainer_kwargs(cfg, method)
            )
            curves = deletion_insertion(
                model, x_ref[i], sal, target=int(pred_ref[i]), steps=int(exp["faithfulness_steps"])
            )
            random_map = rng.random(sal.shape)
            random_curves = deletion_insertion(
                model, x_ref[i], random_map, target=int(pred_ref[i]), steps=int(exp["faithfulness_steps"])
            )
            faith_rows.append(
                {
                    "explainer": method,
                    "image": int(i),
                    "deletion_auc": curves["deletion_auc"],
                    "insertion_auc": curves["insertion_auc"],
                    "deletion_auc_random": random_curves["deletion_auc"],
                    "insertion_auc_random": random_curves["insertion_auc"],
                }
            )

    # ----------------------------------------------------------------- sanity check
    sanity_rows: list[dict] = []
    for method in methods:
        for i in idx[: int(exp.get("n_sanity", 8))]:
            for record in cascading_randomization(
                model,
                x_ref[i],
                EXPLAINERS[method],
                target=int(pred_ref[i]),
                seed=cfg["seed"],
                **_explainer_kwargs(cfg, method),
            ):
                sanity_rows.append({"explainer": method, "image": int(i), **record})

    paths = {
        "performance": out_dir / "performance_under_shift.csv",
        "stability": out_dir / "saliency_stability.csv",
        "faithfulness": out_dir / "faithfulness.csv",
        "sanity": out_dir / "sanity_checks.csv",
    }
    pd.DataFrame(perf_rows).to_csv(paths["performance"], index=False)
    pd.DataFrame(stability_rows).to_csv(paths["stability"], index=False)
    pd.DataFrame(faith_rows).to_csv(paths["faithfulness"], index=False)
    pd.DataFrame(sanity_rows).to_csv(paths["sanity"], index=False)
    print(f"wrote {len(paths)} tables to {out_dir} in {time.time() - started:.1f} s")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the explanation quality-control protocol.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
