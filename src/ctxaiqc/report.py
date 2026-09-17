"""Figures and the summary table for the README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .perturb import PERTURBATIONS  # noqa: E402
from .utils import ensure_dir, load_config  # noqa: E402

PRIMARY = "#1b365d"
SECONDARY = "#c1553b"


def main_figure(tables_dir: Path, out_path: Path, family: str = "dose") -> Path:
    """Prediction and explanation against the level of one perturbation family.

    The x axis is the ordered list of levels, reference first, labelled with the
    level itself. Plotting the index rather than the value keeps the figure
    identical for families whose levels are numbers and for families whose levels
    are names, such as the reconstruction kernel.

    The point of the figure is the contrast between the two axes: where the
    left-hand curves stay flat and the right-hand ones fall, the model's output
    survived the acquisition change and the explanation offered for it did not.
    """
    perf = pd.read_csv(tables_dir / "performance_under_shift.csv")
    stab = pd.read_csv(tables_dir / "saliency_stability.csv")
    perf = perf[perf["perturbation"] == family]
    stab = stab[stab["perturbation"] == family]
    if perf.empty:
        raise ValueError(f"no rows for perturbation {family!r} in {tables_dir}")

    order = [str(level) for level in PERTURBATIONS[family]["levels"]]
    index = {label: i for i, label in enumerate(order)}
    perf = perf.assign(_x=perf["level"].astype(str).map(index)).sort_values("_x")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(perf["_x"], perf["balanced_accuracy"], "o-", color=PRIMARY, label="Balanced accuracy")
    ax.plot(perf["_x"], perf["prediction_agreement"], "s--", color=PRIMARY, alpha=0.55,
            label="Predictions unchanged vs reference")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel(f"{PERTURBATIONS[family]['label']}  [{PERTURBATIONS[family]['unit']}]")
    ax.set_ylabel("Prediction", color=PRIMARY)
    ax.set_ylim(0, 1.04)

    ax2 = ax.twinx()
    for method, marker in zip(sorted(stab["explainer"].unique()), ["^-", "v-", "d-"], strict=False):
        sub = stab[stab["explainer"] == method].assign(
            _x=stab.loc[stab["explainer"] == method, "level"].astype(str).map(index)
        ).sort_values("_x")
        ax2.plot(sub["_x"], sub["top_k_jaccard_mean"], marker, color=SECONDARY,
                 label=f"Top-10% overlap ({method})")
        ax2.plot(sub["_x"], sub["spearman_mean"], marker.replace("-", ":"), color=SECONDARY,
                 alpha=0.5, label=f"Rank correlation ({method})")
    ax2.set_ylabel("Explanation (agreement with reference map)", color=SECONDARY)
    ax2.tick_params(axis="y", labelcolor=SECONDARY)
    ax2.set_ylim(0, 1.04)

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.set_title("Reference acquisition on the left, increasing change to the right", fontsize=10)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def family_figure(tables_dir: Path, out_path: Path) -> Path:
    """Saliency agreement at the strongest level of every perturbation family."""
    stab = pd.read_csv(tables_dir / "saliency_stability.csv")
    worst = (
        stab.sort_values("top_k_jaccard_mean")
        .groupby(["perturbation", "explainer"], as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    families = sorted(worst["perturbation"].unique())
    methods = sorted(worst["explainer"].unique())
    width = 0.8 / max(len(methods), 1)
    for j, method in enumerate(methods):
        sub = worst[worst["explainer"] == method].set_index("perturbation").reindex(families)
        ax.bar(
            [i + j * width for i in range(len(families))],
            sub["top_k_jaccard_mean"].to_numpy(),
            width=width,
            label=method,
            color=PRIMARY if j == 0 else SECONDARY,
            alpha=0.85,
        )
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(families))])
    ax.set_xticklabels(families)
    ax.set_ylabel("Saliency agreement (top-10% overlap)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Worst-case explanation agreement, by perturbation family")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def summary_table(tables_dir: Path, out_path: Path) -> Path:
    """One markdown table: accuracy kept, explanation agreement lost."""
    perf = pd.read_csv(tables_dir / "performance_under_shift.csv")
    stab = pd.read_csv(tables_dir / "saliency_stability.csv")
    merged = (
        stab.merge(perf, on=["perturbation", "level"], how="left")
        .loc[:, [
            "perturbation",
            "level",
            "explainer",
            "balanced_accuracy",
            "prediction_agreement",
            "spearman_mean",
            "ssim_mean",
            "top_k_jaccard_mean",
            "n_images",
        ]]
        .round(3)
    )
    ensure_dir(out_path.parent)
    out_path.write_text(merged.to_markdown(index=False), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the figures and the summary table.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--family", default="dose", help="perturbation family for the main figure")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg["experiment"]["out_dir"])
    tables = root / "tables"
    figures = ensure_dir(root / "figures")

    print(main_figure(tables, figures / "stability_vs_dose.png", family=args.family))
    print(family_figure(tables, figures / "stability_by_family.png"))
    print(summary_table(tables, tables / "summary.md"))


if __name__ == "__main__":  # pragma: no cover
    main()
