from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _macro_accuracy_pivot(accuracy_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not accuracy_records:
        return pd.DataFrame()
    df = pd.DataFrame(accuracy_records)
    macro = (
        df.groupby(["model_id", "condition_id"], as_index=False)
        .agg(macro_accuracy=("accuracy", "mean"))
    )
    pivot = macro.pivot(
        index="model_id",
        columns="condition_id",
        values="macro_accuracy",
    )
    return pivot.sort_index()


def plot_accuracy_heatmap(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = _macro_accuracy_pivot(metrics.get("accuracy_by_model_condition_subject", []))
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.8), max(4, pivot.shape[0] * 0.6)))

    if pivot.empty:
        ax.text(0.5, 0.5, "No accuracy data", ha="center", va="center")
        ax.axis("off")
    else:
        data = pivot.fillna(0).to_numpy()
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Model")
        fig.colorbar(im, ax=ax, label="Macro accuracy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_flip_rate_by_subject(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flip_by_subject = metrics.get("flip_rates", {}).get("by_subject", {})
    subjects = sorted(flip_by_subject.keys())
    values = [flip_by_subject[s] for s in subjects]

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.35), 6))
    if not subjects:
        ax.text(0.5, 0.5, "No flip rate data", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(subjects, values, color="steelblue")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Mean flip rate")
        ax.set_title("Flip rate distribution by subject")
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _perturbation_type_for_condition(
    accuracy_df: pd.DataFrame,
    condition_id: str,
) -> str:
    rows = accuracy_df[accuracy_df["condition_id"] == condition_id]
    if rows.empty:
        return "unknown"
    return str(rows.iloc[0].get("perturbation_type", "unknown"))


def plot_accuracy_scatter(
    metrics: dict[str, Any],
    figures_dir: str | Path,
    original_condition: str = "original",
) -> list[Path]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    accuracy_records = metrics.get("accuracy_by_model_condition_subject", [])
    if not accuracy_records:
        return []

    df = pd.DataFrame(accuracy_records)
    perturbed_conditions = sorted(
        c for c in df["condition_id"].unique() if c != original_condition
    )

    output_paths: list[Path] = []
    for condition_id in perturbed_conditions:
        perturbation_type = _perturbation_type_for_condition(df, condition_id)
        orig = df[df["condition_id"] == original_condition][
            ["model_id", "subject", "accuracy"]
        ].rename(columns={"accuracy": "original_accuracy"})
        pert = df[df["condition_id"] == condition_id][
            ["model_id", "subject", "accuracy"]
        ].rename(columns={"accuracy": "perturbed_accuracy"})
        merged = orig.merge(pert, on=["model_id", "subject"], how="inner")

        fig, ax = plt.subplots(figsize=(7, 7))
        if merged.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
        else:
            ax.scatter(
                merged["original_accuracy"],
                merged["perturbed_accuracy"],
                alpha=0.7,
            )
            lims = [
                np.min([merged["original_accuracy"].min(), merged["perturbed_accuracy"].min()]),
                np.max([merged["original_accuracy"].max(), merged["perturbed_accuracy"].max()]),
            ]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Original accuracy")
            ax.set_ylabel("Perturbed accuracy")
            ax.set_title(f"{perturbation_type} ({condition_id})")

        safe_name = condition_id.replace("/", "_").replace("+", "_")
        output_path = figures_dir / f"scatter_{safe_name}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def generate_all_figures(
    metrics: dict[str, Any],
    figures_dir: str | Path,
) -> list[Path]:
    figures_dir = Path(figures_dir)
    paths = [
        plot_accuracy_heatmap(metrics, figures_dir / "accuracy_heatmap.png"),
        plot_flip_rate_by_subject(metrics, figures_dir / "flip_rate_by_subject.png"),
    ]
    paths.extend(plot_accuracy_scatter(metrics, figures_dir))
    return paths
