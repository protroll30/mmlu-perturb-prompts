from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _macro_accuracy_pivot(metrics: dict[str, Any]) -> pd.DataFrame:
    records = metrics.get("macro_accuracy_by_model_condition", [])
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    pivot = df.pivot(
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

    pivot = _macro_accuracy_pivot(metrics)
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

    macro_records = metrics.get("accuracy_delta", [])
    if not macro_records:
        return []

    delta_df = pd.DataFrame(macro_records)
    perturbed_conditions = sorted(delta_df["condition_id"].unique())

    output_paths: list[Path] = []
    for condition_id in perturbed_conditions:
        perturbation_type = _perturbation_type_for_condition(
            pd.DataFrame(metrics.get("accuracy_by_model_condition_subject", [])),
            condition_id,
        )
        merged = delta_df[delta_df["condition_id"] == condition_id][
            [
                "model_id",
                "original_macro_accuracy",
                "perturbed_macro_accuracy",
            ]
        ].rename(
            columns={
                "original_macro_accuracy": "original_accuracy",
                "perturbed_macro_accuracy": "perturbed_accuracy",
            }
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        if merged.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
        else:
            ax.scatter(
                merged["original_accuracy"],
                merged["perturbed_accuracy"],
                alpha=0.9,
                s=80,
            )
            for _, row in merged.iterrows():
                ax.annotate(
                    row["model_id"],
                    (row["original_accuracy"], row["perturbed_accuracy"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
            lims = [
                np.min([merged["original_accuracy"].min(), merged["perturbed_accuracy"].min()]),
                np.max([merged["original_accuracy"].max(), merged["perturbed_accuracy"].max()]),
            ]
            pad = max(0.05, (lims[1] - lims[0]) * 0.1)
            lims = [lims[0] - pad, lims[1] + pad]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Original macro accuracy (matched set)")
            ax.set_ylabel("Perturbed macro accuracy (matched set)")
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
