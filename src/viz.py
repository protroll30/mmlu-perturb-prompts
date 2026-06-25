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


def plot_flip_rate_by_perturbation_type(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Bar chart of pooled flip rate per perturbation type, sorted descending."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_type = metrics.get("flip_rates", {}).get("by_perturbation_type", {})
    if not by_type:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No flip rate data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    sorted_items = sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)
    labels = [kv[0] for kv in sorted_items]
    values = [kv[1] for kv in sorted_items]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color="steelblue")
    ax.bar_label(bars, fmt="{:.1%}", padding=3, fontsize=9)
    ax.set_xlabel("Perturbation type")
    ax.set_ylabel("Pooled flip rate (matched set)")
    ax.set_title("Answer flip rate by perturbation type")
    ax.set_ylim(0, min(1.0, max(values) * 1.25))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_flip_rate_by_model_perturbation_type(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Grouped bar chart of flip rate per perturbation type, one bar per model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_model_type: dict[str, dict[str, float]] = metrics.get(
        "flip_rates", {}
    ).get("by_model_perturbation_type", {})
    model_ids = sorted(by_model_type.keys())
    all_types = sorted(
        {ptype for m in by_model_type.values() for ptype in m},
        key=lambda t: -np.mean(
            [by_model_type[m].get(t, 0.0) for m in model_ids]
        ),
    )

    if not model_ids or not all_types:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No flip rate data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    x = np.arange(len(all_types))
    width = 0.8 / max(len(model_ids), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(all_types) * 1.4), 5))

    for i, model_id in enumerate(model_ids):
        vals = [by_model_type[model_id].get(t, float("nan")) for t in all_types]
        offset = (i - (len(model_ids) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model_id, alpha=0.85)
        ax.bar_label(bars, fmt="{:.0%}", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=20, ha="right")
    ax.set_xlabel("Perturbation type")
    ax.set_ylabel("Flip rate (matched set)")
    ax.set_title("Per-model flip rate by perturbation type")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", loc="upper right", fontsize=8)

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
        plot_flip_rate_by_perturbation_type(
            metrics, figures_dir / "flip_rate_by_perturbation_type.png"
        ),
        plot_flip_rate_by_model_perturbation_type(
            metrics, figures_dir / "flip_rate_by_model_perturbation_type.png"
        ),
    ]
    paths.extend(plot_accuracy_scatter(metrics, figures_dir))
    return paths
