from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.io_utils import iter_raw_result_files, read_jsonl


def _load_raw_dataframe(raw_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in iter_raw_result_files(raw_dir):
        rows.extend(read_jsonl(path))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_correct" not in df.columns:
        return pd.DataFrame(
            columns=[
                "model_id",
                "condition_id",
                "subject",
                "perturbation_type",
                "accuracy",
                "n",
            ]
        )

    valid = df.dropna(subset=["is_correct"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "condition_id",
                "subject",
                "perturbation_type",
                "accuracy",
                "n",
            ]
        )

    grouped = (
        valid.groupby(
            ["model_id", "condition_id", "subject", "perturbation_type"],
            as_index=False,
        )
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "count"))
    )
    return grouped


def _macro_accuracy_table(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    if accuracy_df.empty:
        return pd.DataFrame(columns=["model_id", "condition_id", "macro_accuracy"])
    grouped = (
        accuracy_df.groupby(["model_id", "condition_id"], as_index=False)
        .agg(macro_accuracy=("accuracy", "mean"))
    )
    return grouped


def _spearman_rank_correlation(
    accuracy_df: pd.DataFrame,
    original_condition: str = "original",
) -> dict[str, Any]:
    if accuracy_df.empty:
        return {"per_condition": {}, "macro_mean": {}}

    conditions = sorted(
        c for c in accuracy_df["condition_id"].unique() if c != original_condition
    )
    subjects = sorted(accuracy_df["subject"].unique())
    per_condition: dict[str, Any] = {}
    macro_values: list[float] = []

    for condition_id in conditions:
        per_subject: dict[str, float | None] = {}
        for subject in subjects:
            orig = accuracy_df[
                (accuracy_df["condition_id"] == original_condition)
                & (accuracy_df["subject"] == subject)
            ][["model_id", "accuracy"]]
            pert = accuracy_df[
                (accuracy_df["condition_id"] == condition_id)
                & (accuracy_df["subject"] == subject)
            ][["model_id", "accuracy"]]

            merged = orig.merge(pert, on="model_id", suffixes=("_orig", "_pert"))
            if len(merged) < 2:
                per_subject[subject] = None
                continue

            corr, _ = spearmanr(merged["accuracy_orig"], merged["accuracy_pert"])
            value = float(corr) if not np.isnan(corr) else None
            per_subject[subject] = value
            if value is not None:
                macro_values.append(value)

        valid_subject_values = [v for v in per_subject.values() if v is not None]
        per_condition[condition_id] = {
            "per_subject": per_subject,
            "mean_spearman": (
                float(np.mean(valid_subject_values)) if valid_subject_values else None
            ),
        }

    return {
        "per_condition": per_condition,
        "macro_mean": {
            "across_conditions": (
                float(np.mean(macro_values)) if macro_values else None
            )
        },
    }


def _flip_rates(
    df: pd.DataFrame,
    original_condition: str = "original",
) -> dict[str, Any]:
    if df.empty or "parsed_answer" not in df.columns:
        return {
            "per_record": [],
            "by_subject": {},
            "by_condition": {},
            "by_perturbation_type": {},
        }

    original = df[df["condition_id"] == original_condition][
        ["model_id", "question_id", "subject", "parsed_answer"]
    ].rename(columns={"parsed_answer": "original_answer"})

    perturbed = df[df["condition_id"] != original_condition].copy()
    merged = perturbed.merge(
        original,
        on=["model_id", "question_id", "subject"],
        how="inner",
    )

    valid = merged.dropna(subset=["original_answer", "parsed_answer"])
    valid = valid.copy()
    valid["flipped"] = valid["parsed_answer"] != valid["original_answer"]

    per_record = valid[
        [
            "model_id",
            "question_id",
            "subject",
            "condition_id",
            "perturbation_type",
            "flipped",
        ]
    ].to_dict(orient="records")

    by_subject = (
        valid.groupby("subject", as_index=False)
        .agg(flip_rate=("flipped", "mean"), n=("flipped", "count"))
        .set_index("subject")["flip_rate"]
        .to_dict()
    )

    by_condition = (
        valid.groupby("condition_id", as_index=False)
        .agg(flip_rate=("flipped", "mean"), n=("flipped", "count"))
        .set_index("condition_id")["flip_rate"]
        .to_dict()
    )

    by_perturbation_type = (
        valid.groupby("perturbation_type", as_index=False)
        .agg(flip_rate=("flipped", "mean"), n=("flipped", "count"))
        .set_index("perturbation_type")["flip_rate"]
        .to_dict()
    )

    return {
        "per_record": per_record,
        "by_subject": by_subject,
        "by_condition": by_condition,
        "by_perturbation_type": by_perturbation_type,
    }


def _accuracy_deltas(
    accuracy_df: pd.DataFrame,
    df: pd.DataFrame,
    original_condition: str = "original",
) -> pd.DataFrame:
    if accuracy_df.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "perturbation_type",
                "condition_id",
                "accuracy_delta",
            ]
        )

    orig_macro = (
        accuracy_df[accuracy_df["condition_id"] == original_condition]
        .groupby("model_id", as_index=False)
        .agg(original_macro_accuracy=("accuracy", "mean"))
    )

    perturbed_conditions = sorted(
        c for c in accuracy_df["condition_id"].unique() if c != original_condition
    )
    condition_to_type = (
        df.drop_duplicates("condition_id")
        .set_index("condition_id")["perturbation_type"]
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for condition_id in perturbed_conditions:
        pert_macro = (
            accuracy_df[accuracy_df["condition_id"] == condition_id]
            .groupby("model_id", as_index=False)
            .agg(perturbed_macro_accuracy=("accuracy", "mean"))
        )
        merged = orig_macro.merge(pert_macro, on="model_id", how="inner")
        for _, row in merged.iterrows():
            rows.append(
                {
                    "model_id": row["model_id"],
                    "perturbation_type": condition_to_type.get(
                        condition_id, "unknown"
                    ),
                    "condition_id": condition_id,
                    "accuracy_delta": float(
                        row["original_macro_accuracy"] - row["perturbed_macro_accuracy"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def _build_summary_csv(
    accuracy_df: pd.DataFrame,
    flip_rates: dict[str, Any],
    deltas_df: pd.DataFrame,
    spearman: dict[str, Any],
) -> pd.DataFrame:
    summary = accuracy_df.copy()
    if not summary.empty:
        summary = summary.rename(columns={"accuracy": "subject_accuracy"})

    if not deltas_df.empty:
        summary = summary.merge(
            deltas_df,
            on=["model_id", "condition_id"],
            how="left",
        )

    flip_by_condition = flip_rates.get("by_condition", {})
    if not summary.empty:
        summary["condition_flip_rate"] = summary["condition_id"].map(
            flip_by_condition
        )

    spearman_per_condition = spearman.get("per_condition", {})
    if not summary.empty:
        summary["condition_mean_spearman"] = summary["condition_id"].map(
            {
                cond: values.get("mean_spearman")
                for cond, values in spearman_per_condition.items()
            }
        )

    return summary


def compute_metrics(
    raw_dir: str | Path,
    seed: int,
    original_condition: str = "original",
) -> dict[str, Any]:
    df = _load_raw_dataframe(raw_dir)
    accuracy_df = _accuracy_table(df)
    macro_accuracy_df = _macro_accuracy_table(accuracy_df)
    spearman = _spearman_rank_correlation(accuracy_df, original_condition)
    flip_rates = _flip_rates(df, original_condition)
    deltas_df = _accuracy_deltas(accuracy_df, df, original_condition)
    summary_df = _build_summary_csv(accuracy_df, flip_rates, deltas_df, spearman)

    return {
        "seed": seed,
        "accuracy_by_model_condition_subject": accuracy_df.to_dict(orient="records"),
        "macro_accuracy_by_model_condition": macro_accuracy_df.to_dict(orient="records"),
        "spearman_rank_correlation": spearman,
        "flip_rates": flip_rates,
        "accuracy_delta": deltas_df.to_dict(orient="records"),
        "summary_table": summary_df.to_dict(orient="records"),
    }


def export_metrics(
    metrics: dict[str, Any],
    metrics_dir: str | Path,
) -> tuple[Path, Path]:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metrics_dir / "summary.csv"
    metrics_path = metrics_dir / "metrics.json"

    summary_df = pd.DataFrame(metrics.get("summary_table", []))
    summary_df.to_csv(summary_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return summary_path, metrics_path
