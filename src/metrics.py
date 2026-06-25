from __future__ import annotations

import json
import math
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

    df = pd.DataFrame(rows)
    dedupe_keys = ["model_id", "condition_id", "question_id"]
    return df.drop_duplicates(subset=dedupe_keys, keep="last")


def _resolve_model_ids(df: pd.DataFrame, model_ids: list[str] | None) -> list[str]:
    if model_ids:
        return sorted(model_ids)
    if df.empty or "model_id" not in df.columns:
        return []
    return sorted(df["model_id"].unique())


def _matched_question_ids(
    df: pd.DataFrame,
    model_ids: list[str],
    condition_id: str,
    original_condition: str,
) -> set[str]:
    """Questions where every model has a parseable answer on original and condition."""

    def complete_for_condition(cond: str) -> set[str]:
        sub = df[
            (df["condition_id"] == cond)
            & (df["model_id"].isin(model_ids))
            & df["parsed_answer"].notna()
        ]
        counts = sub.groupby("question_id")["model_id"].nunique()
        return set(counts[counts == len(model_ids)].index)

    return complete_for_condition(original_condition) & complete_for_condition(
        condition_id
    )


def _filter_matched(
    df: pd.DataFrame,
    model_ids: list[str],
    condition_id: str,
    original_condition: str,
) -> pd.DataFrame:
    question_ids = _matched_question_ids(
        df, model_ids, condition_id, original_condition
    )
    if not question_ids:
        return df.iloc[0:0].copy()

    conditions = {original_condition, condition_id}
    return df[
        df["question_id"].isin(question_ids)
        & df["condition_id"].isin(conditions)
        & df["model_id"].isin(model_ids)
        & df["parsed_answer"].notna()
    ].copy()


def _accuracy_table(
    matched_df: pd.DataFrame,
    matched_n: int,
) -> pd.DataFrame:
    columns = [
        "model_id",
        "condition_id",
        "subject",
        "perturbation_type",
        "accuracy",
        "n",
        "matched_n",
    ]
    if matched_df.empty or "is_correct" not in matched_df.columns:
        return pd.DataFrame(columns=columns)

    grouped = (
        matched_df.groupby(
            ["model_id", "condition_id", "subject", "perturbation_type"],
            as_index=False,
        )
        .agg(accuracy=("is_correct", "mean"), n=("is_correct", "count"))
    )
    grouped["matched_n"] = matched_n
    return grouped


def _safe_spearmanr(x: Any, y: Any, min_n: int = 3) -> float | None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < min_n or y_arr.size < min_n:
        return None
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return None
    try:
        corr, _ = spearmanr(x_arr, y_arr)
    except (ValueError, AttributeError, TypeError):
        return None
    return float(corr) if not np.isnan(corr) else None


def _spearman_rank_correlation(
    accuracy_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    model_ids: list[str],
    original_condition: str = "original",
    min_questions_per_subject: int = 5,
) -> dict[str, Any]:
    if accuracy_df.empty:
        return {"per_condition": {}, "macro_mean": {}}

    conditions = sorted(
        c for c in accuracy_df["condition_id"].unique() if c != original_condition
    )
    per_condition: dict[str, Any] = {}
    macro_values: list[float] = []

    for condition_id in conditions:
        if "paired_condition" in macro_df.columns:
            orig_macro = macro_df[
                (macro_df["condition_id"] == original_condition)
                & (macro_df["paired_condition"] == condition_id)
                & (macro_df["model_id"].isin(model_ids))
            ].set_index("model_id")["macro_accuracy"]
            pert_macro = macro_df[
                (macro_df["condition_id"] == condition_id)
                & (macro_df["paired_condition"] == condition_id)
                & (macro_df["model_id"].isin(model_ids))
            ].set_index("model_id")["macro_accuracy"]
        else:
            orig_macro = macro_df[
                (macro_df["condition_id"] == original_condition)
                & (macro_df["model_id"].isin(model_ids))
            ].set_index("model_id")["macro_accuracy"]
            pert_macro = macro_df[
                (macro_df["condition_id"] == condition_id)
                & (macro_df["model_id"].isin(model_ids))
            ].set_index("model_id")["macro_accuracy"]

        merged_macro = pd.DataFrame({"orig": orig_macro, "pert": pert_macro}).dropna()
        condition_spearman = _safe_spearmanr(
            merged_macro["orig"],
            merged_macro["pert"],
            min_n=len(model_ids),
        )

        per_subject: dict[str, float | None] = {}
        subjects = sorted(accuracy_df["subject"].unique())
        for subject in subjects:
            orig = accuracy_df[
                (accuracy_df["condition_id"] == original_condition)
                & (accuracy_df["subject"] == subject)
            ]
            pert = accuracy_df[
                (accuracy_df["condition_id"] == condition_id)
                & (accuracy_df["subject"] == subject)
            ]
            if (
                orig.empty
                or pert.empty
                or int(orig["n"].iloc[0]) < min_questions_per_subject
            ):
                per_subject[subject] = None
                continue

            merged = orig.merge(
                pert,
                on="model_id",
                suffixes=("_orig", "_pert"),
            )
            per_subject[subject] = _safe_spearmanr(
                merged["accuracy_orig"],
                merged["accuracy_pert"],
                min_n=len(model_ids),
            )

        if condition_spearman is not None:
            macro_values.append(condition_spearman)

        per_condition[condition_id] = {
            "condition_level_spearman": condition_spearman,
            "per_subject": per_subject,
            "mean_spearman": condition_spearman,
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
    matched_df: pd.DataFrame,
    original_condition: str = "original",
) -> dict[str, Any]:
    empty = {
        "per_record": [],
        "by_subject": {},
        "by_condition": {},
        "by_perturbation_type": {},
        "by_model_perturbation_type": {},
    }
    if matched_df.empty or "parsed_answer" not in matched_df.columns:
        return empty

    original = matched_df[matched_df["condition_id"] == original_condition][
        ["model_id", "question_id", "subject", "parsed_answer"]
    ].rename(columns={"parsed_answer": "original_answer"})

    perturbed = matched_df[matched_df["condition_id"] != original_condition].copy()
    merged = perturbed.merge(
        original,
        on=["model_id", "question_id", "subject"],
        how="inner",
    )

    valid = merged.copy()
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

    by_model_perturbation_type: dict[str, dict[str, float]] = {}
    grouped = (
        valid.groupby(["model_id", "perturbation_type"], as_index=False)
        .agg(flip_rate=("flipped", "mean"))
    )
    for _, row in grouped.iterrows():
        by_model_perturbation_type.setdefault(
            str(row["model_id"]), {}
        )[str(row["perturbation_type"])] = float(row["flip_rate"])

    return {
        "per_record": per_record,
        "by_subject": by_subject,
        "by_condition": by_condition,
        "by_perturbation_type": by_perturbation_type,
        "by_model_perturbation_type": by_model_perturbation_type,
    }


def _heatmap_macro_table(macro_df: pd.DataFrame, original_condition: str) -> pd.DataFrame:
    """One macro-accuracy row per model/condition for plotting."""
    if macro_df.empty:
        return macro_df

    if "paired_condition" not in macro_df.columns:
        return macro_df.drop_duplicates(subset=["model_id", "condition_id"])

    baseline = macro_df[
        (macro_df["condition_id"] == original_condition)
        & macro_df["paired_condition"].isna()
    ]
    perturbed = macro_df[
        (macro_df["condition_id"] != original_condition)
        & (macro_df["condition_id"] == macro_df["paired_condition"])
    ]
    return pd.concat([baseline, perturbed], ignore_index=True).drop_duplicates(
        subset=["model_id", "condition_id"],
        keep="last",
    )


def _accuracy_deltas(
    macro_df: pd.DataFrame,
    condition_to_type: dict[str, str],
    original_condition: str = "original",
) -> pd.DataFrame:
    columns = [
        "model_id",
        "condition_id",
        "perturbation_type",
        "accuracy_delta",
        "original_macro_accuracy",
        "perturbed_macro_accuracy",
        "n",
        "matched_n",
    ]
    if macro_df.empty:
        return pd.DataFrame(columns=columns)

    condition_to_type = condition_to_type or {}
    rows: list[dict[str, Any]] = []

    if "paired_condition" in macro_df.columns:
        orig = macro_df[
            (macro_df["condition_id"] == original_condition)
            & macro_df["paired_condition"].notna()
        ]
        pert = macro_df[
            (macro_df["condition_id"] != original_condition)
            & (macro_df["condition_id"] == macro_df["paired_condition"])
        ]
        merged = orig.merge(
            pert,
            on=["model_id", "paired_condition"],
            suffixes=("_orig", "_pert"),
        )
        for _, row in merged.iterrows():
            condition_id = str(row["paired_condition"])
            rows.append(
                {
                    "model_id": row["model_id"],
                    "condition_id": condition_id,
                    "perturbation_type": condition_to_type.get(condition_id, "unknown"),
                    "original_macro_accuracy": float(row["macro_accuracy_orig"]),
                    "perturbed_macro_accuracy": float(row["macro_accuracy_pert"]),
                    "accuracy_delta": float(
                        row["macro_accuracy_orig"] - row["macro_accuracy_pert"]
                    ),
                    "n": int(row["n_pert"]),
                    "matched_n": int(row["matched_n_pert"]),
                }
            )
        return pd.DataFrame(rows)

    orig = macro_df[macro_df["condition_id"] == original_condition].set_index("model_id")
    perturbed_conditions = sorted(
        c for c in macro_df["condition_id"].unique() if c != original_condition
    )

    for condition_id in perturbed_conditions:
        pert = macro_df[macro_df["condition_id"] == condition_id].set_index("model_id")
        for model_id in sorted(set(orig.index) & set(pert.index)):
            rows.append(
                {
                    "model_id": model_id,
                    "condition_id": condition_id,
                    "perturbation_type": condition_to_type.get(condition_id, "unknown"),
                    "original_macro_accuracy": float(orig.loc[model_id, "macro_accuracy"]),
                    "perturbed_macro_accuracy": float(pert.loc[model_id, "macro_accuracy"]),
                    "accuracy_delta": float(
                        orig.loc[model_id, "macro_accuracy"]
                        - pert.loc[model_id, "macro_accuracy"]
                    ),
                    "n": int(pert.loc[model_id, "n"]),
                    "matched_n": int(pert.loc[model_id, "matched_n"]),
                }
            )

    return pd.DataFrame(rows)


def _build_per_condition_metrics(
    df: pd.DataFrame,
    model_ids: list[str],
    original_condition: str,
    min_questions_per_subject: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, Any]]:
    """Build accuracy tables from per-condition matched intersections."""
    all_conditions = sorted(df["condition_id"].unique())
    accuracy_parts: list[pd.DataFrame] = []
    macro_rows: list[dict[str, Any]] = []
    flip_parts: list[pd.DataFrame] = []
    matched_counts: dict[str, int] = {}

    original_question_ids = _matched_question_ids(
        df, model_ids, original_condition, original_condition
    )
    original_matched = df[
        df["question_id"].isin(original_question_ids)
        & (df["condition_id"] == original_condition)
        & df["model_id"].isin(model_ids)
        & df["parsed_answer"].notna()
    ].copy()
    matched_counts[original_condition] = int(original_matched["question_id"].nunique())
    if not original_matched.empty:
        accuracy_parts.append(
            _accuracy_table(original_matched, matched_counts[original_condition])
        )
        for (model_id, condition_id), sub in original_matched.groupby(
            ["model_id", "condition_id"]
        ):
            macro_rows.append(
                {
                    "model_id": model_id,
                    "condition_id": condition_id,
                    "macro_accuracy": float(sub["is_correct"].mean()),
                    "n": int(len(sub)),
                    "matched_n": matched_counts[original_condition],
                    "paired_condition": None,
                }
            )

    for condition_id in all_conditions:
        if condition_id == original_condition:
            continue

        matched = _filter_matched(
            df, model_ids, condition_id, original_condition
        )
        matched_counts[condition_id] = int(
            matched[matched["condition_id"] == condition_id]["question_id"].nunique()
        )
        if matched.empty:
            continue

        matched_n = matched_counts[condition_id]
        accuracy_parts.append(_accuracy_table(matched, matched_n))
        for (model_id, cond), sub in matched.groupby(["model_id", "condition_id"]):
            macro_rows.append(
                {
                    "model_id": model_id,
                    "condition_id": cond,
                    "macro_accuracy": float(sub["is_correct"].mean()),
                    "n": int(len(sub)),
                    "matched_n": matched_n,
                    "paired_condition": condition_id,
                }
            )

        flip = _flip_rates(matched, original_condition)
        if flip["per_record"]:
            flip_parts.append(pd.DataFrame(flip["per_record"]))

    accuracy_df = (
        pd.concat(accuracy_parts, ignore_index=True)
        if accuracy_parts
        else pd.DataFrame()
    )
    macro_df = pd.DataFrame(macro_rows) if macro_rows else pd.DataFrame()

    if flip_parts:
        flip_df = pd.concat(flip_parts, ignore_index=True)
        flip_rates = _flip_rates_from_records(flip_df)
    else:
        flip_rates = _flip_rates(pd.DataFrame(), original_condition)

    return accuracy_df, macro_df, matched_counts, flip_rates


def _flip_rates_from_records(flip_df: pd.DataFrame) -> dict[str, Any]:
    by_subject = (
        flip_df.groupby("subject", as_index=False)
        .agg(flip_rate=("flipped", "mean"), n=("flipped", "count"))
        .set_index("subject")["flip_rate"]
        .to_dict()
    )
    by_condition = (
        flip_df.groupby("condition_id", as_index=False)
        .agg(flip_rate=("flipped", "mean"), n=("flipped", "count"))
        .set_index("condition_id")["flip_rate"]
        .to_dict()
    )
    by_perturbation_type = (
        flip_df.groupby("perturbation_type", as_index=False)
        .agg(flip_rate=("flipped", "mean"), n=("flipped", "count"))
        .set_index("perturbation_type")["flip_rate"]
        .to_dict()
    )

    # Per-model breakdown: {model_id: {perturbation_type: flip_rate}}
    by_model_perturbation_type: dict[str, dict[str, float]] = {}
    if not flip_df.empty:
        grouped = (
            flip_df.groupby(["model_id", "perturbation_type"], as_index=False)
            .agg(flip_rate=("flipped", "mean"))
        )
        for _, row in grouped.iterrows():
            by_model_perturbation_type.setdefault(
                str(row["model_id"]), {}
            )[str(row["perturbation_type"])] = float(row["flip_rate"])

    return {
        "per_record": flip_df.to_dict(orient="records"),
        "by_subject": by_subject,
        "by_condition": by_condition,
        "by_perturbation_type": by_perturbation_type,
        "by_model_perturbation_type": by_model_perturbation_type,
    }


def _spearman_by_perturbation_type(
    accuracy_df: pd.DataFrame,
    condition_to_type: dict[str, str],
    model_ids: list[str],
    original_condition: str = "original",
    min_questions_per_subject: int = 5,
) -> dict[str, dict[str, Any]]:
    """Spearman rank correlation per perturbation type, pooling (model, subject) pairs.

    With only 3 models, condition-level Spearman has 3 data points. This function
    pools across subjects within each perturbation type, giving ~N_models * N_subjects
    data points for a more robust estimate. Subjects with fewer than
    min_questions_per_subject questions are excluded.
    """
    if accuracy_df.empty:
        return {}

    orig_data = accuracy_df[
        (accuracy_df["condition_id"] == original_condition)
        & (accuracy_df["model_id"].isin(model_ids))
        & (accuracy_df["n"] >= min_questions_per_subject)
    ][["model_id", "subject", "accuracy"]].rename(columns={"accuracy": "accuracy_orig"})

    perturbed_conditions = [
        c for c in accuracy_df["condition_id"].unique() if c != original_condition
    ]

    # Group conditions by perturbation type
    type_to_conditions: dict[str, list[str]] = {}
    for cond in perturbed_conditions:
        ptype = condition_to_type.get(cond, "unknown")
        type_to_conditions.setdefault(ptype, []).append(cond)

    results: dict[str, dict[str, Any]] = {}
    for ptype, conds in sorted(type_to_conditions.items()):
        x_vals: list[float] = []
        y_vals: list[float] = []

        for cond in conds:
            pert_data = accuracy_df[
                (accuracy_df["condition_id"] == cond)
                & (accuracy_df["model_id"].isin(model_ids))
                & (accuracy_df["n"] >= min_questions_per_subject)
            ][["model_id", "subject", "accuracy"]].rename(
                columns={"accuracy": "accuracy_pert"}
            )

            merged = orig_data.merge(pert_data, on=["model_id", "subject"])
            if not merged.empty:
                x_vals.extend(merged["accuracy_orig"].tolist())
                y_vals.extend(merged["accuracy_pert"].tolist())

        spearman = _safe_spearmanr(x_vals, y_vals, min_n=max(len(model_ids), 3))
        results[ptype] = {
            "spearman": spearman,
            "n_pairs": len(x_vals),
            "conditions": sorted(conds),
        }

    return results


def _build_summary_csv(
    accuracy_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    flip_rates: dict[str, Any],
    deltas_df: pd.DataFrame,
    spearman: dict[str, Any],
) -> pd.DataFrame:
    summary = accuracy_df.copy()
    if not summary.empty:
        summary = summary.rename(columns={"accuracy": "subject_accuracy"})

    if not macro_df.empty:
        summary = summary.merge(
            macro_df[["model_id", "condition_id", "macro_accuracy", "matched_n"]],
            on=["model_id", "condition_id"],
            how="left",
        )

    if not deltas_df.empty:
        summary = summary.merge(
            deltas_df[
                [
                    "model_id",
                    "condition_id",
                    "accuracy_delta",
                    "original_macro_accuracy",
                    "perturbed_macro_accuracy",
                ]
            ],
            on=["model_id", "condition_id"],
            how="left",
        )

    flip_by_condition = flip_rates.get("by_condition", {})
    if not summary.empty:
        summary["condition_flip_rate"] = summary["condition_id"].map(flip_by_condition)

    spearman_per_condition = spearman.get("per_condition", {})
    if not summary.empty:
        summary["condition_mean_spearman"] = summary["condition_id"].map(
            {
                cond: values.get("mean_spearman")
                for cond, values in spearman_per_condition.items()
            }
        )

    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        item = value.item()
        if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
            return None
        return item
    if value is pd.NA:
        return None
    return value


def compute_metrics(
    raw_dir: str | Path,
    seed: int,
    model_ids: list[str] | None = None,
    original_condition: str = "original",
    min_questions_per_subject: int = 5,
) -> dict[str, Any]:
    df = _load_raw_dataframe(raw_dir)
    resolved_model_ids = _resolve_model_ids(df, model_ids)

    accuracy_df, macro_df, matched_counts, flip_rates = _build_per_condition_metrics(
        df,
        resolved_model_ids,
        original_condition,
        min_questions_per_subject,
    )

    if not macro_df.empty and "perturbation_type" not in macro_df.columns:
        condition_to_type = (
            df.drop_duplicates("condition_id")
            .set_index("condition_id")["perturbation_type"]
            .to_dict()
        )
        macro_df = macro_df.copy()
        macro_df["perturbation_type"] = macro_df["condition_id"].map(condition_to_type)
    else:
        condition_to_type = (
            df.drop_duplicates("condition_id")
            .set_index("condition_id")["perturbation_type"]
            .to_dict()
        )

    spearman = _spearman_rank_correlation(
        accuracy_df,
        macro_df,
        resolved_model_ids,
        original_condition,
        min_questions_per_subject,
    )
    spearman_by_type = _spearman_by_perturbation_type(
        accuracy_df,
        condition_to_type,
        resolved_model_ids,
        original_condition,
        min_questions_per_subject,
    )
    deltas_df = _accuracy_deltas(macro_df, condition_to_type, original_condition)
    heatmap_macro_df = _heatmap_macro_table(macro_df, original_condition)
    summary_df = _build_summary_csv(
        accuracy_df, heatmap_macro_df, flip_rates, deltas_df, spearman
    )

    return {
        "seed": seed,
        "model_ids": resolved_model_ids,
        "matched_question_counts": matched_counts,
        "min_questions_per_subject": min_questions_per_subject,
        "accuracy_by_model_condition_subject": accuracy_df.to_dict(orient="records"),
        "macro_accuracy_by_model_condition": heatmap_macro_df.to_dict(
            orient="records"
        ),
        "spearman_rank_correlation": spearman,
        "spearman_by_perturbation_type": spearman_by_type,
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
        json.dump(_json_safe(metrics), f, indent=2, ensure_ascii=False)

    return summary_path, metrics_path
