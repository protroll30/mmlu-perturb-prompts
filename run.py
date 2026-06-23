from __future__ import annotations

import argparse
import logging
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.conditions import normalize_condition, parse_conditions_arg
from src.evaluator import run_evaluation
from src.io_utils import atomic_write_json, load_config
from src.loader import load_or_sample
from src.metrics import compute_metrics, export_metrics
from src.paraphrase_cache import (
    collect_paraphrase_targets,
    conditions_need_paraphrase,
    warm_paraphrase_cache,
)
from src.perturbations import generate_perturbed_sets
from src.viz import generate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_models_arg(value: str | None, config) -> list:
    if value is None or value.strip().lower() == "all":
        return config.models
    requested = {token.strip() for token in value.split(",") if token.strip()}
    selected = [m for m in config.models if m.id in requested]
    missing = requested - {m.id for m in selected}
    if missing:
        raise ValueError(f"Unknown model ids: {sorted(missing)}")
    return selected


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _write_run_meta(
    metrics_dir: Path,
    seed: int,
    sample_size: int,
    model_ids: list[str],
    condition_ids: list[str],
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "models": model_ids,
        "conditions": condition_ids,
        "git_hash": _git_hash(),
    }
    atomic_write_json(metrics_dir / "run_meta.json", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MMLU benchmark sensitivity evaluation framework",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--sample-size", type=int, default=None, help="Override sample size")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model ids or 'all'",
    )
    parser.add_argument(
        "--conditions",
        default="all",
        help="Comma-separated condition tokens or 'all'",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Re-sample MMLU questions even if sampled.jsonl exists",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip model evaluation")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics export")
    parser.add_argument("--skip-viz", action="store_true", help="Skip figure generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = config.seed
    sample_size = args.sample_size if args.sample_size is not None else config.sample_size
    _set_seeds(seed)

    conditions = [normalize_condition(c) for c in parse_conditions_arg(args.conditions)]
    models = _parse_models_arg(args.models, config)

    condition_ids = [c.condition_id for c in conditions]
    model_ids = [m.id for m in models]

    metrics_dir = Path(config.paths.metrics_dir)
    _write_run_meta(metrics_dir, seed, sample_size, model_ids, condition_ids)

    # 1. Load once
    logger.info("Loading or sampling %s MMLU questions (seed=%s)", sample_size, seed)
    sampled = load_or_sample(
        config.paths.sampled,
        sample_size=sample_size,
        seed=seed,
        force_resample=args.force_resample,
    )
    logger.info("Using %s sampled questions", len(sampled))

    # 2. Warm paraphrase cache upfront (before perturbation pipelines or eval)
    if conditions_need_paraphrase(conditions):
        targets = collect_paraphrase_targets(sampled, conditions, seed)
        logger.info("Warming paraphrase cache for %s unique texts", len(targets))
        warm_paraphrase_cache(targets, config)
    else:
        logger.info("No semantic_paraphrase conditions; skipping paraphrase cache warm")

    # 3. Fan out to independent perturbation pipelines (one JSONL each)
    logger.info("Running %s perturbation pipelines", len(conditions))
    perturbed_paths = generate_perturbed_sets(sampled, conditions, config)

    # 4. Single evaluation loop with question-level checkpoint
    if not args.skip_eval:
        logger.info("Starting unified evaluation loop")
        run_evaluation(
            models=models,
            perturbed_paths=perturbed_paths,
            raw_results_dir=Path(config.paths.raw_results_dir),
            seed=seed,
        )
    else:
        logger.info("Skipping evaluation")

    # 5. Metrics over full results JSONL
    if not args.skip_metrics:
        logger.info("Computing metrics")
        metrics = compute_metrics(config.paths.raw_results_dir, seed=seed)
        summary_path, metrics_path = export_metrics(metrics, config.paths.metrics_dir)
        logger.info("Wrote metrics to %s and %s", summary_path, metrics_path)
    else:
        logger.info("Skipping metrics")
        metrics = compute_metrics(config.paths.raw_results_dir, seed=seed)

    # 6. Visualization as final pass over metrics
    if not args.skip_viz:
        logger.info("Generating figures")
        figure_paths = generate_all_figures(metrics, config.paths.figures_dir)
        for path in figure_paths:
            logger.info("Wrote figure %s", path)
    else:
        logger.info("Skipping visualization")

    return 0


if __name__ == "__main__":
    sys.exit(main())
