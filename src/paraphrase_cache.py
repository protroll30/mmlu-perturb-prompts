from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from src.io_utils import atomic_write_json, get_env_api_key, read_json
from src.types import AppConfig, Condition, ParaphraseConfig, Question

logger = logging.getLogger(__name__)


def cache_key(question_text: str) -> str:
    return hashlib.sha256(question_text.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, question_text: str) -> Path:
    return cache_dir / f"{cache_key(question_text)}.json"


def get_cached_paraphrase(cache_dir: Path, question_text: str) -> str | None:
    path = cache_path(cache_dir, question_text)
    if not path.exists():
        return None
    return str(read_json(path)["paraphrased_text"])


def conditions_need_paraphrase(conditions: list[Condition]) -> bool:
    return any(
        any(step.name == "semantic_paraphrase" for step in condition.steps)
        for condition in conditions
    )


def collect_paraphrase_targets(
    sampled: list[Question],
    conditions: list[Condition],
    seed: int,
) -> list[str]:
    """Return sorted unique question texts that semantic_paraphrase will need."""
    from src.perturbations import question_text_before_step

    targets: set[str] = set()
    for condition in conditions:
        if not any(step.name == "semantic_paraphrase" for step in condition.steps):
            continue
        for question in sampled:
            text = question_text_before_step(
                question,
                condition.steps,
                "semantic_paraphrase",
                seed,
            )
            targets.add(text)
    return sorted(targets)


def _paraphrase_via_api(
    question_text: str,
    paraphrase_config: ParaphraseConfig,
) -> str:
    api_key = get_env_api_key(paraphrase_config.api_key_env)
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "Paraphrase the following multiple-choice question stem only. "
        "Preserve the exact meaning. Do not include answer options, letters, "
        "numbers, or any commentary. Return only the paraphrased question text.\n\n"
        f"Question stem:\n{question_text}"
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=paraphrase_config.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            paraphrased = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            if not paraphrased:
                raise ValueError("Empty paraphrase response")
            return paraphrased
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = 2**attempt
            logger.warning(
                "Paraphrase API attempt %s failed: %s",
                attempt + 1,
                exc,
            )
            if attempt < 2:
                time.sleep(wait)

    raise RuntimeError(f"Paraphrase failed after retries: {last_error}")


def warm_paraphrase_cache(
    question_texts: list[str],
    config: AppConfig,
) -> None:
    """Generate and persist all paraphrases before perturbation or evaluation."""
    cache_dir = Path(config.paths.paraphrase_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    to_fetch = [
        text
        for text in question_texts
        if get_cached_paraphrase(cache_dir, text) is None
    ]
    if not to_fetch:
        logger.info("Paraphrase cache warm: all %s entries already cached", len(question_texts))
        return

    logger.info(
        "Paraphrase cache warm: fetching %s / %s texts",
        len(to_fetch),
        len(question_texts),
    )
    for i, text in enumerate(to_fetch, start=1):
        paraphrased = _paraphrase_via_api(text, config.paraphrase)
        payload = {
            "paraphrased_text": paraphrased,
            "model": config.paraphrase.model,
            "source_hash": cache_key(text),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(cache_path(cache_dir, text), payload)
        logger.info("Cached paraphrase %s / %s", i, len(to_fetch))
