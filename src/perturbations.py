from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from src.conditions import build_condition_id, normalize_condition
from src.io_utils import (
    atomic_write_json,
    condition_to_filename,
    dataclass_to_dict,
    get_env_api_key,
    read_json,
    write_jsonl,
)
from src.types import (
    AppConfig,
    Condition,
    ParaphraseConfig,
    PerturbedQuestion,
    PerturbationStep,
    Question,
)

logger = logging.getLogger(__name__)

Transform = Callable[[Question, dict[str, Any], random.Random], Question]

LABEL_STYLES = {
    "alpha_upper": ["A", "B", "C", "D"],
    "numeric": ["1", "2", "3", "4"],
    "parenthetical": ["(a)", "(b)", "(c)", "(d)"],
}


def _question_rng(seed: int, original_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{original_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _label_style_from_steps(steps: tuple[PerturbationStep, ...]) -> str:
    for step in steps:
        if step.name == "label_remap":
            style = step.params.get("style", "numeric")
            if style == "parenthetical":
                return "parenthetical"
            return "numeric"
    return "alpha_upper"


def _apply_label_remap(
    question: Question,
    params: dict[str, Any],
    rng: random.Random,
) -> Question:
    _ = rng
    style = params.get("style", "numeric")
    if style not in {"numeric", "parenthetical"}:
        raise ValueError(f"Unsupported label_remap style: {style}")
    return question


def _apply_option_shuffle(
    question: Question,
    params: dict[str, Any],
    rng: random.Random,
) -> Question:
    _ = params
    indices = list(range(len(question.options)))
    rng.shuffle(indices)
    shuffled_options = [question.options[i] for i in indices]
    new_correct = indices.index(question.correct_answer_index)
    return Question(
        original_id=question.original_id,
        subject=question.subject,
        question_text=question.question_text,
        options=shuffled_options,
        correct_answer_index=new_correct,
    )


def _apply_context_inject(
    question: Question,
    params: dict[str, Any],
    rng: random.Random,
) -> Question:
    _ = rng
    text = params.get("text", "The following is a multiple choice question.")
    return Question(
        original_id=question.original_id,
        subject=question.subject,
        question_text=f"{text}\n\n{question.question_text}",
        options=list(question.options),
        correct_answer_index=question.correct_answer_index,
    )


def _apply_instruction_style(
    question: Question,
    params: dict[str, Any],
    rng: random.Random,
) -> Question:
    _ = params, rng
    return question


def _cache_path(cache_dir: Path, question_text: str) -> Path:
    key = hashlib.sha256(question_text.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _paraphrase_question(
    question_text: str,
    paraphrase_config: ParaphraseConfig,
    cache_dir: Path,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, question_text)
    if path.exists():
        cached = read_json(path)
        return str(cached["paraphrased_text"])

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
            payload = {
                "paraphrased_text": paraphrased,
                "model": paraphrase_config.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(path, payload)
            return paraphrased
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = 2**attempt
            logger.warning(
                "Paraphrase attempt %s failed for cache key %s: %s",
                attempt + 1,
                path.name,
                exc,
            )
            if attempt < 2:
                import time

                time.sleep(wait)

    raise RuntimeError(f"Paraphrase failed after retries: {last_error}")


def _apply_semantic_paraphrase(
    question: Question,
    params: dict[str, Any],
    rng: random.Random,
    paraphrase_config: ParaphraseConfig,
    cache_dir: Path,
) -> Question:
    _ = params, rng
    paraphrased = _paraphrase_question(
        question.question_text,
        paraphrase_config,
        cache_dir,
    )
    return Question(
        original_id=question.original_id,
        subject=question.subject,
        question_text=paraphrased,
        options=list(question.options),
        correct_answer_index=question.correct_answer_index,
    )


TRANSFORMS: dict[str, Transform] = {
    "label_remap": _apply_label_remap,
    "option_shuffle": _apply_option_shuffle,
    "context_inject": _apply_context_inject,
    "instruction_style": _apply_instruction_style,
}


def apply_stack(
    question: Question,
    steps: tuple[PerturbationStep, ...],
    seed: int,
    paraphrase_config: ParaphraseConfig | None = None,
    cache_dir: Path | None = None,
) -> Question:
    current = question
    for step in steps:
        if step.name == "semantic_paraphrase":
            if paraphrase_config is None or cache_dir is None:
                raise ValueError("semantic_paraphrase requires paraphrase config")
            rng = _question_rng(seed, current.original_id)
            current = _apply_semantic_paraphrase(
                current,
                step.params,
                rng,
                paraphrase_config,
                cache_dir,
            )
            continue

        transform = TRANSFORMS.get(step.name)
        if transform is None:
            raise ValueError(f"Unknown perturbation step: {step.name}")

        rng = _question_rng(seed, f"{current.original_id}:{step.name}")
        current = transform(current, step.params, rng)
    return current


def _steps_to_params(steps: tuple[PerturbationStep, ...]) -> dict[str, Any]:
    return {
        "stack": [
            {"name": step.name, "params": dict(step.params)} for step in steps
        ]
    }


def _perturbation_type(steps: tuple[PerturbationStep, ...]) -> str:
    if not steps:
        return "original"
    if len(steps) == 1:
        return steps[0].name
    return "composed"


def to_perturbed_question(
    question: Question,
    condition: Condition,
    seed: int,
    paraphrase_config: ParaphraseConfig | None = None,
    cache_dir: Path | None = None,
) -> PerturbedQuestion:
    if condition.is_original:
        return PerturbedQuestion(
            original_id=question.original_id,
            subject=question.subject,
            question_text=question.question_text,
            options=list(question.options),
            correct_answer_index=question.correct_answer_index,
            perturbation_type="original",
            perturbation_params={},
            condition_id="original",
            label_style="alpha_upper",
        )

    transformed = apply_stack(
        question,
        condition.steps,
        seed,
        paraphrase_config=paraphrase_config,
        cache_dir=cache_dir,
    )
    return PerturbedQuestion(
        original_id=transformed.original_id,
        subject=transformed.subject,
        question_text=transformed.question_text,
        options=list(transformed.options),
        correct_answer_index=transformed.correct_answer_index,
        perturbation_type=_perturbation_type(condition.steps),
        perturbation_params=_steps_to_params(condition.steps),
        condition_id=condition.condition_id,
        label_style=_label_style_from_steps(condition.steps),
    )


def generate_perturbed_sets(
    sampled: list[Question],
    conditions: list[Condition],
    config: AppConfig,
) -> dict[str, Path]:
    perturbed_dir = Path(config.paths.perturbed_dir)
    perturbed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config.paths.paraphrase_cache_dir)

    output_paths: dict[str, Path] = {}
    for condition in conditions:
        normalized = normalize_condition(condition)
        out_path = perturbed_dir / condition_to_filename(normalized.condition_id)
        records: list[PerturbedQuestion] = []
        for question in sampled:
            records.append(
                to_perturbed_question(
                    question,
                    normalized,
                    config.seed,
                    paraphrase_config=config.paraphrase,
                    cache_dir=cache_dir,
                )
            )
        write_jsonl(out_path, (dataclass_to_dict(r) for r in records))
        output_paths[normalized.condition_id] = out_path
        logger.info("Wrote %s perturbed questions to %s", len(records), out_path)

    return output_paths
