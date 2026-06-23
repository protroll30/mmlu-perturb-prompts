from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.conditions import normalize_condition
from src.io_utils import condition_to_filename, dataclass_to_dict, write_jsonl
from src.paraphrase_cache import get_cached_paraphrase
from src.types import (
    AppConfig,
    Condition,
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


def _apply_semantic_paraphrase(
    question: Question,
    params: dict[str, Any],
    rng: random.Random,
    cache_dir: Path,
) -> Question:
    _ = params, rng
    paraphrased = get_cached_paraphrase(cache_dir, question.question_text)
    if paraphrased is None:
        raise RuntimeError(
            "Paraphrase cache miss for question "
            f"{question.original_id!r}. Run warm_paraphrase_cache before perturbation."
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
    cache_dir: Path | None = None,
) -> Question:
    current = question
    for step in steps:
        if step.name == "semantic_paraphrase":
            if cache_dir is None:
                raise ValueError("semantic_paraphrase requires cache_dir")
            rng = _question_rng(seed, current.original_id)
            current = _apply_semantic_paraphrase(current, step.params, rng, cache_dir)
            continue

        transform = TRANSFORMS.get(step.name)
        if transform is None:
            raise ValueError(f"Unknown perturbation step: {step.name}")

        rng = _question_rng(seed, f"{current.original_id}:{step.name}")
        current = transform(current, step.params, rng)
    return current


def question_text_before_step(
    question: Question,
    steps: tuple[PerturbationStep, ...],
    step_name: str,
    seed: int,
    cache_dir: Path | None = None,
) -> str:
    pre_steps: list[PerturbationStep] = []
    for step in steps:
        if step.name == step_name:
            break
        pre_steps.append(step)
    transformed = apply_stack(question, tuple(pre_steps), seed, cache_dir=cache_dir)
    return transformed.question_text


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

    transformed = apply_stack(question, condition.steps, seed, cache_dir=cache_dir)
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


def generate_perturbed_set(
    sampled: list[Question],
    condition: Condition,
    seed: int,
    perturbed_dir: Path,
    cache_dir: Path,
) -> Path:
    """Run one independent perturbation pipeline and write its JSONL."""
    normalized = normalize_condition(condition)
    out_path = perturbed_dir / condition_to_filename(normalized.condition_id)
    records = [
        to_perturbed_question(question, normalized, seed, cache_dir=cache_dir)
        for question in sampled
    ]
    write_jsonl(out_path, (dataclass_to_dict(r) for r in records))
    logger.info(
        "Pipeline %s: wrote %s questions to %s",
        normalized.condition_id,
        len(records),
        out_path,
    )
    return out_path


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
        out_path = generate_perturbed_set(
            sampled,
            normalized,
            config.seed,
            perturbed_dir,
            cache_dir,
        )
        output_paths[normalized.condition_id] = out_path

    return output_paths
