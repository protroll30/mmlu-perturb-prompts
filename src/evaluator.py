from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.io_utils import (
    append_jsonl,
    atomic_write_json,
    load_perturbed_questions,
    read_json,
)
from src.perturbations import LABEL_STYLES
from src.types import EvalRecord, ModelConfig, PerturbedQuestion

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "results.jsonl"
CHECKPOINT_FILENAME = "eval_checkpoint.json"

INSTRUCTION_PREFIXES = {
    "none": "",
    "minimal": "",
    "verbose": (
        "You are a helpful assistant. Answer the following question carefully "
        "and select the best option."
    ),
}

ANSWER_SUFFIXES = {
    "none": "",
    "minimal": "Answer:",
    "verbose": "Answer:",
}


@dataclass(frozen=True)
class EvalTask:
    model_id: str
    condition_id: str
    question_id: str
    question: PerturbedQuestion


@dataclass(frozen=True)
class ModelClient:
    model_id: str
    base_url: str
    model: str
    api_key: str

    def complete(self, prompt: str) -> str | None:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 16,
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait = 2**attempt
                logger.warning(
                    "Model %s attempt %s failed: %s",
                    self.model_id,
                    attempt + 1,
                    exc,
                )
                if attempt < 2:
                    time.sleep(wait)

        logger.error("Model %s failed after retries: %s", self.model_id, last_error)
        return None


def _instruction_style_from_question(question: PerturbedQuestion) -> str:
    stack = question.perturbation_params.get("stack", [])
    for step in stack:
        if step.get("name") == "instruction_style":
            return str(step.get("params", {}).get("style", "minimal"))
    return "none"


def build_prompt(question: PerturbedQuestion) -> str:
    style = _instruction_style_from_question(question)
    labels = LABEL_STYLES.get(question.label_style, LABEL_STYLES["alpha_upper"])

    parts: list[str] = []
    prefix = INSTRUCTION_PREFIXES.get(style, "")
    if prefix:
        parts.append(prefix)

    parts.append(question.question_text)
    parts.append("")

    for idx, option in enumerate(question.options):
        parts.append(f"{labels[idx]}. {option}")

    suffix = ANSWER_SUFFIXES.get(style, "")
    if suffix:
        parts.append("")
        parts.append(suffix)

    return "\n".join(parts)


_LEADING_PATTERNS = [
    re.compile(r"^\s*\(?([A-Da-d])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([1-4])\)?"),
    re.compile(r"^\s*\(([a-d])\)", re.IGNORECASE),
]

_FALLBACK_PATTERNS = [
    re.compile(r"\b([A-Da-d])\b"),
    re.compile(r"\b([1-4])\b"),
    re.compile(r"\(([a-d])\)", re.IGNORECASE),
]


def _index_from_token(token: str, label_style: str) -> int | None:
    token = token.strip()
    if label_style == "numeric":
        if token.isdigit():
            value = int(token)
            if 1 <= value <= 4:
                return value - 1
        return None

    if label_style == "parenthetical":
        token = token.strip("()").lower()
        mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
        return mapping.get(token)

    token = token.upper()
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(token)


def parse_answer(response: str, label_style: str = "alpha_upper") -> int | None:
    if not response:
        return None

    text = response.strip()
    for pattern in _LEADING_PATTERNS:
        match = pattern.match(text)
        if match:
            idx = _index_from_token(match.group(1), label_style)
            if idx is not None:
                return idx

    for pattern in _FALLBACK_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            idx = _index_from_token(matches[-1], label_style)
            if idx is not None:
                return idx

    return None


def checkpoint_key(model_id: str, condition_id: str, question_id: str) -> str:
    return f"{model_id}|{condition_id}|{question_id}"


def _results_path(raw_dir: Path) -> Path:
    return raw_dir / RESULTS_FILENAME


def _checkpoint_path(raw_dir: Path) -> Path:
    return raw_dir / "checkpoints" / CHECKPOINT_FILENAME


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = read_json(path)
    return set(data.get("completed_keys", []))


def _save_checkpoint(path: Path, completed_keys: set[str], seed: int) -> None:
    atomic_write_json(
        path,
        {
            "completed_keys": sorted(completed_keys),
            "seed": seed,
        },
    )


def _build_eval_queue(
    models: list[ModelConfig],
    perturbed_paths: dict[str, Path],
) -> list[EvalTask]:
    questions_by_condition: dict[str, dict[str, PerturbedQuestion]] = {}
    for condition_id, path in perturbed_paths.items():
        questions = load_perturbed_questions(path)
        questions_by_condition[condition_id] = {
            q.original_id: q for q in questions
        }

    question_ids = sorted(
        {
            qid
            for by_qid in questions_by_condition.values()
            for qid in by_qid
        }
    )
    condition_ids = sorted(perturbed_paths.keys())
    model_ids = [m.id for m in models]

    tasks: list[EvalTask] = []
    for question_id in question_ids:
        for condition_id in condition_ids:
            question = questions_by_condition[condition_id].get(question_id)
            if question is None:
                continue
            for model in models:
                tasks.append(
                    EvalTask(
                        model_id=model.id,
                        condition_id=condition_id,
                        question_id=question_id,
                        question=question,
                    )
                )
    return tasks


def run_evaluation(
    models: list[ModelConfig],
    perturbed_paths: dict[str, Path],
    raw_results_dir: Path,
    seed: int,
) -> Path:
    """Single evaluation loop: question → condition → model, with question-level checkpoint."""
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(raw_results_dir)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(raw_results_dir)

    clients = {m.id: build_model_client(m) for m in models}
    completed = _load_checkpoint(checkpoint_path)
    queue = _build_eval_queue(models, perturbed_paths)

    total = len(queue)
    done_before = len(completed)
    logger.info(
        "Evaluation queue: %s tasks (%s already checkpointed)",
        total,
        done_before,
    )

    for i, task in enumerate(queue, start=1):
        key = checkpoint_key(task.model_id, task.condition_id, task.question_id)
        if key in completed:
            continue

        client = clients[task.model_id]
        prompt = build_prompt(task.question)
        response = client.complete(prompt)
        parsed = (
            parse_answer(response, task.question.label_style) if response else None
        )
        is_correct = (
            parsed == task.question.correct_answer_index
            if parsed is not None
            else None
        )

        record = EvalRecord(
            question_id=task.question_id,
            subject=task.question.subject,
            condition_id=task.condition_id,
            perturbation_type=task.question.perturbation_type,
            perturbation_params=task.question.perturbation_params,
            model_id=task.model_id,
            model_response=response,
            parsed_answer=parsed,
            is_correct=is_correct,
            original_correct_answer=task.question.correct_answer_index,
            seed=seed,
        )
        append_jsonl(results_path, record)
        completed.add(key)
        _save_checkpoint(checkpoint_path, completed, seed)

        if i % 25 == 0 or i == total:
            logger.info("Evaluation progress: %s / %s tasks", len(completed), total)

    return results_path


def build_model_client(model: ModelConfig) -> ModelClient:
    from src.io_utils import get_env_api_key

    return ModelClient(
        model_id=model.id,
        base_url=model.base_url,
        model=model.model,
        api_key=get_env_api_key(model.api_key_env),
    )
