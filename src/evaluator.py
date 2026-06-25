from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    "none": "Reply with only the option label (e.g. A), nothing else.",
    "minimal": "Reply with only the option label (e.g. A), nothing else.",
    "verbose": "Reply with only the option label (e.g. A), nothing else.",
}

MAX_TASK_ATTEMPTS = 5
_RETRY_IN_MESSAGE_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _parse_duration_seconds(value: str) -> float | None:
    text = value.strip()
    if text.endswith("s"):
        try:
            return float(text[:-1])
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_retry_seconds(exc: Exception) -> float | None:
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code != 429:
        return None

    retry_after = exc.response.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass

    try:
        payload = exc.response.json()
    except Exception:  # noqa: BLE001
        payload = None

    if payload is None:
        return None

    bodies: list[Any] = [payload]
    if isinstance(payload, list):
        bodies = payload

    for body in bodies:
        if not isinstance(body, dict):
            continue
        error = body.get("error", body)
        if isinstance(error, dict):
            message = str(error.get("message", ""))
            match = _RETRY_IN_MESSAGE_RE.search(message)
            if match:
                return float(match.group(1))
            for detail in error.get("details", []):
                if not isinstance(detail, dict):
                    continue
                retry_delay = detail.get("retryDelay")
                if retry_delay is not None:
                    parsed = _parse_duration_seconds(str(retry_delay))
                    if parsed is not None:
                        return parsed

    return None


@dataclass(frozen=True)
class EvalTask:
    model_id: str
    condition_id: str
    question_id: str
    question: PerturbedQuestion


@dataclass
class ModelClient:
    model_id: str
    provider: str
    base_url: str
    model: str
    api_key: str
    min_request_interval_seconds: float = 0.0
    _rate_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    def _throttle(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = self.min_request_interval_seconds - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _retry_wait(self, exc: Exception, attempt: int) -> None:
        delay = _parse_retry_seconds(exc)
        if delay is not None:
            sleep_for = delay + 0.5
            logger.info(
                "Model %s rate limited; sleeping %.1fs",
                self.model_id,
                sleep_for,
            )
            time.sleep(sleep_for)
            return
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            time.sleep(max(10.0, 2**attempt))
            return
        if attempt < 2:
            time.sleep(2**attempt)

    def complete(self, prompt: str) -> str | None:
        self._throttle()
        if self.provider == "anthropic":
            return self._complete_anthropic(prompt)
        return self._complete_openai(prompt)

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status == 429 or status >= 500
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status == 429 or status >= 500
        return True

    def _format_error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                body = exc.response.json()
            except Exception:  # noqa: BLE001
                body = exc.response.text
            return f"{exc} body={body}"
        return str(exc)

    def _complete_openai(self, prompt: str) -> str | None:
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
        for attempt in range(5):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Model %s attempt %s failed: %s",
                    self.model_id,
                    attempt + 1,
                    self._format_error(exc),
                )
                if attempt < 4 and self._should_retry(exc):
                    self._retry_wait(exc, attempt)
                elif not self._should_retry(exc):
                    break

        logger.error(
            "Model %s failed after retries: %s",
            self.model_id,
            self._format_error(last_error) if last_error else "unknown",
        )
        return None

    def _complete_anthropic(self, prompt: str) -> str | None:
        import anthropic

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                client = anthropic.Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=16,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Model %s attempt %s failed: %s",
                    self.model_id,
                    attempt + 1,
                    self._format_error(exc),
                )
                if attempt < 2 and self._should_retry(exc):
                    self._retry_wait(exc, attempt)
                elif not self._should_retry(exc):
                    break

        logger.error(
            "Model %s failed after retries: %s",
            self.model_id,
            self._format_error(last_error) if last_error else "unknown",
        )
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
    re.compile(r"(?:answer|option|choice)(?:\s+is)?\s*:?\s*\(?([A-Da-d1-4])\)?", re.IGNORECASE),
    re.compile(r"\b(?:choice|option)\s+([A-Da-d1-4])\b", re.IGNORECASE),
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

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        idx = _index_from_token(lines[-1], label_style)
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


def _load_valid_completed_keys(checkpoint_path: Path, results_path: Path) -> set[str]:
    """Checkpoint keys count only when a parseable answer exists in results."""
    checkpointed = _load_checkpoint(checkpoint_path)
    if not results_path.exists():
        return set()

    from src.io_utils import read_jsonl

    valid_keys: set[str] = set()
    for record in read_jsonl(results_path):
        if record.get("parsed_answer") is None:
            continue
        valid_keys.add(
            checkpoint_key(
                str(record["model_id"]),
                str(record["condition_id"]),
                str(record["question_id"]),
            )
        )

    scrubbed = checkpointed & valid_keys
    if scrubbed != checkpointed:
        logger.info(
            "Scrubbed %s checkpoint keys lacking a parseable answer",
            len(checkpointed) - len(scrubbed),
        )
    return scrubbed


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
    concurrency: int = 1,
) -> tuple[Path, bool]:
    """Evaluate pending tasks with optional parallel API calls (checkpoint-safe).

    Returns (results_path, is_complete). is_complete is True only when every
    (model, condition, question) cell has a parseable answer. If False, re-run
    to retry remaining cells — the checkpoint prevents redundant API calls.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    raw_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(raw_results_dir)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    results_path = _results_path(raw_results_dir)

    clients = {m.id: build_model_client(m) for m in models}
    completed = _load_valid_completed_keys(checkpoint_path, results_path)
    if completed != _load_checkpoint(checkpoint_path):
        _save_checkpoint(checkpoint_path, completed, seed)
    queue = _build_eval_queue(models, perturbed_paths)

    total = len(queue)
    done_before = len(completed)
    pending = [
        task
        for task in queue
        if checkpoint_key(task.model_id, task.condition_id, task.question_id)
        not in completed
    ]
    rng = random.Random(seed)
    rng.shuffle(pending)

    logger.info(
        "Evaluation queue: %s tasks (%s done, %s pending), concurrency=%s",
        total,
        done_before,
        len(pending),
        concurrency,
    )

    if not pending:
        return results_path, True

    lock = threading.Lock()
    pending_total = len(pending)
    completed_this_run = 0

    def process_task(task: EvalTask) -> None:
        nonlocal completed_this_run
        key = checkpoint_key(task.model_id, task.condition_id, task.question_id)

        with lock:
            if key in completed:
                return

        client = clients[task.model_id]
        prompt = build_prompt(task.question)
        response: str | None = None
        parsed: int | None = None
        for attempt in range(MAX_TASK_ATTEMPTS):
            response = client.complete(prompt)
            if response is None:
                logger.warning(
                    "API failure for %s (attempt %s/%s); retrying",
                    key,
                    attempt + 1,
                    MAX_TASK_ATTEMPTS,
                )
                continue
            parsed = parse_answer(response, task.question.label_style)
            if parsed is not None:
                break
            logger.warning(
                "Unparseable response for %s (attempt %s/%s); retrying",
                key,
                attempt + 1,
                MAX_TASK_ATTEMPTS,
            )

        if response is None or parsed is None:
            logger.warning(
                "Incomplete cell %s after %s attempts; not checkpointing",
                key,
                MAX_TASK_ATTEMPTS,
            )
            return

        is_correct = parsed == task.question.correct_answer_index

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

        with lock:
            if key in completed:
                return
            append_jsonl(results_path, record)
            completed.add(key)
            _save_checkpoint(checkpoint_path, completed, seed)
            completed_this_run += 1
            if completed_this_run % 25 == 0 or completed_this_run == pending_total:
                logger.info(
                    "Evaluation progress: %s / %s pending (%s total checkpointed / %s)",
                    completed_this_run,
                    pending_total,
                    len(completed),
                    total,
                )

    if concurrency == 1:
        for task in pending:
            process_task(task)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_task, task) for task in pending]
            for future in as_completed(futures):
                future.result()

    still_pending = [
        task
        for task in queue
        if checkpoint_key(task.model_id, task.condition_id, task.question_id)
        not in completed
    ]
    is_complete = len(still_pending) == 0
    if not is_complete:
        logger.error(
            "Evaluation incomplete: %s / %s cells missing parseable answers. "
            "Re-run to retry remaining tasks.",
            len(still_pending),
            total,
        )
        by_condition: dict[str, int] = {}
        for task in still_pending:
            by_condition[task.condition_id] = by_condition.get(task.condition_id, 0) + 1
        for cond_id, cnt in sorted(by_condition.items()):
            logger.error("  Missing %s cells for condition '%s'", cnt, cond_id)
    else:
        logger.info("Evaluation complete: all %s cells filled", total)

    return results_path, is_complete


def build_model_client(model: ModelConfig) -> ModelClient:
    from src.io_utils import get_env_api_key

    return ModelClient(
        model_id=model.id,
        provider=model.provider,
        base_url=model.base_url,
        model=model.model,
        api_key=get_env_api_key(model.api_key_env),
        min_request_interval_seconds=model.min_request_interval_seconds,
    )
