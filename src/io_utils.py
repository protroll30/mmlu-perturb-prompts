from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import yaml

from src.types import (
    AppConfig,
    Condition,
    EvalRecord,
    ModelConfig,
    ParaphraseConfig,
    PathsConfig,
    PerturbedQuestion,
    PerturbationStep,
    Question,
)

T = TypeVar("T")

_dotenv_loaded = False


def load_dotenv_file() -> None:
    """Load variables from .env in the project root (once per process)."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return

    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    _dotenv_loaded = True


def _parse_model_config(raw: dict[str, Any]) -> ModelConfig:
    provider = str(raw.get("provider", "openai"))
    base_url = str(raw.get("base_url", ""))
    if provider == "openai" and not base_url:
        raise ValueError(f"Model {raw['id']} requires base_url when provider is openai")
    return ModelConfig(
        id=str(raw["id"]),
        model=str(raw["model"]),
        api_key_env=str(raw["api_key_env"]),
        provider=provider,
        base_url=base_url,
        min_request_interval_seconds=float(raw.get("min_request_interval_seconds", 0.0)),
    )


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths = PathsConfig(**raw["paths"])
    models = [_parse_model_config(m) for m in raw["models"]]
    paraphrase = ParaphraseConfig(**raw["paraphrase"])
    return AppConfig(
        seed=int(raw["seed"]),
        sample_size=int(raw["sample_size"]),
        paths=paths,
        models=models,
        paraphrase=paraphrase,
        eval_concurrency=int(raw.get("eval_concurrency", 1)),
        min_questions_per_subject=int(raw.get("min_questions_per_subject", 5)),
    )


def get_env_api_key(env_var: str) -> str:
    load_dotenv_file()
    value = os.environ.get(env_var)
    if not value:
        raise ValueError(f"Missing required environment variable: {env_var}")
    return value


def _api_key_is_set(env_var: str) -> bool:
    load_dotenv_file()
    value = os.environ.get(env_var)
    return bool(value and value.strip())


def validate_api_keys(
    config: AppConfig,
    models: list[ModelConfig],
    *,
    need_paraphrase: bool,
    need_eval: bool,
) -> None:
    """Fail fast before costly stages if required API keys are missing."""
    required: dict[str, str] = {}

    if need_eval:
        for model in models:
            required[model.api_key_env] = f"eval model '{model.id}'"

    if need_paraphrase:
        required[config.paraphrase.api_key_env] = "semantic_paraphrase cache warm"

    missing = [
        env_var
        for env_var in sorted(required)
        if not _api_key_is_set(env_var)
    ]
    if not missing:
        return

    lines = "\n".join(f"  - {var} ({required[var]})" for var in missing)
    raise ValueError(
        "Missing API keys in environment or .env. Add them before running:\n"
        f"{lines}"
    )


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any] | Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "__dataclass_fields__"):
                payload = dataclass_to_dict(record)
            else:
                payload = record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any] | Any) -> None:
    ensure_parent_dir(path)
    if hasattr(record, "__dataclass_fields__"):
        payload = dataclass_to_dict(record)
    else:
        payload = record
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(obj)


def question_from_dict(data: dict[str, Any]) -> Question:
    return Question(
        original_id=data["original_id"],
        subject=data["subject"],
        question_text=data["question_text"],
        options=list(data["options"]),
        correct_answer_index=int(data["correct_answer_index"]),
    )


def perturbed_question_from_dict(data: dict[str, Any]) -> PerturbedQuestion:
    return PerturbedQuestion(
        original_id=data["original_id"],
        subject=data["subject"],
        question_text=data["question_text"],
        options=list(data["options"]),
        correct_answer_index=int(data["correct_answer_index"]),
        perturbation_type=data["perturbation_type"],
        perturbation_params=dict(data.get("perturbation_params", {})),
        condition_id=data["condition_id"],
        label_style=data.get("label_style", "alpha_upper"),
    )


def eval_record_from_dict(data: dict[str, Any]) -> EvalRecord:
    return EvalRecord(
        question_id=data["question_id"],
        subject=data["subject"],
        condition_id=data["condition_id"],
        perturbation_type=data["perturbation_type"],
        perturbation_params=dict(data.get("perturbation_params", {})),
        model_id=data["model_id"],
        model_response=data.get("model_response"),
        parsed_answer=data.get("parsed_answer"),
        is_correct=data.get("is_correct"),
        original_correct_answer=int(data["original_correct_answer"]),
        seed=int(data["seed"]),
    )


def load_questions(path: str | Path) -> list[Question]:
    return [question_from_dict(row) for row in read_jsonl(path)]


def load_perturbed_questions(path: str | Path) -> list[PerturbedQuestion]:
    return [perturbed_question_from_dict(row) for row in read_jsonl(path)]


def load_eval_records(path: str | Path) -> list[EvalRecord]:
    return [eval_record_from_dict(row) for row in read_jsonl(path)]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    target = Path(path)
    last_error: OSError | None = None
    for attempt in range(5):
        temp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            temp.replace(target)
            return
        except OSError as exc:
            last_error = exc
            if temp.exists():
                temp.unlink(missing_ok=True)
            time.sleep(0.05 * (2**attempt))
    raise last_error if last_error else OSError(f"Failed to write {target}")


def condition_to_filename(condition_id: str) -> str:
    safe = condition_id.replace("/", "_").replace("\\", "_")
    return f"{safe}.jsonl"


def iter_raw_result_files(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        return []
    unified = root / "results.jsonl"
    if unified.exists():
        return [unified]
    return sorted(p for p in root.glob("*.jsonl") if p.is_file())


def map_records(records: Iterable[dict[str, Any]], fn: Callable[[dict[str, Any]], T]) -> list[T]:
    return [fn(r) for r in records]
