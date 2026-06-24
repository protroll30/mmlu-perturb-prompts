from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Question:
    original_id: str
    subject: str
    question_text: str
    options: list[str]
    correct_answer_index: int


@dataclass(frozen=True)
class PerturbedQuestion(Question):
    perturbation_type: str
    perturbation_params: dict[str, Any]
    condition_id: str
    label_style: str = "alpha_upper"


@dataclass(frozen=True)
class EvalRecord:
    question_id: str
    subject: str
    condition_id: str
    perturbation_type: str
    perturbation_params: dict[str, Any]
    model_id: str
    model_response: str | None
    parsed_answer: int | None
    is_correct: bool | None
    original_correct_answer: int
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    id: str
    model: str
    api_key_env: str
    provider: str = "openai"
    base_url: str = ""
    min_request_interval_seconds: float = 0.0


@dataclass(frozen=True)
class ParaphraseConfig:
    model: str
    api_key_env: str


@dataclass(frozen=True)
class PathsConfig:
    sampled: str
    perturbed_dir: str
    paraphrase_cache_dir: str
    raw_results_dir: str
    metrics_dir: str
    figures_dir: str


@dataclass(frozen=True)
class AppConfig:
    seed: int
    sample_size: int
    paths: PathsConfig
    models: list[ModelConfig]
    paraphrase: ParaphraseConfig
    eval_concurrency: int = 1
    min_questions_per_subject: int = 5


@dataclass(frozen=True)
class PerturbationStep:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Condition:
    condition_id: str
    steps: tuple[PerturbationStep, ...]

    @property
    def is_original(self) -> bool:
        return self.condition_id == "original"
