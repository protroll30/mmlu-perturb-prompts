from __future__ import annotations

from typing import Any

from src.types import Condition, PerturbationStep

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "label_remap": {"style": "numeric"},
    "option_shuffle": {},
    "context_inject": {
        "text": "The following is a multiple choice question.",
    },
    "instruction_style": {"style": "minimal"},
    "semantic_paraphrase": {},
}

STEP_VARIANTS: dict[str, list[dict[str, Any]]] = {
    "label_remap": [
        {"style": "numeric"},
        {"style": "parenthetical"},
    ],
    "option_shuffle": [{}],
    "context_inject": [DEFAULT_PARAMS["context_inject"]],
    "instruction_style": [
        {"style": "none"},
        {"style": "minimal"},
        {"style": "verbose"},
    ],
    "semantic_paraphrase": [{}],
}

REGISTERED_STEPS = frozenset(STEP_VARIANTS.keys())


def _step_token(step: PerturbationStep) -> str:
    if step.name == "label_remap":
        return f"label_remap:{step.params.get('style', 'numeric')}"
    if step.name == "instruction_style":
        return f"instruction_style:{step.params.get('style', 'minimal')}"
    return step.name


def build_condition_id(steps: tuple[PerturbationStep, ...]) -> str:
    if not steps:
        return "original"
    return "+".join(_step_token(step) for step in steps)


def _parse_step_token(token: str) -> PerturbationStep:
    token = token.strip()
    if not token:
        raise ValueError("Empty perturbation step token")

    if token == "original":
        return PerturbationStep(name="original", params={})

    if ":" in token:
        name, value = token.split(":", 1)
        name = name.strip()
        value = value.strip()
        if name not in REGISTERED_STEPS:
            raise ValueError(f"Unknown perturbation step: {name}")
        if name == "label_remap":
            if value not in {"numeric", "parenthetical"}:
                raise ValueError(f"Invalid label_remap style: {value}")
            return PerturbationStep(name=name, params={"style": value})
        if name == "instruction_style":
            if value not in {"none", "minimal", "verbose"}:
                raise ValueError(f"Invalid instruction_style style: {value}")
            return PerturbationStep(name=name, params={"style": value})
        raise ValueError(f"Step {name} does not accept parameter suffix :{value}")

    if token not in REGISTERED_STEPS:
        raise ValueError(f"Unknown perturbation step: {token}")

    defaults = dict(DEFAULT_PARAMS[token])
    return PerturbationStep(name=token, params=defaults)


def parse_condition_token(token: str) -> Condition:
    token = token.strip()
    if token == "original":
        return Condition(condition_id="original", steps=())

    parts = [part.strip() for part in token.split("+") if part.strip()]
    steps = tuple(_parse_step_token(part) for part in parts)
    condition_id = build_condition_id(steps)
    return Condition(condition_id=condition_id, steps=steps)


def expand_all_conditions() -> list[Condition]:
    conditions = [Condition(condition_id="original", steps=())]
    for step_name, variants in STEP_VARIANTS.items():
        for params in variants:
            step = PerturbationStep(name=step_name, params=dict(params))
            conditions.append(
                Condition(
                    condition_id=build_condition_id((step,)),
                    steps=(step,),
                )
            )
    return conditions


def parse_conditions_arg(value: str | None) -> list[Condition]:
    if value is None or value.strip().lower() == "all":
        return expand_all_conditions()

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No conditions specified")
    return [parse_condition_token(token) for token in tokens]


def merge_step_params(step: PerturbationStep) -> PerturbationStep:
    defaults = dict(DEFAULT_PARAMS.get(step.name, {}))
    merged = {**defaults, **step.params}
    return PerturbationStep(name=step.name, params=merged)


def normalize_condition(condition: Condition) -> Condition:
    if condition.is_original:
        return condition
    normalized_steps = tuple(merge_step_params(step) for step in condition.steps)
    return Condition(
        condition_id=build_condition_id(normalized_steps),
        steps=normalized_steps,
    )
