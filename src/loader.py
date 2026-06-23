from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.io_utils import dataclass_to_dict, write_jsonl
from src.types import Question


def load_mmlu_test() -> list[tuple[Question, int]]:
    dataset = load_dataset("cais/mmlu", "all", split="test")
    questions: list[tuple[Question, int]] = []
    for idx, row in enumerate(dataset):
        subject = str(row["subject"])
        question = Question(
            original_id=f"{subject}:{idx}",
            subject=subject,
            question_text=str(row["question"]),
            options=[str(c) for c in row["choices"]],
            correct_answer_index=int(row["answer"]),
        )
        questions.append((question, idx))
    return questions


def stratified_sample(
    questions: list[tuple[Question, int]],
    n: int,
    seed: int,
) -> list[Question]:
    if n <= 0:
        raise ValueError("sample size must be positive")
    if n > len(questions):
        raise ValueError(f"sample size {n} exceeds dataset size {len(questions)}")

    rng = random.Random(seed)
    by_subject: dict[str, list[tuple[Question, int]]] = defaultdict(list)
    for item in questions:
        by_subject[item[0].subject].append(item)

    subjects = sorted(by_subject.keys())
    base = n // len(subjects)
    remainder = n % len(subjects)

    extra_candidates = sorted(
        subjects,
        key=lambda s: (-len(by_subject[s]), s),
    )
    extra_subjects = set(extra_candidates[:remainder])

    sampled: list[Question] = []
    for subject in subjects:
        count = base + (1 if subject in extra_subjects else 0)
        pool = by_subject[subject]
        if count > len(pool):
            raise ValueError(
                f"subject {subject} has only {len(pool)} questions, need {count}"
            )
        chosen = rng.sample(pool, count)
        sampled.extend(q for q, _ in chosen)

    sampled.sort(key=lambda q: q.original_id)
    return sampled


def save_sampled(questions: list[Question], path: str | Path) -> None:
    write_jsonl(path, (dataclass_to_dict(q) for q in questions))


def load_or_sample(
    path: str | Path,
    sample_size: int,
    seed: int,
    force_resample: bool = False,
) -> list[Question]:
    path = Path(path)
    if path.exists() and not force_resample:
        from src.io_utils import load_questions

        return load_questions(path)

    all_questions = load_mmlu_test()
    sampled = stratified_sample(all_questions, sample_size, seed)
    save_sampled(sampled, path)
    return sampled
