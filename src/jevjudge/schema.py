"""Item types and JSONL helpers shared by data building, judging and reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass
class BinaryItem:
    """One question + one candidate answer with a ground-truth correctness label."""

    id: str
    group: str  # items built from the same question share a group (for cluster bootstrap)
    source: str  # gsm8k | mbpp | halueval | private
    question: str
    candidate: str
    label: int  # 1 = candidate is correct, 0 = incorrect
    knowledge: str | None = None  # evidence the judge may rely on (HaluEval)
    example_test: str | None = None  # one visible unit test (MBPP)


@dataclass
class RubricItem:
    """One prompt + one response with human 0-4 ratings per criterion."""

    id: str
    source: str
    prompt: str
    response: str
    labels: dict[str, int]


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_binary(data_dir: Path = DATA_DIR) -> list[BinaryItem]:
    path = data_dir / "binary.jsonl"
    return [BinaryItem(**d) for d in read_jsonl(path)] if path.exists() else []


def load_rubric(data_dir: Path = DATA_DIR) -> list[RubricItem]:
    path = data_dir / "rubric.jsonl"
    return [RubricItem(**d) for d in read_jsonl(path)] if path.exists() else []
