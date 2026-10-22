"""Build the evaluation datasets as JSONL under ``data/``.

Binary (pointwise correctness, one correct and one incorrect candidate per question):
  * gsm8k    - candidate is just a final answer; wrong answers are near-miss numbers.
               The judge must solve the problem (reasoning-heavy).
  * mbpp     - candidate is code; label comes from executing the tests. Incorrect
               candidates are single-edit mutants that fail with an AssertionError.
  * halueval - candidate answer vs. supplied knowledge (hallucination detection).
  * private  - optional ``data/private.jsonl`` (your own model outputs, see README).

Rubric: HelpSteer2 human ratings (0-4) for helpfulness, correctness, coherence.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from .mbpp_exec import make_incorrect, normalize, run_tests
from .rubrics import CRITERIA
from .schema import DATA_DIR, BinaryItem, RubricItem, read_jsonl, write_jsonl

# questions per binary source (2 items each); rubric is a count of items
DEFAULT_COUNTS = {"gsm8k": 200, "halueval": 200, "mbpp": 150, "rubric": 300}


# ---------------------------------------------------------------------------- GSM8K


def _gsm8k_gold(answer: str) -> int | None:
    tail = answer.split("####")[-1].strip().replace(",", "")
    try:
        return int(tail)
    except ValueError:
        return None


def _swap_digits(n: int, rng: random.Random) -> int | None:
    s = str(abs(n))
    if len(s) < 2:
        return None
    i = rng.randrange(len(s) - 1)
    t = list(s)
    t[i], t[i + 1] = t[i + 1], t[i]
    return int("".join(t)) * (1 if n >= 0 else -1)


def wrong_number(gold: int, question: str, rng: random.Random) -> int:
    """A plausible wrong answer: off-by-small, a dropped step, or a question number mixed in."""
    nums = [int(x) for x in re.findall(r"\d+", question.replace(",", ""))]
    cands = {gold + 1, gold - 1, gold + 10, gold - 10, gold * 2}
    if gold % 2 == 0:
        cands.add(gold // 2)
    swapped = _swap_digits(gold, rng)
    if swapped is not None:
        cands.add(swapped)
    for n in nums[:6]:
        cands.update({gold + n, gold - n})
    cands = {c for c in cands if c != gold and c >= 0}
    return rng.choice(sorted(cands))


def build_gsm8k(n: int, seed: int) -> list[BinaryItem]:
    from datasets import load_dataset

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[BinaryItem] = []
    for idx, row in enumerate(rows):
        if len(out) >= 2 * n:
            break
        gold = _gsm8k_gold(row["answer"])
        if gold is None:
            continue
        group = f"gsm8k-{idx}"
        wrong = wrong_number(gold, row["question"], rng)
        for tag, value, label in (("c", gold, 1), ("w", wrong, 0)):
            out.append(BinaryItem(id=f"{group}-{tag}", group=group, source="gsm8k", question=row["question"],
                                  candidate=f"The answer is {value}", label=label))
    return out


# -------------------------------------------------------------------------- HaluEval

# HaluEval's hallucinated answers are usually full sentences while the right answers are
# short phrases (median 58 vs 12 chars; "shorter => correct" gives AUROC 0.97), so a judge
# could score well from length alone. Keeping only pairs of similar length removes that
# shortcut (residual AUROC ~0.51) at the cost of a smaller pool (~380 questions).
HALUEVAL_LEN_RATIO = (0.8, 1.25)


def build_halueval(n: int, seed: int) -> list[BinaryItem]:
    from datasets import load_dataset

    lo, hi = HALUEVAL_LEN_RATIO
    rows = [
        r
        for r in load_dataset("pminervini/HaluEval", "qa", split="data")
        if lo <= len(r["hallucinated_answer"]) / max(1, len(r["right_answer"])) <= hi
    ]
    if n > len(rows):
        raise ValueError(f"HaluEval length-matched pool has only {len(rows)} questions; requested {n}")
    random.Random(seed).shuffle(rows)
    out: list[BinaryItem] = []
    for idx, row in enumerate(rows[:n]):
        group = f"halueval-{idx}"
        for tag, ans, label in (("c", row["right_answer"], 1), ("w", row["hallucinated_answer"], 0)):
            out.append(BinaryItem(id=f"{group}-{tag}", group=group, source="halueval", question=row["question"],
                                  candidate=ans, label=label, knowledge=row["knowledge"]))
    return out


# ----------------------------------------------------------------------------- MBPP


def build_mbpp(n: int, seed: int) -> list[BinaryItem]:
    from datasets import load_dataset

    rows = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[BinaryItem] = []
    for row in rows:
        if len(out) >= 2 * n:
            break
        tests, imports = list(row["test_list"]), list(row["test_imports"] or [])
        gold = normalize(row["code"])
        if not run_tests(gold, imports, tests).passed:
            continue
        made = make_incorrect(gold, imports, tests, rng)
        if made is None:
            continue
        group = f"mbpp-{row['task_id']}"
        for tag, code, label in (("c", gold, 1), ("w", made[0], 0)):
            out.append(BinaryItem(id=f"{group}-{tag}", group=group, source="mbpp", question=row["prompt"],
                                  candidate=code, label=label, example_test=tests[0]))
    return out


# -------------------------------------------------------------------------- private


def load_private(path: Path) -> list[BinaryItem]:
    """Optional user-supplied binary items. One JSON object per line with fields:
    id, question, candidate, label (1/0); optional: knowledge, group.
    """
    if not path.exists():
        return []
    return [
        BinaryItem(id=f"private-{d['id']}", group=str(d.get("group", d["id"])), source="private",
                   question=d["question"], candidate=d["candidate"], label=int(d["label"]),
                   knowledge=d.get("knowledge"))
        for d in read_jsonl(path)
    ]


# ---------------------------------------------------------------------- HelpSteer2

_TURN = re.compile(r"<extra_id_\d>(User|Assistant)?\n?")


def clean_helpsteer_prompt(prompt: str) -> str:
    """Multi-turn prompts use <extra_id_N> markers; render them as readable turns."""
    return _TURN.sub(lambda m: f"\n\n{m.group(1)}: " if m.group(1) else "\n\n", prompt).strip()


def build_helpsteer2(n: int, seed: int) -> list[RubricItem]:
    from datasets import load_dataset

    rows = list(load_dataset("nvidia/HelpSteer2", split="validation"))
    random.Random(seed).shuffle(rows)
    return [
        RubricItem(id=f"helpsteer2-{i}", source="helpsteer2", prompt=clean_helpsteer_prompt(row["prompt"]),
                   response=row["response"], labels={c: int(row[c]) for c in CRITERIA})
        for i, row in enumerate(rows[:n])
    ]


# ------------------------------------------------------------------------- orchestrate


def build_all(
    data_dir: Path = DATA_DIR,
    counts: dict[str, int] | None = None,
    seed: int = 0,
    tasks: tuple[str, ...] = ("binary", "rubric"),
) -> dict[str, int]:
    """Write data/binary.jsonl and data/rubric.jsonl; returns the number of items written."""
    counts = {**DEFAULT_COUNTS, **(counts or {})}
    written: dict[str, int] = {}
    if "binary" in tasks:
        items = build_gsm8k(counts["gsm8k"], seed) + build_halueval(counts["halueval"], seed)
        items += build_mbpp(counts["mbpp"], seed) + load_private(data_dir / "private.jsonl")
        written["binary.jsonl"] = write_jsonl(data_dir / "binary.jsonl", items)
    if "rubric" in tasks:
        written["rubric.jsonl"] = write_jsonl(data_dir / "rubric.jsonl", build_helpsteer2(counts["rubric"], seed))
    return written
