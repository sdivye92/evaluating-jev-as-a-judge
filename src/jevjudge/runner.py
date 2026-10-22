"""Run a judge over items with an on-disk cache so interrupted runs resume for free."""

from __future__ import annotations

import csv
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from .judges import Judge
from .schema import RESULTS_DIR, BinaryItem, RubricItem

RAW_DIR = RESULTS_DIR / "raw"

# results/calls.csv: one row per API call with the raw model output, so every verdict / score can be
# traced and re-analysed without calling the APIs again. A retried item appends new rows.
CALL_COLUMNS = ["judge", "task", "item_id", "criterion", "sample", "raw_output", "parsed", "stop_reason",
                "input_tokens", "output_tokens", "latency_s", "error", "served_model", "thinking"]


def raw_path(judge_name: str, task: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', judge_name)}__{task}.jsonl"


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """Latest record per item id (later lines win, so a retry replaces an earlier error)."""
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["item_id"]] = rec
    return out


def run_judge(
    judge: Judge,
    task: str,
    items: Sequence[BinaryItem | RubricItem],
    workers: int = 4,
    raw_dir: Path = RAW_DIR,
    calls_path: Path | None = None,
) -> Path:
    """Judge every item not already cached without error.

    Appends one summary record per item to ``raw_dir`` (JSONL) and one row per API call to
    ``calls_path`` (default: ``calls.csv`` next to ``raw_dir``).
    """
    calls_path = calls_path or raw_dir.parent / "calls.csv"
    path = raw_path(judge.name, task, raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {k for k, v in load_records(path).items() if not v.get("error")}
    todo = [it for it in items if it.id not in done]
    if not todo:
        return path
    call = judge.judge_binary if task == "binary" else judge.judge_rubric
    lock = threading.Lock()

    def work(item: BinaryItem | RubricItem) -> dict[str, Any]:
        return {"item_id": item.id, "judge": judge.name, "task": task, **asdict(call(item))}

    new_csv = not calls_path.exists() or calls_path.stat().st_size == 0
    with path.open("a") as f, calls_path.open("a", newline="") as cf, ThreadPoolExecutor(max_workers=workers) as ex:
        writer = csv.DictWriter(cf, fieldnames=CALL_COLUMNS)
        if new_csv:
            writer.writeheader()
        futures = [ex.submit(work, it) for it in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{judge.name} {task}"):
            rec = fut.result()
            calls = rec.pop("calls", [])
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                for row in calls:
                    writer.writerow({"judge": judge.name, "task": task, "item_id": rec["item_id"], **row})
                cf.flush()
    return path
