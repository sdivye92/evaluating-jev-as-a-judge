"""Runner cache and the summarize() printout, using fake judges (no network)."""

import csv
import json
import random

from jevjudge.evaluate import summarize
from jevjudge.judges import BinaryResult, Judge, RubricResult
from jevjudge.rubrics import CRITERIA
from jevjudge.runner import load_records, raw_path, run_judge
from jevjudge.schema import BinaryItem, RubricItem, write_jsonl


class FakeJudge(Judge):
    def __init__(self, name, acc, latency=0.1, seed=0):
        self.name, self.acc, self.latency = name, acc, latency
        self.rng, self.calls = random.Random(seed), 0

    def judge_binary(self, item):
        self.calls += 1
        said_correct = bool(item.label) == (self.rng.random() < self.acc)
        return BinaryResult(p_correct=0.9 if said_correct else 0.1, latency_s=self.latency,
                            input_tokens=100, output_tokens=2,
                            calls=[{"criterion": "correct", "sample": n, "raw_output": f'{{"verdict": "YES"}}, "x"',
                                    "parsed": 1} for n in range(2)])

    def judge_rubric(self, item):
        self.calls += 1
        scores = {c: min(4.0, max(0.0, item.labels[c] + self.rng.choice([-1, 0, 0, 1]))) for c in CRITERIA}
        return RubricResult(scores=scores, probs={c: [0.2] * 5 for c in CRITERIA}, latency_s=self.latency)


def binary_items(n):
    return [BinaryItem(id=f"gsm8k-{i}-{t}", group=f"gsm8k-{i}", source="gsm8k",
                       question=f"q{i}", candidate="x" * (5 + 3 * lab), label=lab)
            for i in range(n) for t, lab in (("c", 1), ("w", 0))]


def rubric_items(n):
    rng = random.Random(1)
    return [RubricItem(id=f"hs-{i}", source="helpsteer2", prompt="p", response="r",
                       labels={c: rng.choice([2, 3, 3, 4, 4]) for c in CRITERIA}) for i in range(n)]


def test_runner_resumes_and_retries_only_errors(tmp_path):
    items, j = binary_items(5), FakeJudge("fake", 1.0)
    run_judge(j, "binary", items, workers=2, raw_dir=tmp_path)
    run_judge(j, "binary", items, workers=2, raw_dir=tmp_path)
    assert j.calls == 10  # second run fully cached

    path = raw_path("fake", "binary", tmp_path)
    first = next(iter(load_records(path).values()))
    with path.open("a") as f:  # an errored record is superseded by the retry
        f.write(json.dumps(dict(first, error="boom", p_correct=None)) + "\n")
    run_judge(j, "binary", items, workers=2, raw_dir=tmp_path)
    assert j.calls == 11 and not load_records(path)[first["item_id"]].get("error")


def test_summarize_prints_core_metrics(tmp_path, capsys):
    data, raw = tmp_path / "data", tmp_path / "raw"
    write_jsonl(data / "binary.jsonl", binary_items(40))
    write_jsonl(data / "rubric.jsonl", rubric_items(30))
    judges = [FakeJudge("jev-1.13.0", 0.7, 0.05, 1), FakeJudge("claude-sonnet-5-5:system1", 0.95, 0.5, 2)]
    for task, items in (("binary", binary_items(40)), ("rubric", rubric_items(30))):
        for j in judges:
            run_judge(j, task, items, workers=2, raw_dir=raw)
    summarize([j.name for j in judges], ["binary", "rubric"], data_dir=data, raw_dir=raw)
    out = capsys.readouterr().out
    for needle in ["Binary correctness", "length-only baseline AUROC", "jev-1.13.0", "ALL", "Rubric scoring",
                   "majority-class baseline", "helpfulness", "Latency, binary", "Latency, rubric",
                   "Paired differences, all binary items pooled: jev-1.13.0 minus", "Paired differences in rubric MAE"]:
        assert needle in out, needle


def test_every_call_is_dumped_to_csv_and_kept_out_of_the_jsonl(tmp_path):
    raw = tmp_path / "raw"
    run_judge(FakeJudge("fake", 1.0), "binary", binary_items(3), workers=2, raw_dir=raw)
    with (tmp_path / "calls.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 * 2  # 6 items x 2 logged calls
    assert {r["judge"] for r in rows} == {"fake"} and {r["task"] for r in rows} == {"binary"}
    assert rows[0]["raw_output"] == '{"verdict": "YES"}, "x"'  # commas / quotes survive the CSV round trip
    assert set(rows[0]) >= {"item_id", "criterion", "sample", "raw_output", "parsed", "latency_s", "error"}
    assert all("calls" not in r for r in load_records(raw_path("fake", "binary", raw)).values())

    run_judge(FakeJudge("fake", 1.0), "binary", binary_items(3), workers=2, raw_dir=raw)  # cached: no new rows
    with (tmp_path / "calls.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 12
