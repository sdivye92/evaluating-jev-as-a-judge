"""Run the evaluation without the CLI: edit the constants below, then `python run_eval.py`.

Results are cached per item in results/raw/, so an interrupted run resumes without
re-calling the APIs. Needs TYPESAFE_API_KEY and ANTHROPIC_API_KEY (see .env.example).
The datasets are built on first use (same as `jevjudge build-data`).
"""

from dotenv import load_dotenv

from jevjudge.data import build_all
from jevjudge.evaluate import summarize
from jevjudge.judges import DEFAULT_JEV_MODEL, make_judge
from jevjudge.rubrics import CRITERIA
from jevjudge.runner import run_judge
from jevjudge.schema import DATA_DIR, load_binary, load_rubric

# Which judges to run. The FIRST one is the baseline in the "paired differences" tables, so keep
# "jev" first. "haiku" / "sonnet" are the one-shot (System 1) arms: thinking off, structured YES/NO or
# digit reply. The thinking arms ("haiku-reasoning", "sonnet-reasoning") use the same prompt and the same
# structured reply, but adaptive thinking at effort "max" (so the model really thinks; at lower effort
# Sonnet skips thinking on many items). Reasoning stays in hidden thinking blocks. Their results exist only
# for the first LIMIT items below; the other three judges are cached for all items.
# Every API call is logged to results/calls.csv with its raw output and, for the thinking arms, the
# thinking summary.
JUDGES = ["jev", "haiku", "sonnet", "haiku-reasoning", "sonnet-reasoning"]

TASKS = ["binary", "rubric"]

# Jev model version. Pinned (not "jev-latest") so a TypeSafe release cannot change results mid-run.
JEV_MODEL = DEFAULT_JEV_MODEL

# Claude-only setting. How many times Claude is asked each binary (correct / incorrect) question.
# Claude answers YES or NO and cannot give a probability, so its "probability of correct" is the
# share of YES answers:
#   1 -> a single vote, so the probability is exactly 0 or 1 (AUROC and ECE are not reported for Claude)
#   3 or 5 -> a graded probability such as 0.6, which lets Claude have AUROC and ECE like Jev
# Use an odd number to avoid ties. Cost and calls scale linearly with it. Jev is unaffected: it returns
# its own probability in one call.
BINARY_SAMPLES = 1

# Claude-only setting. How many times Claude scores each rubric criterion (a 0-4 digit). The average is
# Claude's expected score, the counterpart of the expected score Jev computes from its level probabilities.
# One rubric item costs Claude 3 criteria x RUBRIC_SAMPLES calls (15 by default). Claude has no logprobs
# and the 5.5 models reject non-default temperature, so averaging samples is the only way to get a
# continuous score.
RUBRIC_SAMPLES = 1

# How many items are judged at the same time (threads). Higher finishes sooner, but can hit API rate
# limits and inflates the measured latency because requests queue. For Claude rubric scoring this means
# up to WORKERS x RUBRIC_SAMPLES x 3 requests in flight at once (60 by default).
WORKERS = 4

# None runs every item. A number runs only the first N items of each task. Currently 20: the reasoning arms
# were run on these 20 items only (10 GSM8K questions, 1 sample each, to stay within budget), so every judge
# is reported on the same 20 items. Results are cached in results/raw/, so a later full run (None) reuses them.
LIMIT = None


def main() -> None:
    load_dotenv()
    if not (DATA_DIR / "binary.jsonl").exists():
        build_all()
    fanout = max(8, WORKERS * max(BINARY_SAMPLES, RUBRIC_SAMPLES) * len(CRITERIA))
    names = []
    for key in JUDGES:
        judge = make_judge(key, JEV_MODEL, BINARY_SAMPLES, RUBRIC_SAMPLES, fanout)
        names.append(judge.name)
        for task in TASKS:
            items = (load_binary if task == "binary" else load_rubric)()[:LIMIT]
            run_judge(judge, task, items, workers=WORKERS)
    summarize(names, TASKS, LIMIT)


if __name__ == "__main__":
    main()
