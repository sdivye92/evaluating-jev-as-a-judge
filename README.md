# Jev as a judge

Pointwise evaluation: can Jev (TypeSafe, pinned `jev-1.13.0`) replace an LLM judge? Compared against
Claude Haiku 5.5 (smaller) and Claude Sonnet 5.5 (larger). All calls go through LangChain
(`langchain-typesafe`, `langchain-anthropic`).

## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env        # fill in TYPESAFE_API_KEY and ANTHROPIC_API_KEY

python run_eval.py          # edit the constants at the top (judges, samples, LIMIT for a smoke test)
# or
jevjudge build-data                                   # or e.g. --gsm8k 1000 --rubric 1038
jevjudge run --judges jev haiku sonnet --limit 20     # drop --limit for the full run
```

Results are cached per item in `results/raw/`; rerunning resumes and only retries errored items.
`results/calls.csv` has one row per API call (judge, task, item, criterion, sample, raw output, parsed value, stop
reason, tokens, latency, error), so any verdict can be traced and re-analysed without calling the APIs again.
`pytest` runs the offline tests (no API keys needed).

## Settings

Set in `run_eval.py` (constants) or as `jevjudge run` flags. Explained in full in the comments of `run_eval.py`.

| `run_eval.py` | CLI flag | Meaning |
|---|---|---|
| `JUDGES` | `--judges` | Judges to run; the first is the baseline in the paired-difference tables |
| `BINARY_SAMPLES` | `--binary-samples` | Claude only: asks per binary item. 1 gives a 0/1 vote; 3 or 5 gives a graded probability (share of YES) |
| `RUBRIC_SAMPLES` | `--rubric-samples` | Claude only: asks per rubric criterion, averaged into an expected score (15 calls per item at 5) |
| `WORKERS` | `--workers` | Items judged concurrently; more is faster but can hit rate limits and inflates measured latency |
| `LIMIT` | `--limit` | Only the first N items per task, for a cheap smoke test; `None` / unset runs everything |
| `JEV_MODEL` | `--jev-model` | Jev version, pinned to `jev-1.13.0` |

## What is judged

| Task | Data | Label source |
|---|---|---|
| Binary correctness | GSM8K (answer only, near-miss wrong numbers) | by construction |
| | MBPP (code; single-edit mutants) | test execution |
| | HaluEval QA (answer vs. knowledge) | dataset, length-matched pairs |
| | `data/private.jsonl` (optional) | yours |
| Rubric 0-4 | HelpSteer2: helpfulness, correctness, coherence | human ratings |

Private set format, one JSON per line: `id`, `question`, `candidate`, `label` (1 correct / 0 incorrect),
optional `knowledge`, `group`.

## Judging setup

- Same information to every judge. Jev gets typed questions (`Noul` for binary, one `Score` per criterion);
  Claude gets the same wording and the same JSON state.
- Claude "system1" arm: thinking off, effort `low`, and the reply is forced to `{"verdict": "YES"|"NO"}` (binary) or
  `{"score": "0".."4"}` (rubric) with structured outputs. Asking for "one word" in the prompt is not enough: on GSM8K and
  MBPP the models start solving the problem in the visible reply and hit the token cap with no verdict.
- Claude "reasoning" arm (`haiku-reasoning`, `sonnet-reasoning`): the same prompt and the same structured reply, with
  adaptive thinking at effort `max`, so the two arms differ only in whether the model thinks. Thinking is adaptive (the
  model decides how much) and the 5.5 models reject a fixed budget, so effort is the only dial: in a small probe Sonnet
  thought on 2 of 8 binary items at `high`, 5 at `xhigh` and all 8 at `max`. The thinking summary of every call is logged
  to `calls.csv`. The arm does not run unless you add it to `JUDGES`.
- Claude has no logprobs and rejects non-default temperature on the 5.5 models, so a graded score is the mean of
  `--rubric-samples` samples (default 5). Binary is one vote per item unless `--binary-samples` is set to 3 or 5.
- Server-side refusal fallbacks are off, so a refusal is an error and never answered by a different model.
- Every judge's output is used as returned: a binary verdict is `P(correct) >= 0.5`, and a rubric score is the expected
  value of the level distribution. Nothing is tuned.
- Printed metrics: accuracy with a question-clustered 95% CI, false-accept / false-reject rate, MCC, AUROC and ECE
  (graded probabilities only), majority baseline, and a length-only AUROC as a shortcut check; rubric exact / MAE / QWK /
  Spearman against a majority baseline; latency and tokens.

## Caveats

- Not yet run against the live APIs: no keys were available when this was written.
- HaluEval's hallucinated answers are mostly full sentences against short right answers (length alone gives AUROC 0.97),
  so only length-matched pairs (0.8-1.25x) are used, which caps the pool at about 380 questions.
- Resolving a 5-point gap needs roughly 1000 items per slice; the defaults are a pilot size. GSM8K (1,319 questions) and
  the rubric task (1,038 items) can go to full size; HaluEval and MBPP are capped at a few hundred questions.
- Latency is client-side wall clock under `--workers` concurrency.
