"""Print the evaluation metrics for judges whose results are cached in results/raw/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import metrics as M
from .rubrics import CRITERIA
from .runner import RAW_DIR, load_records, raw_path
from .schema import DATA_DIR, load_binary, load_rubric


def summarize(
    judge_names: list[str],
    tasks: list[str],
    limit: int | None = None,
    data_dir: Path = DATA_DIR,
    raw_dir: Path = RAW_DIR,
) -> None:
    for task in tasks:
        items = (load_binary if task == "binary" else load_rubric)(data_dir)[:limit]
        if not items:
            print(f"[skip] no {task} data in {data_dir}")
            continue
        records = {n: load_records(raw_path(n, task, raw_dir)) for n in judge_names}
        (print_binary if task == "binary" else print_rubric)(items, records)
        (print_binary_diffs if task == "binary" else print_rubric_diffs)(items, records)
        print_latency(task, records)


def _fmt(x: float | None, spec: str) -> str:
    return f"{'–':>6}" if x is None or np.isnan(x) else format(x, spec)


def print_binary(items, records) -> None:
    print("\n== Binary correctness ==")
    print("false-accept = incorrect marked correct; false-reject = correct marked incorrect")
    print("AUROC / ECE are shown only for judges that return graded probabilities")
    print(f"{'source':9} {'judge':28} {'n':>5} {'err':>4} {'accuracy [95% CI]':>24} {'FAR':>6} {'FRR':>6} "
          f"{'MCC':>6} {'AUROC':>6} {'ECE':>6}")
    sources = sorted({i.source for i in items})
    for source in [*sources, "ALL"]:
        sl = [i for i in items if source in ("ALL", i.source)]
        y_all = [i.label for i in sl]
        if len(set(y_all)) == 2:  # a judge that only reads form would score about this well
            auc = roc_auc_score(y_all, [-len(i.candidate) for i in sl])
            print(f"{source:9} {'(length-only baseline AUROC)':28} {len(sl):5} {'':4} {auc:24.2f}")
        for name, recs in records.items():
            ok = [(i, recs[i.id]["p_correct"]) for i in sl
                  if i.id in recs and not recs[i.id].get("error") and recs[i.id].get("p_correct") is not None]
            if not ok:
                continue
            m = M.binary_metrics([i.label for i, _ in ok], [p for _, p in ok], [i.group for i, _ in ok])
            lo, hi = m["accuracy_ci"]
            print(f"{source:9} {name:28} {m['n']:5} {len(sl) - m['n']:4} "
                  f"{100 * m['accuracy']:8.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] "
                  f"{100 * m['far']:5.1f}% {100 * m['frr']:5.1f}% {m['mcc']:6.2f} "
                  f"{_fmt(m.get('auroc'), '6.2f')} {_fmt(m.get('ece'), '6.3f')}   (majority {100 * m['majority_acc']:.0f}%)")


def _fmt_diff(point: float, ci: tuple[float, float], scale: float = 100.0, unit: str = " pts", digits: int = 1) -> str:
    flag = "*" if not (ci[0] <= 0 <= ci[1]) else " "  # * = the CI excludes 0
    return f"{scale * point:+.{digits}f}{unit} [{scale * ci[0]:+.{digits}f}, {scale * ci[1]:+.{digits}f}]{flag}"


def print_binary_diffs(items, records) -> None:
    names = list(records)
    if len(names) < 2:
        return
    base = names[0]
    print(f"\n== Paired differences, all binary items pooled: {base} minus each other judge ==")
    print("95% CI from a bootstrap that resamples whole questions and recomputes both judges on the same resample.")
    print("Accuracy: negative = base is worse. False-accept / false-reject: positive = base is worse.  * = CI excludes 0")
    print(f"{'vs':28} {'n':>5} {'accuracy':>26} {'false-accept':>26} {'false-reject':>26}")
    ok = lambda r: r is not None and not r.get("error") and r.get("p_correct") is not None
    for other in names[1:]:
        both = [i for i in items if ok(records[base].get(i.id)) and ok(records[other].get(i.id))]
        if not both:
            continue
        d = M.paired_binary_diff([i.label for i in both],
                                 [records[base][i.id]["p_correct"] for i in both],
                                 [records[other][i.id]["p_correct"] for i in both],
                                 [i.group for i in both])
        print(f"{other:28} {len(both):5} " + " ".join(f"{_fmt_diff(*d[k]):>26}" for k in ("accuracy", "far", "frr")))


def print_rubric_diffs(items, records) -> None:
    names = list(records)
    if len(names) < 2:
        return
    base = names[0]
    print(f"\n== Paired differences in rubric MAE: {base} minus each other judge ==")
    print("Per-item absolute error on the expected score; positive = base is worse.  * = 95% bootstrap CI excludes 0")
    print(f"{'vs':28} {'n':>5} " + " ".join(f"{c:>26}" for c in [*CRITERIA, "mean of 3 criteria"]))
    by_id = {i.id: i for i in items}
    has_all = lambda r: r is not None and all(r.get("scores", {}).get(c) is not None for c in CRITERIA)
    for other in names[1:]:
        ids = [k for k in by_id if has_all(records[base].get(k)) and has_all(records[other].get(k))]
        if not ids:
            continue
        err = lambda name, c: [abs(records[name][k]["scores"][c] - by_id[k].labels[c]) for k in ids]
        cells = [_fmt_diff(*M.paired_mean_diff(err(base, c), err(other, c)), scale=1.0, unit="", digits=2) for c in CRITERIA]
        mean_err = lambda name: np.mean([err(name, c) for c in CRITERIA], axis=0)
        cells.append(_fmt_diff(*M.paired_mean_diff(mean_err(base), mean_err(other)), scale=1.0, unit="", digits=2))
        print(f"{other:28} {len(ids):5} " + " ".join(f"{c:>26}" for c in cells))


def print_rubric(items, records) -> None:
    print("\n== Rubric scoring (HelpSteer2 human ratings 0-4) ==")
    print(f"{'criterion':12} {'judge':28} {'n':>5} {'exact':>7} {'MAE':>6} {'QWK':>6} {'Spearman':>9}  predicted-level histogram")
    by_id = {i.id: i for i in items}
    for c in CRITERIA:
        printed_baseline = False
        for name, recs in records.items():
            pairs = [(by_id[k].labels[c], r["scores"][c]) for k, r in recs.items()
                     if k in by_id and r.get("scores", {}).get(c) is not None]
            if not pairs:
                continue
            y, s = zip(*pairs)
            m = M.rubric_metrics(y, s)
            if not printed_baseline:
                print(f"{c:12} {'majority-class baseline':28} {m['n']:5} {100 * m['majority_exact']:6.1f}% {m['majority_mae']:6.2f}")
                printed_baseline = True
            print(f"{c:12} {name:28} {m['n']:5} {100 * m['exact']:6.1f}% {m['mae']:6.2f} {m['qwk']:6.2f} "
                  f"{m['spearman']:9.2f}  {m['pred_hist']} (gold {m['gold_hist']})")


def print_latency(task: str, records) -> None:
    print(f"\n-- Latency, {task} (ms per judged item, as seen by the client; includes network and concurrency) --")
    print(f"{'judge':28} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'calls':>6} {'tokens in/out':>14} {'errors':>7}")
    for name, recs in records.items():
        ok = [r for r in recs.values() if not r.get("error")]
        if not ok:
            continue
        lat = M.percentiles([r["latency_s"] * 1000 for r in ok])
        print(f"{name:28} {lat['mean']:7.0f} {lat['p50']:7.0f} {lat['p95']:7.0f} {lat['p99']:7.0f} "
              f"{np.mean([r['n_calls'] for r in ok]):6.1f} "
              f"{np.mean([r.get('input_tokens') or 0 for r in ok]):7.0f}/{np.mean([r.get('output_tokens') or 0 for r in ok]):<6.0f} "
              f"{len(recs) - len(ok):7}")
