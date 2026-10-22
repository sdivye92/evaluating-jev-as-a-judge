"""Compute every number used in the article from the saved results (no API calls)."""

import collections
import csv
import os
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo root, one level above article/
sys.path.insert(0, f"{PROJECT}/src")

from jevjudge import metrics as M  # noqa: E402
from jevjudge.rubrics import CRITERIA  # noqa: E402
from jevjudge.runner import load_records, raw_path  # noqa: E402
from jevjudge.schema import load_binary, load_rubric  # noqa: E402

# display name -> judge name used in results/raw. Order = order in tables and charts.
JUDGES = {
    "Jev 1.13": "jev-1.13.0",
    "Haiku 5.5": "claude-haiku-5-5:system1",
    "Haiku 5.5 + thinking": "claude-haiku-5-5:reasoning",
    "Sonnet 5.5": "claude-sonnet-5-5:system1",
    "Sonnet 5.5 + thinking": "claude-sonnet-5-5:reasoning",
}
NAMES = list(JUDGES)
THINK = {"Haiku 5.5 + thinking": "Haiku 5.5", "Sonnet 5.5 + thinking": "Sonnet 5.5"}  # thinking arm -> its one-shot arm
SOURCES = ["gsm8k", "mbpp", "halueval", "ALL"]
SRC_LABEL = {"gsm8k": "GSM8K", "mbpp": "MBPP", "halueval": "HaluEval", "ALL": "All pooled"}
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}  # $ per 1M tokens (in, out)
N_BOOT = M.N_BOOT  # the same value run_eval uses, so the article's intervals match its printout


def compute():
    D = {}
    items = {i.id: i for i in load_binary()}
    rec = {n: load_records(raw_path(JUDGES[n], "binary")) for n in NAMES}

    def valid(n, src):
        return [k for k, i in items.items()
                if (src == "ALL" or i.source == src) and k in rec[n]
                and not rec[n][k].get("error") and rec[n][k].get("p_correct") is not None]

    D["n_items"] = {s: sum(1 for i in items.values() if s == "ALL" or i.source == s) for s in SOURCES}
    D["bin"] = {s: {} for s in SOURCES}
    for s in SOURCES:
        for n in NAMES:
            ids = valid(n, s)
            D["bin"][s][n] = M.binary_metrics([items[k].label for k in ids], [rec[n][k]["p_correct"] for k in ids],
                                              [items[k].group for k in ids], N_BOOT)

    def paired(s, a, b):
        ids = sorted(set(valid(a, s)) & set(valid(b, s)))
        d = M.paired_binary_diff([items[k].label for k in ids], [rec[a][k]["p_correct"] for k in ids],
                                 [rec[b][k]["p_correct"] for k in ids], [items[k].group for k in ids], N_BOOT)
        d["n"] = len(ids)
        return d

    D["paired"] = {s: {o: paired(s, NAMES[0], o) for o in NAMES[1:]} for s in SOURCES}  # Jev minus other
    D["think_gain"] = {s: {t: paired(s, t, o) for t, o in THINK.items()} for s in SOURCES}  # thinking minus one-shot

    # failed items (no verdict / score) per judge and task
    D["fail"] = {}
    for n in NAMES:
        for task in ("binary", "rubric"):
            recs = load_records(raw_path(JUDGES[n], task))
            kinds = collections.Counter()
            for r in recs.values():
                e = r.get("error") or ""
                if not e:
                    continue
                kinds["max_tokens" if "max_tokens" in e else "refusal" if "refusal" in e
                      else "credit" if "InvalidRequest" in e else "other"] += 1
            D["fail"][(n, task)] = (len(recs), sum(kinds.values()), dict(kinds))

    # reliability (pooled), only for judges that return graded probabilities
    D["rel"] = {}
    for n in NAMES:
        ids = valid(n, "ALL")
        y = np.array([items[k].label for k in ids])
        p = np.array([rec[n][k]["p_correct"] for k in ids])
        conf, ok = np.maximum(p, 1 - p), (p >= 0.5) == y
        if M.is_graded(p):
            pts = []
            for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
                m = (conf >= lo) & ((conf < lo + 0.1) if lo < 0.9 else (conf <= 1.0))
                if m.sum():
                    pts.append((float(conf[m].mean()), float(ok[m].mean()), int(m.sum())))
            D["rel"][n] = pts
        D["bin"]["ALL"][n]["frac_conf_ge_0.9"] = float((conf >= 0.9).mean())
        D["bin"]["ALL"][n]["acc_when_conf_ge_0.9"] = float(ok[conf >= 0.9].mean())

    # per-call log: single-call view of one-shot Claude, thinking effort, latency, tokens, spend
    rows = list(csv.DictReader(open(f"{PROJECT}/results/calls.csv")))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["judge"], r["task"])].append(r)
    D["single"], D["call_lat"], D["tokens"], D["think_stats"], D["spend"] = {}, {}, {}, {}, {}
    for n in NAMES:
        j = JUDGES[n]
        for task in ("binary", "rubric"):
            rs = [r for r in by[(j, task)] if not r["error"]]
            D["call_lat"][(n, task)] = float(np.median([float(r["latency_s"]) for r in rs]))
        if j.startswith("claude"):
            allrs = by[(j, "binary")] + by[(j, "rubric")]
            pi, po = PRICE[j.split(":")[0]]
            cost = lambda rs: sum(int(r["input_tokens"] or 0) * pi + int(r["output_tokens"] or 0) * po for r in rs) / 1e6
            D["spend"][n] = cost(allrs)
            for task in ("binary", "rubric"):
                rs = by[(j, task)]
                D.setdefault("cost_item", {})[(n, task)] = cost(rs) / len({r["item_id"] for r in rs})
                D.setdefault("cost_call", {})[(n, task)] = cost(rs) / len(rs)
        if j.endswith(":system1"):
            b = by[(j, "binary")]
            acc, far, frr = [], [], []
            for s in range(5):
                bs = [r for r in b if int(r["sample"]) == s]
                y = np.array([items[r["item_id"]].label for r in bs])
                v = np.array([int(float(r["parsed"])) for r in bs])
                acc.append((y == v).mean()); far.append(((y == 0) & (v == 1)).sum() / (y == 0).sum())
                frr.append(((y == 1) & (v == 0)).sum() / (y == 1).sum())
            votes = collections.defaultdict(list)
            for r in b:
                votes[r["item_id"]].append(int(float(r["parsed"])))
            D["single"][n] = {"acc": float(np.mean(acc)), "far": float(np.mean(far)), "frr": float(np.mean(frr)),
                              "unanimous": float(np.mean([len(set(v)) == 1 for v in votes.values()]))}
        if n in THINK:
            for task in ("binary", "rubric"):
                rs = by[(j, task)]
                groups = {"ALL": rs}
                if task == "binary":
                    for s in ("gsm8k", "mbpp", "halueval"):
                        groups[s] = [r for r in rs if items[r["item_id"]].source == s]
                for g, rr in groups.items():
                    ok = [r for r in rr if not r["error"]]
                    out = np.array([int(r["output_tokens"]) for r in ok])
                    D["think_stats"][(n, task, g)] = {
                        "calls": len(rr), "mean_out": float(out.mean()), "median_out": float(np.median(out)),
                        "p95_out": float(np.percentile(out, 95)), "max_out": int(out.max()),
                        "thinking_share": float(np.mean([bool(r["thinking"].strip()) for r in ok]))}
    # evidence from the earlier thinking run (effort "high", free-text reply), kept in results/old_reasoning_protocol/
    D["old_effort"] = {}
    old_path = f"{PROJECT}/results/old_reasoning_protocol/calls_reasoning.csv"
    if os.path.exists(old_path):
        old_rows = list(csv.DictReader(open(old_path)))
        for j, n in (("claude-haiku-5-5:reasoning", "Haiku 5.5"), ("claude-sonnet-5-5:reasoning", "Sonnet 5.5")):
            for task in ("binary", "rubric"):
                rs = [r for r in old_rows if r["judge"] == j and r["task"] == task and r["output_tokens"]]
                groups = {"ALL": rs}
                if task == "binary":
                    groups.update({g: [r for r in rs if items[r["item_id"]].source == g] for g in ("gsm8k", "mbpp", "halueval")})
                for g, rr_ in groups.items():
                    if rr_:  # a call with <= 8 output tokens contained no thinking, only the bare answer
                        D["old_effort"][(n, task, g)] = float(np.mean([int(r["output_tokens"]) <= 8 for r in rr_]))
    D["refusals_by_judge"] = {n: sum(1 for r in by[(JUDGES[n], "rubric")] + by[(JUDGES[n], "binary")]
                                     if r["error"].startswith("refusal")) for n in NAMES}

    # latency / tokens per judged item (from the per-item records)
    D["lat"], D["tok"] = {}, {}
    for task in ("binary", "rubric"):
        for n in NAMES:
            rs = [r for r in load_records(raw_path(JUDGES[n], task)).values() if not r.get("error")]
            D["lat"][(task, n)] = M.percentiles([r["latency_s"] for r in rs])
            D["tok"][(task, n)] = (float(np.mean([r["input_tokens"] or 0 for r in rs])),
                                   float(np.mean([r["n_calls"] for r in rs])), len(rs),
                                   float(np.mean([r["output_tokens"] or 0 for r in rs])))

    # rubric
    r_items = {i.id: i for i in load_rubric()}
    rr = {n: load_records(raw_path(JUDGES[n], "rubric")) for n in NAMES}
    D["rub"] = {c: {} for c in CRITERIA}
    for c in CRITERIA:
        for n in NAMES:
            pairs = [(r_items[k].labels[c], r["scores"][c]) for k, r in rr[n].items()
                     if k in r_items and r.get("scores", {}).get(c) is not None]
            y = np.array([p[0] for p in pairs]); s = np.array([p[1] for p in pairs])
            m = M.rubric_metrics(y, s)
            rnd = np.clip(np.rint(s), 0, 4).astype(int)
            m["mae_ci"] = M.cluster_bootstrap(len(y), None, lambda i, y=y, s=s: float(np.abs(s[i] - y[i]).mean()), N_BOOT)
            m["qwk_ci"] = M.cluster_bootstrap(
                len(y), None, lambda i, y=y, r=rnd: float(cohen_kappa_score(y[i], r[i], weights="quadratic", labels=list(range(5)))), N_BOOT)
            D["rub"][c][n] = m

    def rub_pair(a, b):
        ids = [k for k in r_items if all(rr[n].get(k, {}).get("scores", {}).get(c) is not None for n in (a, b) for c in CRITERIA)]
        err = lambda n, c: [abs(rr[n][k]["scores"][c] - r_items[k].labels[c]) for k in ids]
        row = {c: M.paired_mean_diff(err(a, c), err(b, c), N_BOOT) for c in CRITERIA}
        mean3 = lambda n: np.mean([err(n, c) for c in CRITERIA], axis=0)
        row["mean"] = M.paired_mean_diff(mean3(a), mean3(b), N_BOOT)
        return len(ids), row

    D["rub_paired"] = {o: rub_pair(NAMES[0], o) for o in NAMES[1:]}
    D["rub_think_gain"] = {t: rub_pair(t, o) for t, o in THINK.items()}
    return D


if __name__ == "__main__":
    D = compute()
    a = D["bin"]["ALL"]
    print({n: round(a[n]["accuracy"], 4) for n in NAMES})
    print(D["fail"])
    print({n: round(v, 2) for n, v in D["spend"].items()})
