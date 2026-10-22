"""Metrics for pointwise binary and rubric judging. Pure numpy/scipy, no API access."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score


# Bootstrap resamples behind every 95% interval (per-judge and paired). One value everywhere, so all
# judges and all comparisons are computed the same way.
N_BOOT = 1000


def percentiles(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def cluster_bootstrap(
    n: int,
    groups: Sequence[str] | None,
    stat: Callable[[np.ndarray], float],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI of ``stat(indices)``, resampling whole groups (questions) with replacement."""
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if groups is None:
        idx_by_group = [np.array([i]) for i in range(n)]
    else:
        buckets: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            buckets.setdefault(g, []).append(i)
        idx_by_group = [np.array(v) for v in buckets.values()]
    k = len(idx_by_group)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, size=k)
        v = stat(np.concatenate([idx_by_group[j] for j in pick]))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------- binary


def _confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp, tn, fp, fn


def _rate(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d else 0.0


def _stat(name: str, y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> float:
    tp, tn, fp, fn = _confusion(y[idx], pred[idx])
    if name == "accuracy":
        return _rate(tp + tn, len(idx))
    if name == "far":  # false-accept rate: incorrect answers marked correct
        return _rate(fp, fp + tn)
    if name == "frr":  # false-reject rate: correct answers marked incorrect
        return _rate(fn, fn + tp)
    if name == "mcc":
        return _mcc(tp, tn, fp, fn)
    raise KeyError(name)


def is_graded(p: np.ndarray) -> bool:
    """False when probabilities are just 0/1 votes (calibration is then meaningless)."""
    return len(np.unique(np.round(p, 6))) > 2


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the verdict (p >= 0.5) against its confidence max(p, 1-p)."""
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def binary_metrics(
    y: Sequence[int],
    p: Sequence[float],
    groups: Sequence[str] | None = None,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Verdict is the judge's own: correct when P(correct) >= 0.5."""
    y_a, p_a = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n = len(y_a)
    if n == 0:
        return {"n": 0}
    pred = (p_a >= 0.5).astype(int)
    tp, tn, fp, fn = _confusion(y_a, pred)
    out: dict[str, Any] = {
        "n": n,
        "majority_acc": float(max(y_a.mean(), 1 - y_a.mean())),
        "accuracy": _rate(tp + tn, n),
        "far": _rate(fp, fp + tn),
        "frr": _rate(fn, fn + tp),
        "mcc": _mcc(tp, tn, fp, fn),
    }
    for name in ("accuracy", "far", "frr", "mcc"):
        out[f"{name}_ci"] = cluster_bootstrap(n, groups, lambda idx, nm=name: _stat(nm, y_a, pred, idx), n_boot)
    if is_graded(p_a):  # AUROC and ECE only mean something for graded probabilities
        out["ece"] = ece(y_a, p_a)
        if len(np.unique(y_a)) == 2:
            out["auroc"] = float(roc_auc_score(y_a, p_a))
    return out


def paired_binary_diff(
    y: Sequence[int],
    p_base: Sequence[float],
    p_other: Sequence[float],
    groups: Sequence[str],
    n_boot: int = N_BOOT,
) -> dict[str, tuple[float, tuple[float, float]]]:
    """base - other for accuracy / false-accept / false-reject on the same items.

    Each bootstrap resample draws whole questions, then recomputes both judges' metric on that
    same resample, so the pairing and the question-level dependence are both preserved.
    """
    y_a = np.asarray(y, dtype=int)
    pb, po = ((np.asarray(p, dtype=float) >= 0.5).astype(int) for p in (p_base, p_other))
    everything = np.arange(len(y_a))
    out = {}
    for name in ("accuracy", "far", "frr"):
        diff = lambda idx, nm=name: _stat(nm, y_a, pb, idx) - _stat(nm, y_a, po, idx)
        out[name] = (diff(everything), cluster_bootstrap(len(y_a), groups, diff, n_boot))
    return out


def paired_mean_diff(err_base: Sequence[float], err_other: Sequence[float], n_boot: int = N_BOOT):
    """Mean of per-item (base error - other error) with a bootstrap 95% CI over items."""
    d = np.asarray(err_base, dtype=float) - np.asarray(err_other, dtype=float)
    return float(d.mean()), cluster_bootstrap(len(d), None, lambda idx: float(d[idx].mean()), n_boot)


# ----------------------------------------------------------------------------- rubric


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def rubric_metrics(y: Sequence[int], score: Sequence[float], levels: int = 5) -> dict[str, Any]:
    """``score`` is the judge's expected score in [0, levels-1]; rounded for exact match and QWK."""
    y_a, s_a = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    n = len(y_a)
    if n == 0:
        return {"n": 0}
    r = np.clip(np.rint(s_a), 0, levels - 1).astype(int)
    mode = int(np.bincount(y_a, minlength=levels).argmax())
    both_constant = len(np.unique(y_a)) < 2 and len(np.unique(r)) < 2
    qwk = float("nan") if both_constant else float(
        cohen_kappa_score(y_a, r, weights="quadratic", labels=list(range(levels)))
    )
    return {
        "n": n,
        "exact": float((r == y_a).mean()),
        "mae": float(np.abs(s_a - y_a).mean()),
        "qwk": qwk,
        "spearman": _spearman(s_a, y_a),
        "majority_exact": float((y_a == mode).mean()),
        "majority_mae": float(np.abs(mode - y_a).mean()),
        "pred_hist": np.bincount(r, minlength=levels).tolist(),
        "gold_hist": np.bincount(y_a, minlength=levels).tolist(),
    }
