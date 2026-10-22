import numpy as np
import pytest

from jevjudge import metrics as M


def test_binary_metrics_known_confusion():
    # 4 correct, 4 incorrect. Judge accepts 3 of 4 correct and 1 of 4 incorrect.
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0]
    m = M.binary_metrics(y, p, n_boot=50)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["far"] == pytest.approx(0.25) and m["frr"] == pytest.approx(0.25)
    assert m["mcc"] == pytest.approx(0.5)
    assert m["majority_acc"] == 0.5
    assert "ece" not in m and "auroc" not in m  # 0/1 votes: calibration not reported


def test_verdict_uses_the_judges_own_probability_at_half():
    m = M.binary_metrics([1, 1, 0, 0], [0.6, 0.51, 0.49, 0.4], n_boot=10)  # 60% yes counts as yes
    assert m["accuracy"] == 1.0 and "ece" in m and m["auroc"] == 1.0


def test_chance_level_judge_has_mcc_zero():
    m = M.binary_metrics([1, 0] * 50, [1] * 100, n_boot=20)  # accepts everything
    assert m["far"] == 1.0 and m["frr"] == 0.0 and m["mcc"] == 0.0 and m["accuracy"] == 0.5


def test_calibration_perfect_vs_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 5000)
    y_cal = (rng.uniform(size=5000) < p).astype(int)  # truth follows the stated probability
    assert M.ece(y_cal, p) < 0.03
    y_bad = (rng.uniform(size=5000) < 0.5).astype(int)  # confident but coin-flip
    assert M.ece(y_bad, np.where(rng.uniform(size=5000) < 0.5, 0.95, 0.05)) > 0.3


def test_cluster_bootstrap_widens_with_correlated_groups():
    # Each question contributes two perfectly correlated items; clustering must not shrink the CI.
    rng = np.random.default_rng(1)
    y = np.repeat(rng.integers(0, 2, 100), 2)
    pred = np.repeat(rng.integers(0, 2, 100), 2)
    groups = [f"q{i // 2}" for i in range(200)]
    f = lambda idx: float((y[idx] == pred[idx]).mean())
    lo_c, hi_c = M.cluster_bootstrap(200, groups, f, 300)
    lo_i, hi_i = M.cluster_bootstrap(200, None, f, 300)
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_rubric_metrics_perfect_and_collapsed():
    y = [0, 1, 2, 3, 4, 4, 3, 2]
    perfect = M.rubric_metrics(y, [float(v) for v in y])
    assert perfect["exact"] == 1.0 and perfect["mae"] == 0 and perfect["qwk"] == pytest.approx(1.0)
    # always saying 4 on a skewed set: decent-looking exact match, no agreement beyond chance
    collapsed = M.rubric_metrics([4, 4, 4, 3, 4, 4, 2, 4], [4.0] * 8)
    assert collapsed["exact"] == pytest.approx(collapsed["majority_exact"])
    assert collapsed["qwk"] == pytest.approx(0.0)
    assert collapsed["pred_hist"] == [0, 0, 0, 0, 8]


def test_rubric_mae_uses_the_expected_score_not_the_rounded_one():
    assert M.rubric_metrics([2, 3, 2, 3], [2.4, 2.6, 2.4, 2.6])["mae"] == pytest.approx(0.4)


def test_percentiles():
    p = M.percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["p95"] == pytest.approx(95.05)


def test_paired_binary_diff_detects_a_real_gap_and_zero_for_identical_judges():
    rng = np.random.default_rng(3)
    n_q = 200
    y = np.tile([1, 0], n_q)
    groups = [f"q{i // 2}" for i in range(2 * n_q)]
    perfect = y.astype(float)
    noisy = np.where(rng.uniform(size=2 * n_q) < 0.7, y, 1 - y).astype(float)  # right ~70% of the time

    d = M.paired_binary_diff(y, perfect, noisy, groups, n_boot=300)
    point, (lo, hi) = d["accuracy"]
    assert point > 0.2 and lo > 0  # perfect judge is better; the CI excludes 0
    assert d["far"][0] <= 0 and d["frr"][0] <= 0  # and makes fewer false accepts / rejects

    same = M.paired_binary_diff(y, noisy, noisy, groups, n_boot=100)
    assert same["accuracy"] == (0.0, (0.0, 0.0))


def test_paired_mean_diff():
    point, (lo, hi) = M.paired_mean_diff([1.0] * 50, [0.5] * 50, n_boot=100)
    assert point == pytest.approx(0.5) and lo == pytest.approx(0.5) == hi
