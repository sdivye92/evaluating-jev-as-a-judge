"""Build jev_as_a_judge.md, jev_as_a_judge.html and figures/*.svg from the saved results.

Reads ../results/raw and ../results/calls.csv (no API calls), fills article_template.md with the numbers,
and writes everything next to this file. Needs: pip install markdown pymdown-extensions
Run from anywhere:  python article/build_article.py
"""

import os
import re

import markdown

import article_svg as S
from article_data import CRITERIA, JUDGES, N_BOOT, NAMES, SRC_LABEL, THINK, compute

OUT = os.path.dirname(os.path.abspath(__file__))
D = compute()
KEY = {"Jev 1.13": "jev", "Haiku 5.5": "hai", "Haiku 5.5 + thinking": "hait", "Sonnet 5.5": "son", "Sonnet 5.5 + thinking": "sont"}
JEV, HAI, HAIT, SON, SONT = NAMES
A = D["bin"]["ALL"]
CRIT_LABEL = {"helpfulness": "Helpfulness", "correctness": "Correctness", "coherence": "Coherence"}
SRCS = ("gsm8k", "mbpp", "halueval", "ALL")
MINUS = S.MINUS


# ------------------------------------------------------------------------ formatting
def P(x, d=1):
    return f"{100 * x:.{d}f}%"


def PP(x, d=1):  # signed percentage points
    if round(100 * x, d) == 0:
        return f"{0:.{d}f}"
    return f"{100 * x:+.{d}f}".replace("-", MINUS)


def PA(x, d=1):
    return f"{abs(100 * x):.{d}f}"


def CI(c, d=1):
    return f"[{100 * c[0]:.{d}f}, {100 * c[1]:.{d}f}]"


def CIP(c, d=1):
    return f"[{PP(c[0], d)}, {PP(c[1], d)}]"


def N2(x, d=2):
    return f"{x:.{d}f}".replace("-", MINUS)


def SG(x, d=2):
    if round(x, d) == 0:
        return f"{0:.{d}f}"
    return f"{x:+.{d}f}".replace("-", MINUS)


def SEC(x, d=2):
    return f"{x:.{d}f} s"


def opt(m, key, d=2):
    return N2(m[key], d) if key in m else "–"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------- figures
FIG = {}
fmt1 = lambda v: f"{v:.1f}%"
tick0 = lambda v: f"{v:.0f}%"
cats = [SRC_LABEL[s] for s in SRCS]
FIG["f1_accuracy_by_source"] = S.grouped_bars(
    cats, NAMES, [[D["bin"][s][n]["accuracy"] * 100 for n in NAMES] for s in SRCS],
    [[tuple(100 * v for v in D["bin"][s][n]["accuracy_ci"]) for n in NAMES] for s in SRCS],
    100, 20, fmt1,
    "Grouped columns: binary accuracy of the five judges on GSM8K, MBPP, HaluEval and all items pooled, with 95% confidence intervals.",
    "Accuracy", cat_sub=[f"{D['n_items'][s]} items" for s in SRCS], ref_lines=[(50, "chance (50%)")], tick_fmt=tick0)

FIG["f2_error_profile"] = S.grouped_bars(
    ["False-accept rate", "False-reject rate"], NAMES,
    [[A[n]["far"] * 100 for n in NAMES], [A[n]["frr"] * 100 for n in NAMES]],
    [[tuple(100 * v for v in A[n]["far_ci"]) for n in NAMES], [tuple(100 * v for v in A[n]["frr_ci"]) for n in NAMES]],
    40, 10, fmt1,
    "Grouped columns: pooled false-accept rate (wrong answers marked correct) and false-reject rate (correct answers marked incorrect) for the five judges, with 95% confidence intervals.",
    "Share of the relevant answers", cat_sub=["of the 550 incorrect answers, share accepted", "of the 550 correct answers, share rejected"],
    tick_fmt=tick0)

pj = D["paired"]["ALL"]
others = NAMES[1:]
flab = lambda o: f"vs {o}"
fgroups = [("Accuracy", [(flab(o), o, 100 * pj[o]["accuracy"][0], *[100 * v for v in pj[o]["accuracy"][1]]) for o in others])]
for key, title in (("far", "False-accepts (fewer is better)"), ("frr", "False-rejects (fewer is better)")):
    rows = []
    for o in others:
        v, (lo, hi) = pj[o][key]
        rows.append((flab(o), o, -100 * v, -100 * hi, -100 * lo))  # flip: positive = Jev better
    fgroups.append((title, rows))
FIG["f3_paired_vs_jev"] = S.forest(
    fgroups, -20, 30, 10, "Jev advantage over the other judge, in percentage points (pooled binary items)",
    "◀ Jev worse", "Jev better ▶",
    "Forest plot of paired differences between Jev and each of the four Claude judges for accuracy, false-accept rate and false-reject rate, with 95% confidence intervals; positive values favour Jev.")

tg = D["think_gain"]
ggroups = []
for s in SRCS:
    ggroups.append((SRC_LABEL[s], [(m, t, 100 * tg[s][t]["accuracy"][0], *[100 * v for v in tg[s][t]["accuracy"][1]])
                                   for t, m in THINK.items()]))
FIG["f4_thinking_gain"] = S.forest(
    ggroups, 0, 40, 10, "Accuracy gain from thinking, in percentage points (thinking arm minus the same model one-shot)",
    "no gain", "gain ▶", "Forest plot of the accuracy gain from enabling thinking for Haiku and Sonnet on each data source, with 95% confidence intervals.")

FIG["f5_reliability"] = S.reliability(
    {n: D["rel"][n] for n in NAMES if n in D["rel"]},
    "Reliability diagram: stated confidence against actual accuracy for the judges that return graded probabilities, pooled over all binary items.")

FIG["f6_rubric_mae"] = S.grouped_bars(
    [CRIT_LABEL[c] for c in CRITERIA], NAMES, [[D["rub"][c][n]["mae"] for n in NAMES] for c in CRITERIA],
    [[D["rub"][c][n]["mae_ci"] for n in NAMES] for c in CRITERIA], 1.4, 0.2, lambda v: f"{v:.2f}",
    "Grouped columns: mean absolute error of each judge's expected score against human ratings for helpfulness, correctness and coherence, with a marker for the error of always predicting the most common human score.",
    "Mean absolute error (0–4 scale; lower is better)", group_marks=[D["rub"][c][JEV]["majority_mae"] for c in CRITERIA],
    mark_label="always predict the most common human score")
FIG["f7_rubric_qwk"] = S.grouped_bars(
    [CRIT_LABEL[c] for c in CRITERIA], NAMES, [[D["rub"][c][n]["qwk"] for n in NAMES] for c in CRITERIA],
    [[D["rub"][c][n]["qwk_ci"] for n in NAMES] for c in CRITERIA], 0.6, 0.1, lambda v: f"{v:.2f}",
    "Grouped columns: quadratic weighted kappa between each judge's rounded score and the human rating for each criterion, with 95% confidence intervals.",
    "Quadratic weighted kappa (higher is better)")

lat = D["lat"]
FIG["f8_latency"] = S.hbars(
    ["Binary verdict", "Rubric score"], NAMES,
    [[lat[(t, n)]["p50"] for n in NAMES] for t in ("binary", "rubric")],
    [[lat[(t, n)]["p95"] for n in NAMES] for t in ("binary", "rubric")], 45, 5,
    "Horizontal bars: median and 95th-percentile latency per judged item for the binary and rubric tasks, five judges.", dp=2)

os.makedirs(f"{OUT}/figures", exist_ok=True)
for f in os.listdir(f"{OUT}/figures"):
    if f.endswith(".svg"):
        os.remove(f"{OUT}/figures/{f}")  # drop figures from an earlier layout
for name, svg in FIG.items():
    open(f"{OUT}/figures/{name}.svg", "w").write(svg)

# ------------------------------------------------------------------------------ tables
T = {}
T["headline"] = table(
    ["Judge", "Items judged", "Accuracy [95% CI]", "False-accept [CI]", "False-reject [CI]", "MCC", "AUROC", "ECE"],
    [[n, A[n]["n"], f"{P(A[n]['accuracy'])} {CI(A[n]['accuracy_ci'])}", f"{P(A[n]['far'])} {CI(A[n]['far_ci'])}",
      f"{P(A[n]['frr'])} {CI(A[n]['frr_ci'])}", N2(A[n]["mcc"]), opt(A[n], "auroc"), opt(A[n], "ece", 3)] for n in NAMES])

rows = []
for s in ("gsm8k", "mbpp", "halueval"):
    for n in NAMES:
        m = D["bin"][s][n]
        rows.append([SRC_LABEL[s] if n == JEV else "", n, m["n"], f"{P(m['accuracy'])} {CI(m['accuracy_ci'])}", P(m["far"]),
                     P(m["frr"]), N2(m["mcc"])])
T["by_source"] = table(["Source", "Judge", "Items judged", "Accuracy [95% CI]", "False-accept", "False-reject", "MCC"], rows)

rows = []
for s in ("ALL", "gsm8k", "mbpp", "halueval"):
    for o in others:
        d = D["paired"][s][o]
        rows.append([SRC_LABEL[s] if o == others[0] else "", f"Jev − {o}", d["n"], f"{PP(d['accuracy'][0])} {CIP(d['accuracy'][1])}",
                     f"{PP(d['far'][0])} {CIP(d['far'][1])}", f"{PP(d['frr'][0])} {CIP(d['frr'][1])}"])
T["paired"] = table(["Slice", "Comparison", "Items", "Δ accuracy (pts)", "Δ false-accept (pts)", "Δ false-reject (pts)"], rows)

rows = []
for s in ("ALL", "gsm8k", "mbpp", "halueval"):
    for t, m in THINK.items():
        d = tg[s][t]
        rows.append([SRC_LABEL[s] if m == "Haiku 5.5" else "", f"{m}: thinking − one-shot", d["n"], f"{PP(d['accuracy'][0])} {CIP(d['accuracy'][1])}",
                     f"{PP(d['far'][0])} {CIP(d['far'][1])}", f"{PP(d['frr'][0])} {CIP(d['frr'][1])}"])
T["think_gain"] = table(["Slice", "Comparison", "Items", "Δ accuracy (pts)", "Δ false-accept (pts)", "Δ false-reject (pts)"], rows)

CAUSE = {"max_tokens": "thinking used all 16,000 output tokens before an answer", "refusal": "safety classifier refused",
         "credit": "API credit ran out", "other": "other"}
rows = []
for (n, task), (total, nfail, kinds) in D["fail"].items():
    if nfail:
        rows.append([n, task, f"{nfail} of {total}", "; ".join(f"{v} {CAUSE[k]}" for k, v in kinds.items())])
T["failures"] = table(["Judge", "Task", "Items without a result", "Cause"], rows)

rows = []
for n in (HAI, SON):
    sg = D["single"][n]
    rows.append([n, P(A[n]["accuracy"]), P(sg["acc"]), P(sg["far"]), P(sg["frr"]), P(sg["unanimous"])])
T["single"] = table(["Judge (one-shot)", "Accuracy, 5-vote majority", "Accuracy, one call", "False-accept, one call", "False-reject, one call", "Items with 5/5 identical votes"], rows)

rows = []
for n in THINK:
    for g, gl in (("gsm8k", "GSM8K"), ("mbpp", "MBPP"), ("halueval", "HaluEval"), ("ALL", "All binary")):
        st = D["think_stats"][(n, "binary", g)]
        rows.append([n if g == "gsm8k" else "", gl, st["calls"], f"{st['mean_out']:.0f}", f"{st['median_out']:.0f}", f"{st['p95_out']:.0f}",
                     f"{st['max_out']:,}", P(st["thinking_share"], 0)])
    st = D["think_stats"][(n, "rubric", "ALL")]
    rows.append(["", "Rubric", st["calls"], f"{st['mean_out']:.0f}", f"{st['median_out']:.0f}", f"{st['p95_out']:.0f}", f"{st['max_out']:,}", P(st["thinking_share"], 0)])
T["think_effort"] = table(["Arm (effort max)", "Slice", "Calls", "Mean output tokens", "Median", "95th pct", "Max", "Calls with thinking"], rows)

rows = []
for c in CRITERIA:
    m0 = D["rub"][c][JEV]
    rows.append([CRIT_LABEL[c], "*most common human score*", m0["n"], P(m0["majority_exact"]), N2(m0["majority_mae"]), "0.00", "–"])
    for n in NAMES:
        m = D["rub"][c][n]
        rows.append(["", n, m["n"], P(m["exact"]), f"{N2(m['mae'])} [{N2(m['mae_ci'][0])}, {N2(m['mae_ci'][1])}]",
                     f"{N2(m['qwk'])} [{N2(m['qwk_ci'][0])}, {N2(m['qwk_ci'][1])}]", N2(m["spearman"])])
T["rubric"] = table(["Criterion", "Judge", "n", "Exact match", "MAE [95% CI]", "QWK [95% CI]", "Spearman"], rows)

cell = lambda d, k: f"{SG(d[k][0])} [{SG(d[k][1][0])}, {SG(d[k][1][1])}]"
rows = []
for o in others:
    n_, d = D["rub_paired"][o]
    rows.append([f"Jev − {o}", n_, cell(d, "helpfulness"), cell(d, "correctness"), cell(d, "coherence"), cell(d, "mean")])
T["rub_paired"] = table(["Comparison", "n", "Helpfulness", "Correctness", "Coherence", "Mean of three"], rows)
rows = []
for t, m in THINK.items():
    n_, d = D["rub_think_gain"][t]
    rows.append([f"{m}: thinking − one-shot", n_, cell(d, "helpfulness"), cell(d, "correctness"), cell(d, "coherence"), cell(d, "mean")])
T["rub_think_gain"] = table(["Comparison", "n", "Helpfulness", "Correctness", "Coherence", "Mean of three"], rows)

rows = []
for t, tl in (("binary", "Binary"), ("rubric", "Rubric")):
    for n in NAMES:
        p = lat[(t, n)]
        tin, calls, _, tout = D["tok"][(t, n)]
        cost = f"${1000 * D['cost_item'][(n, t)]:.1f}" if (n, t) in D["cost_item"] else "n/a"
        rows.append([tl if n == JEV else "", n, f"{1000 * p['p50']:.0f}", f"{1000 * p['p95']:.0f}", f"{1000 * D['call_lat'][(n, t)]:.0f}",
                     f"{calls:.0f}", f"{tin:,.0f} / {tout:,.0f}", cost])
T["latency"] = table(["Task", "Judge", "Per item, median (ms)", "Per item, p95 (ms)", "One call, median (ms)", "Calls per item",
                      "Tokens in / out per item", "Claude API cost per 1,000 items"], rows)

# ----------------------------------------------------------------------------- numbers
V = {"n_boot": str(N_BOOT)}
for n in NAMES:
    k, m = KEY[n], A[n]
    V.update({f"{k}_acc": P(m["accuracy"]), f"{k}_acc_ci": CI(m["accuracy_ci"]), f"{k}_n": str(m["n"]), f"{k}_far": P(m["far"]),
              f"{k}_frr": P(m["frr"]), f"{k}_mcc": N2(m["mcc"]), f"{k}_ece": opt(m, "ece", 3), f"{k}_auroc": opt(m, "auroc"),
              f"{k}_conf_share": P(m["frac_conf_ge_0.9"], 0), f"{k}_conf_acc": P(m["acc_when_conf_ge_0.9"]),
              f"{k}_acc_allwrong": P(m["accuracy"] * m["n"] / 1100)})
    for s in ("gsm8k", "mbpp", "halueval"):
        V[f"{k}_{s}"] = P(D["bin"][s][n]["accuracy"])
        V[f"{k}_{s}_far"] = P(D["bin"][s][n]["far"])
        V[f"{k}_{s}_frr"] = P(D["bin"][s][n]["frr"])
for o in others:
    k = KEY[o]
    for s in SRCS:
        d = D["paired"][s][o]
        sfx = "" if s == "ALL" else f"_{s}"
        V.update({f"d_{k}_acc{sfx}": PP(d["accuracy"][0]), f"d_{k}_acc{sfx}_abs": PA(d["accuracy"][0]),
                  f"d_{k}_acc{sfx}_ci": CIP(d["accuracy"][1]), f"d_{k}_far{sfx}": PP(d["far"][0]), f"d_{k}_far{sfx}_abs": PA(d["far"][0]),
                  f"d_{k}_far{sfx}_ci": CIP(d["far"][1]), f"d_{k}_frr{sfx}": PP(d["frr"][0]), f"d_{k}_frr{sfx}_abs": PA(d["frr"][0]),
                  f"d_{k}_frr{sfx}_ci": CIP(d["frr"][1])})
for t, m in THINK.items():
    k = KEY[m]
    for s in SRCS:
        d = tg[s][t]
        sfx = "" if s == "ALL" else f"_{s}"
        V.update({f"g_{k}_acc{sfx}": PP(d["accuracy"][0]), f"g_{k}_acc{sfx}_ci": CIP(d["accuracy"][1]),
                  f"g_{k}_far{sfx}_abs": PA(d["far"][0]), f"g_{k}_frr{sfx}_abs": PA(d["frr"][0])})
    for g in ("gsm8k", "mbpp", "halueval", "ALL"):
        st = D["think_stats"][(t, "binary", g)]
        V.update({f"ts_{KEY[t]}_{g}_mean": f"{st['mean_out']:.0f}", f"ts_{KEY[t]}_{g}_median": f"{st['median_out']:.0f}"})
    V[f"ts_{KEY[t]}_share"] = P(D["think_stats"][(t, "binary", "ALL")]["thinking_share"], 0)
    V[f"ts_{KEY[t]}_rubric_share"] = P(D["think_stats"][(t, "rubric", "ALL")]["thinking_share"], 0)
    V[f"ts_{KEY[t]}_rubric_mean"] = f"{D['think_stats'][(t, 'rubric', 'ALL')]['mean_out']:.0f}"
for n, k in ((HAI, "hai"), (SON, "son")):
    V.update({f"{k}_single": P(D["single"][n]["acc"]), f"{k}_unan": P(D["single"][n]["unanimous"])})
for (n, task), (total, nfail, kinds) in D["fail"].items():
    V[f"fail_{KEY[n]}_{task}"] = str(nfail)
    for kind in ("max_tokens", "refusal", "credit"):
        V[f"fail_{KEY[n]}_{task}_{kind}"] = str(kinds.get(kind, 0))
V["refusal_calls_son"] = str(D["refusals_by_judge"][SON])

for n in NAMES:  # old effort-high run
    pass
for n, k in ((HAI, "hai"), (SON, "son")):
    for task, g in (("binary", "ALL"), ("binary", "gsm8k"), ("binary", "mbpp"), ("binary", "halueval"), ("rubric", "ALL")):
        V[f"old_{k}_{task}_{g}"] = P(D["old_effort"][(n, task, g)], 0)

qwks = [D["rub"][c][n]["qwk"] for c in CRITERIA for n in NAMES]
V.update({"qwk_min": N2(min(qwks)), "qwk_max": N2(max(qwks))})
for c in CRITERIA:
    V[f"gold4_{c}"] = f"{D['rub'][c][JEV]['gold_hist'][4]}"
    for n in NAMES:
        k, m = KEY[n], D["rub"][c][n]
        V[f"{k}_pred4_{c}"] = str(m["pred_hist"][4])
        V[f"{k}_mae_{c}"] = N2(m["mae"])
        V[f"{k}_exact_{c}"] = P(m["exact"])
    V[f"base_exact_{c}"] = P(D["rub"][c][JEV]["majority_exact"])
    V[f"base_mae_{c}"] = N2(D["rub"][c][JEV]["majority_mae"])
for o in others:
    n_, d = D["rub_paired"][o]
    k = KEY[o]
    V.update({f"rp_{k}_mean": SG(d["mean"][0]), f"rp_{k}_mean_abs": N2(abs(d["mean"][0])),
              f"rp_{k}_mean_ci": f"[{SG(d['mean'][1][0])}, {SG(d['mean'][1][1])}]"})
    for c in CRITERIA:
        V[f"rp_{k}_{c}"] = SG(d[c][0])
for t, m in THINK.items():
    n_, d = D["rub_think_gain"][t]
    k = KEY[m]
    V.update({f"rg_{k}_mean": SG(d["mean"][0]), f"rg_{k}_mean_abs": N2(abs(d["mean"][0])),
              f"rg_{k}_mean_ci": f"[{SG(d['mean'][1][0])}, {SG(d['mean'][1][1])}]"})
    for c in CRITERIA:
        V[f"rg_{k}_{c}"] = SG(d[c][0])
        V[f"rg_{k}_{c}_ci"] = f"[{SG(d[c][1][0])}, {SG(d[c][1][1])}]"

for t in ("binary", "rubric"):
    for n in NAMES:
        k = KEY[n]
        V[f"lat_{t}_{k}"] = SEC(lat[(t, n)]["p50"])
        V[f"lat_{t}_{k}_p95"] = SEC(lat[(t, n)]["p95"])
        V[f"call_{t}_{k}"] = SEC(D["call_lat"][(n, t)])
        V[f"ratio_{t}_{k}"] = f"{lat[(t, n)]['p50'] / lat[(t, JEV)]['p50']:.0f}"
        V[f"callratio_{t}_{k}"] = f"{D['call_lat'][(n, t)] / D['call_lat'][(JEV, t)]:.1f}"
for n in NAMES[1:]:
    k = KEY[n]
    for t in ("binary", "rubric"):
        V[f"cost1000_{t}_{k}"] = f"${1000 * D['cost_item'][(n, t)]:.1f}"
        V[f"costcall_{t}_{k}"] = f"${D['cost_call'][(n, t)]:.4f}"
V["costcall_ratio_son"] = f"{D['cost_call'][(SONT, 'binary')] / D['cost_call'][(SON, 'binary')]:.0f}"
V["costcall_ratio_hai"] = f"{D['cost_call'][(HAIT, 'binary')] / D['cost_call'][(HAI, 'binary')]:.0f}"
V["tok_jev"] = f"{D['tok'][('binary', JEV)][0]:,.0f}"
V["tok_cl"] = f"{D['tok'][('binary', SON)][0]:,.0f}"
import csv as _csv
V["total_calls"] = f"{sum(1 for _ in open(os.path.join(os.path.dirname(OUT), 'results', 'calls.csv'), newline='')) - 1:,}"
V["spend_final"] = f"${sum(D['spend'].values()):.0f}"
V["spend_think"] = f"${D['spend'][HAIT] + D['spend'][SONT]:.0f}"

TEXT = open(os.path.join(OUT, "article_template.md")).read()
for k, v in T.items():
    TEXT = TEXT.replace(f"{{{{TABLE:{k}}}}}", v)
missing = set(re.findall(r"@@(\w+)@@", TEXT)) - set(V)
assert not missing, sorted(missing)
TEXT = re.sub(r"@@(\w+)@@", lambda m: V[m.group(1)], TEXT)
leftover = re.findall(r"\{\{TABLE:\w+\}\}", TEXT)
assert not leftover, leftover


# -------------------------------------------------------------------------- markdown file
def fig_md(m):
    return f"![{m.group(2)}](figures/{m.group(1)}.svg)\n\n*{m.group(2)}*"


MD = re.sub(r"\{\{FIG:(\w+)\|([^}]*)\}\}", fig_md, TEXT)
open(f"{OUT}/jev_as_a_judge.md", "w").write(MD)


# ---------------------------------------------------------------------------- html file
def fig_html(m):
    return f'<div class="fig">\n{FIG[m.group(1)]}\n<div class="cap">{m.group(2)}</div>\n</div>'


body_md = re.sub(r"\{\{FIG:(\w+)\|([^}]*)\}\}", fig_html, TEXT)
body = markdown.markdown(body_md, extensions=["tables", "sane_lists", "pymdownx.arithmatex", "toc"],
                         extension_configs={"pymdownx.arithmatex": {"generic": True}})
title = re.search(r"^# (.+)$", TEXT, re.M).group(1)
HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{S.esc(title)}</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<style>
:root {{ color-scheme: light; }}
html, body {{ background: #ffffff; margin: 0; }}
body {{ color: #0b0b0b; font-family: {S.FONT}; font-size: 17px; line-height: 1.65; }}
main {{ max-width: 800px; margin: 0 auto; padding: 48px 16px 96px; }}
h1 {{ font-size: 2.1rem; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.45rem; margin: 2.4em 0 0.4em; padding-top: 0.4em; border-top: 1px solid #e8e7e3; }}
h3 {{ font-size: 1.15rem; margin: 1.8em 0 0.3em; }}
p, li {{ max-width: 70ch; }}
a {{ color: #1c5cab; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.88em; border: 1px solid #e8e7e3; border-radius: 4px; padding: 0 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.84rem; line-height: 1.4; margin: 1.1em 0 1.5em; display: block; overflow-x: auto; }}
th, td {{ text-align: left; padding: 6px 9px; border-bottom: 1px solid #e8e7e3; vertical-align: top; }}
th {{ border-bottom: 1.5px solid #0b0b0b; font-weight: 600; }}
.fig {{ margin: 1.6em 0 2em; }}
.fig .cap {{ font-size: 0.86rem; color: #52514e; margin-top: 6px; max-width: 70ch; }}
.arithmatex {{ overflow-x: auto; }}
div.arithmatex {{ margin: 0.9em 0; }}
blockquote {{ margin: 1.4em 0; padding: 4px 0 4px 18px; border-left: 3px solid #2a78d6; color: #0b0b0b; }}
blockquote p {{ margin: 0.5em 0; }}
.toc {{ display: none; }}
@media (max-width: 600px) {{ body {{ font-size: 16px; }} h1 {{ font-size: 1.7rem; }} }}
</style>
</head>
<body>
<main>
{body}
</main>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '\\\\[', right: '\\\\]', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}], throwOnError: false}});"></script>
</body>
</html>
"""
open(f"{OUT}/jev_as_a_judge.html", "w").write(HTML)
print("wrote", OUT, "| words in md:", len(MD.split()))
