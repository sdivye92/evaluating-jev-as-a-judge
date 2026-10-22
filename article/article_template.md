# Can Jev Replace the LLM Judge? Measuring Correctness Before Speed

*A pointwise comparison of Jev 1.13 with Claude Haiku 5.5 and Claude Sonnet 5.5, each used one-shot and with thinking, on binary correctness and rubric scoring. Experiments run in October 2026.*

## Introduction

Since TypeSafe introduced Jev as a System One decision model, there has been a lot of interest in using it wherever a system has to make a decision from a context and a question. Jev is built for fast, single-pass decisions: instead of generating text, it returns a typed answer together with calibrated probabilities [1].

One task that fits this shape is evaluating the answers of another AI system. When a language model reads a question and a response and decides whether the response is good, that is *LLM-as-a-judge* [2]. With Jev getting attention, the natural next step is *Jev-as-a-judge*: put Jev in the judge's seat instead of an LLM.

I wanted to find out how good Jev's decisions are compared with an LLM's, and to look at correctness before anything else. A judge that answers in 300 milliseconds is not worth much if its verdicts are wrong. The judge is a gate in the AI system lifecycle: it decides which model version ships, which responses pass a guardrail, which regressions block a release, which examples are kept as training data. If it is wrong, every decision downstream inherits the error, and latency and cost savings do not repair that. A judge that quietly passes one bad answer in six is not cheap; it moves the cost to wherever the bad answers surface.

> **In short**
>
> - **One-shot LLMs:** on binary correctness, Jev 1.13 and Claude Sonnet 5.5 are statistically tied (@@jev_acc@@ vs @@son_acc@@; Jev minus Sonnet @@d_son_acc@@ points, 95% CI @@d_son_acc_ci@@). Jev is clearly ahead of Claude Haiku 5.5 (@@hai_acc@@).
> - **With thinking switched on, the LLM judges are clearly more accurate than Jev:** Haiku 5.5 reaches @@hait_acc@@ and Sonnet 5.5 @@sont_acc@@, about 11 points above Jev, and their false-accept rates are @@hait_far@@ and @@sont_far@@ against @@jev_far@@ for Jev.
> - **Thinking helps most where checking an answer means working it out:** the gain is largest on math word problems (GSM8K), smaller on code (MBPP), and small on hallucination detection (HaluEval).
> - **Jev's mistakes lean toward accepting wrong answers** (@@jev_far@@ of wrong answers accepted), but its probabilities are well calibrated (calibration error @@jev_ece@@), which the Claude vote tallies are not.
> - **Rubric scoring (0 to 4) is hard for every judge.** Agreement with human ratings is moderate (quadratic weighted kappa @@qwk_min@@ to @@qwk_max@@) and thinking barely helps. Jev is level with the thinking judges there.
> - **The price of thinking:** Jev's median time per binary verdict was @@lat_binary_jev@@, against @@lat_binary_hait@@ and @@lat_binary_sont@@ for the thinking judges (about @@ratio_binary_hait@@ to @@ratio_binary_sont@@ times slower). The thinking judges also returned no verdict for @@fail_hait_binary@@ (Haiku) and @@fail_sont_binary@@ (Sonnet) of 1,100 items, because thinking used up the whole output limit.

## What is being evaluated?

The thing under test is the judge's *decision*, not the answers it looks at. Every judge gets one question and one candidate answer and has to decide on that pair alone. This is the *pointwise* setting. I chose it because in most agentic evaluation setups there is one question and one answer, and you need a verdict on it standing alone, not a comparison of two responses.

I tested two kinds of decisions.

**1. Binary correctness.** Is the candidate answer correct or incorrect?

**2. Rubric scoring.** Score the response from 0 to 4 on three criteria: helpfulness, correctness and coherence. Each score level has a written description, such as "0: completely incorrect" up to "4: completely correct".

The judges were tested on four datasets, chosen so that the "right" answer comes from somewhere other than another model's opinion:

| Dataset | What the judge sees | Where the label comes from | Items |
|---|---|---|---|
| GSM8K [3] (math word problems) | the problem and a final answer only | correct: the gold answer. Incorrect: a near-miss number (off by one or ten, doubled or halved, digits swapped, or a number from the problem mixed in) | 400 (200 questions) |
| MBPP [4] (Python functions) | the task description, one visible example test, and the code | by running the tests. Incorrect: the reference solution with a single edit (a flipped comparison, a changed operator or constant), kept only if it still runs and fails a test | 300 (150 questions) |
| HaluEval QA [5] (question answering) | the question, supporting knowledge, and an answer | correct: the right answer. Incorrect: the dataset's hallucinated answer | 400 (200 questions) |
| HelpSteer2 [6] (rubric) | a prompt and a response | human ratings from 0 to 4 on helpfulness, correctness and coherence | 300 responses |

The binary set has 1,100 items, half correct and half incorrect, so a judge that guesses scores 50%. For every question I included one correct and one incorrect candidate. Two details protect against shortcuts:

- GSM8K candidates are the final answer alone, so the judge has to solve the problem and cannot just read a worked solution.
- In HaluEval the hallucinated answers are mostly full sentences and the right answers are short phrases; length alone separates them with an AUROC of 0.97. I kept only questions where the two answers have similar length (within 0.8 to 1.25 times). After that, answer length predicts correctness at chance level (AUROC 0.50 to 0.51 on every source), so a judge cannot score well by reading the form of the answer.

## How it is evaluated, and why these metrics

### Binary correctness

Each judge produces a probability that the candidate is correct. The verdict is whatever the judge itself says: *correct* if that probability is at least 0.5. I did not tune thresholds on any judge. Let $TP$, $TN$, $FP$ and $FN$ count correct answers accepted, incorrect answers rejected, incorrect answers accepted, and correct answers rejected.

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy alone hides the thing that matters for a gate, because the two kinds of mistakes have different costs. So I report them separately:

$$
\text{False-accept rate}=\frac{FP}{FP+TN}
\qquad
\text{False-reject rate}=\frac{FN}{FN+TP}
$$

The false-accept rate is the share of wrong answers the judge lets through. The false-reject rate is the share of good answers it throws out. The Matthews correlation coefficient combines all four counts and equals 0 for a judge that is no better than guessing:

$$
\text{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

I also look at how trustworthy the *probabilities* are, since they decide how much one can lean on a verdict (for example, by sending low-confidence cases to a stronger judge). The *AUROC* is the chance that a randomly picked correct answer gets a higher probability than a randomly picked incorrect one, so it measures ranking quality, independent of any threshold. The *expected calibration error* (ECE) compares stated confidence with actual accuracy. Group the items into confidence bins $B_b$ by $c=\max(p,1-p)$:

$$
\text{ECE}=\sum_{b}\frac{|B_b|}{N}\,\bigl|\,\text{acc}(B_b)-\text{conf}(B_b)\,\bigr|
$$

A perfectly calibrated judge has an ECE of 0: of the answers it calls 80% likely, it is right 80% of the time. AUROC and ECE need a graded probability, so they are only reported for judges that return one.

If a judge returns no verdict for an item (for example, it ran out of output tokens), that item is left out of that judge's metrics and counted in a separate failure table. Because leaving failures out flatters a judge, I also give the accuracy with failures counted as wrong.

### Rubric scoring

For each criterion, the judge gives a probability $p_k$ for each score level $k=0,\dots,4$. Jev returns this distribution directly. Claude has no such output, so $p_k$ is the share of its samples that picked level $k$. The judge's score is the expected value:

$$
\hat{s}=\sum_{k=0}^{4} k\,p_k
$$

I then compare $\hat{s}$ with the human rating $y$. Mean absolute error (MAE) measures the typical distance, and exact match (after rounding $\hat{s}$ to the nearest level) measures how often the judge hits the human score. Because the scale is ordinal, missing by one level is a smaller mistake than missing by four, so I also report *quadratic weighted kappa*, which penalizes disagreements by the square of their distance and corrects for chance agreement:

$$
\kappa_w = 1-\frac{\sum_{i,j} w_{ij}\,O_{ij}}{\sum_{i,j} w_{ij}\,E_{ij}},
\qquad
w_{ij}=\frac{(i-j)^2}{(K-1)^2}
$$

Here $O$ is the observed table of (judge level, human level) pairs, $E$ is the table expected by chance, and $K=5$. A kappa of 0 is chance and 1 is perfect agreement. Spearman's rank correlation checks whether the judge orders responses the way humans do. Human ratings are skewed toward high scores, so a trivial predictor looks decent. I therefore compare against the baseline of *always predicting the most common human score* for each criterion. This baseline peeks at the test labels, which makes it a strong one.

### Uncertainty and paired comparisons

All 95% confidence intervals come from a bootstrap that resamples whole *questions*, not individual items, because the correct and incorrect candidates of one question are not independent. To compare two judges, I use a *paired* bootstrap on the same items. On each resample I recompute both judges' metric and take the difference $\Delta=m(\text{judge A})-m(\text{judge B})$. The 2.5th and 97.5th percentiles of $\Delta$ over @@n_boot@@ resamples form the interval. If it excludes 0, the gap is unlikely to be a sampling accident. Pairing matters because both judges tend to find the same items hard, and the pairing cancels that shared difficulty.

## Experimental setup

| Judge | Model | Thinking | Samples per binary item / per rubric criterion | Interface |
|---|---|---|---|---|
| Jev 1.13 | `jev-1.13.0` (version pinned) | single pass | 1 call returns the probabilities | `langchain-typesafe` |
| Haiku 5.5 | `claude-haiku-5-5` | off | 5 / 5 | `langchain-anthropic` |
| Haiku 5.5 + thinking | `claude-haiku-5-5` | adaptive, effort `max` | **1 / 1** | `langchain-anthropic` |
| Sonnet 5.5 | `claude-sonnet-5-5` | off | 5 / 5 | `langchain-anthropic` |
| Sonnet 5.5 + thinking | `claude-sonnet-5-5` | adaptive, effort `max` | **1 / 1** | `langchain-anthropic` |

- **Same information, same wording.** All judges got the same instruction text and the same fields of each item. Jev received typed questions: a yes/no question for binary correctness, and one five-level `Score` question per rubric criterion, answered in a single request. Claude received the same text as a system prompt and the item as JSON.
- **A fair one-shot comparison.** Jev decides in a single pass. To keep that comparison matched, the one-shot Claude arms were run with thinking turned off and low effort, and their reply was forced into a tiny structured object (`{"verdict": "YES"}` or `{"score": "3"}`). My first attempt only asked for "one word", and that did not work: on math and code items the models began solving the problem in the visible reply and ran out of tokens before giving a verdict (roughly a third of Haiku's and nearly 60% of Sonnet's binary verdicts were unparseable). Constraining the output removed that failure.
- **The thinking arms use the same prompt and the same structured reply**, with thinking turned on. Forcing the reply keeps the reasoning in the model's hidden thinking, where it cannot steer the visible verdict, so the one-shot and thinking arms differ only in whether the model thinks. An earlier version of the thinking arm used free-text replies, and Sonnet wrote its working in the reply, which made the two arms impossible to compare. I discarded that run.
- **Adaptive thinking, and why effort is `max`.** The newer Claude models use *adaptive thinking*: the model itself decides whether a question needs thinking at all and how much to think. The only control is the *effort* setting, because these models reject a fixed thinking budget. In my experiments, at effort `high` Sonnet did not think on many of the data points. In an earlier run at `high`, @@old_son_binary_halueval@@ of Sonnet's HaluEval verdicts and @@old_son_rubric_ALL@@ of its rubric scores were produced with no thinking at all (the call used 8 output tokens or fewer); Haiku skipped thinking on @@old_hai_binary_halueval@@ of HaluEval verdicts. In a small probe of 8 binary items, Sonnet thought on 2 at `high`, 5 at `xhigh` and all 8 at `max`. Haiku thought on most calls even at `high`. That is why the thinking arms run at effort `max`. In the final run thinking was present on @@ts_sont_share@@ of Sonnet's and @@ts_hait_share@@ of Haiku's binary calls. Every call's thinking summary is logged, so this is checkable.
- **One sample for the thinking arms, because of cost.** Thinking calls are expensive, so the thinking arms use one sample per binary item and one per rubric criterion (`BINARY_SAMPLES = 1`, `RUBRIC_SAMPLES = 1`), while the one-shot arms use five. This has consequences. A single binary vote is 0 or 1, so the thinking arms have no graded probability, and AUROC and ECE are not reported for them. A single rubric sample gives an integer score where the one-shot arms average five, which makes the rubric comparison less like-for-like. For binary accuracy the effect should be small: for the one-shot arms, one call was as accurate as the five-vote majority (table below). I did not check this for the rubric. The two thinking arms cost about @@spend_think@@ in Claude API usage at list prices.
- **Sampling.** The Claude 5.5 models reject non-default sampling settings, so every call ran at the default temperature. For the one-shot arms, a binary item's probability is the share of YES votes among five and the rubric probabilities are the shares of each level. Jev returns its probabilities from a single call.
- **Scale.** The experiment made @@total_calls@@ logged judge calls, each with its raw output and, for the thinking arms, the thinking summary. Four items were evaluated at a time, and latency was measured on the client as wall-clock time per judged item.
- **Missing results.** Some items have no result for a judge:

{{TABLE:failures}}

Two causes remain. The thinking judges sometimes used all 16,000 output tokens on thinking and returned no verdict, and Sonnet's safety classifier refused two of the 300 HelpSteer2 responses (the same two for both Sonnet arms, on every attempt). Failed items were retried twice (three attempts in all), which recovered some of them: the first pass left 13 Haiku and 7 Sonnet binary verdicts missing, and the retries cut that to the numbers in the table. In that first pass my API credit also ran out near the end of Sonnet's thinking rubric run, which left 21 responses unjudged; I topped up and retried them, so they are included in the results.

## Results

### Binary correctness: overall accuracy

{{TABLE:headline}}

{{FIG:f1_accuracy_by_source|Figure 1. Binary accuracy by data source and pooled, with 95% confidence intervals (whiskers). Hatched bars are the thinking arms. The line marks chance (50%). Only Jev's bars are labeled; exact values for all judges are in the tables.}}

Among the one-shot judges, Jev (@@jev_acc@@) and Sonnet (@@son_acc@@) are indistinguishable: the paired difference is @@d_son_acc@@ points with a 95% interval of @@d_son_acc_ci@@. Haiku is clearly behind (@@hai_acc@@): Jev is ahead by @@d_hai_acc_abs@@ points (interval @@d_hai_acc_ci@@).

Switching thinking on changes the picture. Haiku with thinking reaches @@hait_acc@@ and Sonnet with thinking @@sont_acc@@, and both are clearly ahead of Jev: Jev is behind Haiku-with-thinking by @@d_hait_acc_abs@@ points (interval @@d_hait_acc_ci@@) and behind Sonnet-with-thinking by @@d_sont_acc_abs@@ points (interval @@d_sont_acc_ci@@). Those accuracies are computed on the items where the judge returned a verdict. If the failed items are counted as wrong, they are @@hait_acc_allwrong@@ and @@sont_acc_allwrong@@, which does not change the ranking.

The pooled number averages over very different tasks:

{{TABLE:by_source}}

- **GSM8K (math).** Jev @@jev_gsm8k@@ beats the one-shot judges (Sonnet @@son_gsm8k@@, Haiku @@hai_gsm8k@@), but the thinking judges are near perfect (Haiku @@hait_gsm8k@@, Sonnet @@sont_gsm8k@@). A one-shot judge has to guess whether a number is right; a thinking judge can compute it.
- **MBPP (code).** Sonnet one-shot is best of the one-shot judges at @@son_mbpp@@; Jev is at @@jev_mbpp@@ and Haiku at @@hai_mbpp@@. With thinking, Haiku reaches @@hait_mbpp@@ and Sonnet @@sont_mbpp@@.
- **HaluEval (hallucination).** Every judge is in the low to mid 90s: Jev @@jev_halueval@@, Haiku @@hai_halueval@@, Sonnet @@son_halueval@@, and with thinking @@hait_halueval@@ and @@sont_halueval@@. Jev's gap to the thinking judges is small here (@@d_hait_acc_halueval@@ points to Haiku with thinking, interval @@d_hait_acc_halueval_ci@@; @@d_sont_acc_halueval@@ points to Sonnet with thinking, interval @@d_sont_acc_halueval_ci@@).

The per-source differences are exploratory: the experiment was not designed to test each source separately, and I made no multiple-comparison correction.

### How much does thinking help?

{{FIG:f4_thinking_gain|Figure 2. Accuracy gain from enabling thinking (thinking arm minus the same model one-shot), by data source, with 95% confidence intervals.}}

{{TABLE:think_gain}}

Thinking raises accuracy by @@g_hai_acc@@ points for Haiku (interval @@g_hai_acc_ci@@) and @@g_son_acc@@ points for Sonnet (interval @@g_son_acc_ci@@), and it lowers both error types. The gain depends on the data. It is largest on GSM8K (@@g_hai_acc_gsm8k@@ points for Haiku, @@g_son_acc_gsm8k@@ for Sonnet), where verifying means doing the arithmetic, smaller on MBPP (@@g_hai_acc_mbpp@@ and @@g_son_acc_mbpp@@), and small on HaluEval (@@g_hai_acc_halueval@@ and @@g_son_acc_halueval@@), where the answer mostly has to be checked against the supplied knowledge. Haiku, the smaller model, gains more than Sonnet.

How much the models thought also depended on the data. With effort `max`, the number of output tokens per call (which includes the hidden thinking) was:

{{TABLE:think_effort}}

Code questions made the models think the longest, and a few calls ran for thousands of tokens. A thinking budget is therefore not a fixed cost per item, which matters for both latency and spend.

### The two judges make different mistakes

{{FIG:f2_error_profile|Figure 3. Pooled false-accept rate (incorrect answers marked correct) and false-reject rate (correct answers marked incorrect), with 95% confidence intervals. Hatched bars are the thinking arms. Only Jev's bars are labeled.}}

Equal accuracy does not mean equal behavior. Among the one-shot judges, Jev leans toward accepting: it let through @@jev_far@@ of the wrong answers and rejected @@jev_frr@@ of the correct ones. Sonnet leans toward rejecting: @@son_far@@ of wrong answers accepted and @@son_frr@@ of correct ones rejected. One-shot Haiku is the most conservative, with a false-reject rate of @@hai_frr@@ (it called more than a third of the correct answers wrong).

The thinking judges are better on both counts: false-accept rates of @@hait_far@@ (Haiku) and @@sont_far@@ (Sonnet), and false-reject rates of @@hait_frr@@ and @@sont_frr@@. Jev's higher false-accept rate shows up in every source against Sonnet one-shot (@@jev_gsm8k_far@@ against @@son_gsm8k_far@@ on GSM8K, @@jev_mbpp_far@@ against @@son_mbpp_far@@ on MBPP, and @@jev_halueval_far@@ against @@son_halueval_far@@ on HaluEval). For a quality gate this is the number to watch, because a false accept lets a bad answer through unnoticed.

{{FIG:f3_paired_vs_jev|Figure 4. Paired differences between Jev and each Claude judge on the pooled binary items, in percentage points, with 95% confidence intervals. Positive values favour Jev (for the two error rates the sign is flipped, so fewer errors count as an advantage). A hollow dot marks a thinking arm.}}

{{TABLE:paired}}

### Can the probabilities be trusted?

{{FIG:f5_reliability|Figure 5. Reliability diagram on the pooled binary items for the judges that return graded probabilities. A judge on the diagonal is calibrated: when it states 80% confidence, it is right 80% of the time. Marker area is proportional to the number of items in the confidence bin. The thinking arms are absent because one sample gives only 0 or 1.}}

Jev's ranking quality is high (AUROC @@jev_auroc@@, against @@son_auroc@@ for Sonnet and @@hai_auroc@@ for Haiku, both one-shot) and its calibration error is small (ECE @@jev_ece@@, against @@son_ece@@ and @@hai_ece@@). In the diagram, Jev's points sit close to the diagonal across the whole range: @@jev_conf_share@@ of its verdicts are made with at least 90% confidence, and those are right @@jev_conf_acc@@ of the time.

The one-shot Claude judges behave differently. Because their probability comes from five votes, almost every item lands in the top bin: @@hai_conf_share@@ of Haiku's and @@son_conf_share@@ of Sonnet's items got five identical votes. Those unanimous verdicts were right only @@hai_conf_acc@@ (Haiku) and @@son_conf_acc@@ (Sonnet) of the time. Agreement between samples is not the same as being correct: a model can be consistently wrong. For the thinking judges I cannot say how well calibrated they are, since one sample gives no probability.

One caveat for fairness: the one-shot Claude probabilities have only six possible values (0, 0.2, ..., 1), while Jev's are continuous, which by itself makes AUROC and ECE harder to compare. The comparison is still informative about *usefulness*: Jev's confidence can be used to decide when to trust a verdict, and the Claude vote tallies mostly cannot.

How much does the five-sample protocol matter for the one-shot accuracy? Very little:

{{TABLE:single}}

A single one-shot call is as accurate as the five-vote majority, because the votes agree on @@hai_unan@@ (Haiku) and @@son_unan@@ (Sonnet) of the items. The five samples mainly buy the coarse probabilities. This is the reason I am fairly relaxed about the thinking arms using one sample for the binary task.

### Rubric scoring

{{TABLE:rubric}}

{{FIG:f6_rubric_mae|Figure 6. Mean absolute error of the expected score against the human rating, with 95% confidence intervals. The black bar in each group is the error of always predicting the most common human score for that criterion. Lower is better. Hatched bars are the thinking arms. Only Jev's bars are labeled.}}

{{FIG:f7_rubric_qwk|Figure 7. Quadratic weighted kappa between each judge's rounded score and the human rating (0 is chance agreement, 1 is perfect). Higher is better.}}

Rubric scoring is clearly harder than the binary decision, for every judge, and thinking does not change that much.

- **Agreement is moderate.** Quadratic weighted kappa ranges from @@qwk_min@@ to @@qwk_max@@ across all judges and criteria, and the rank correlations are in the same range: the judges order responses roughly the way humans do, but not closely.
- **Exact match is below the trivial baseline in every case.** For example, on helpfulness the judges match the human score on @@jev_exact_helpfulness@@ (Jev), @@hai_exact_helpfulness@@ and @@hait_exact_helpfulness@@ (Haiku, one-shot and with thinking) and @@son_exact_helpfulness@@ and @@sont_exact_helpfulness@@ (Sonnet) of responses, while always guessing the most common human score would match @@base_exact_helpfulness@@. For coherence, humans rated 73% of responses a 4, so the baseline hits @@base_exact_coherence@@ and the judges only between @@son_exact_coherence@@ and @@sont_exact_coherence@@.
- **MAE tells a similar story.** On helpfulness the judges are close to the baseline error (@@jev_mae_helpfulness@@ for Jev, against @@base_mae_helpfulness@@). On correctness they are level or slightly worse (baseline @@base_mae_correctness@@). On coherence all of them are much worse than the baseline (@@jev_mae_coherence@@ for Jev and @@sont_mae_coherence@@ for Sonnet with thinking, against @@base_mae_coherence@@).
- **All judges score harder than the humans.** Humans gave a 4 for helpfulness to @@gold4_helpfulness@@ of 300 responses; Jev did so for @@jev_pred4_helpfulness@@, one-shot Haiku for @@hai_pred4_helpfulness@@ and Haiku with thinking for @@hait_pred4_helpfulness@@. The pattern repeats for correctness (humans @@gold4_correctness@@; Jev @@jev_pred4_correctness@@) and, less strongly, for coherence. My level descriptions may be stricter than the annotators' working definition.

Does thinking help here? For Haiku, no: the mean absolute error is unchanged. For Sonnet, a little:

{{TABLE:rub_think_gain}}

Averaged over the three criteria, Sonnet's error with thinking is @@rg_son_mean_abs@@ lower than without it (paired difference @@rg_son_mean@@, 95% CI @@rg_son_mean_ci@@), driven by correctness and coherence, and Haiku's is unchanged (@@rg_hai_mean@@, @@rg_hai_mean_ci@@). Compared with Jev:

{{TABLE:rub_paired}}

Jev is level with both thinking judges on the average error (@@rp_hait_mean@@ against Haiku with thinking, @@rp_sont_mean@@ against Sonnet with thinking, both with intervals that include zero) and has a smaller error than one-shot Sonnet (@@rp_son_mean@@). Read this as "Jev is no worse than the LLMs at a task none of them does well", not as "Jev is good at rubric scoring". Rubric labels from humans are also noisy, which puts a ceiling on how well any judge can match them. Note that the Sonnet numbers cover a few responses fewer than the others (the failure table above).

### Speed and cost

{{FIG:f8_latency|Figure 8. Time per judged item as seen by the client: bar = median, whisker = 95th percentile. A one-shot Claude item takes as long as its slowest of five (binary) or fifteen (rubric) parallel calls; a thinking item is one call (binary) or three parallel calls (rubric). Only Jev's bar is labeled.}}

{{TABLE:latency}}

Jev's median time per binary verdict was @@lat_binary_jev@@ (95th percentile @@lat_binary_jev_p95@@). The one-shot Claude judges took @@lat_binary_hai@@ (Haiku) and @@lat_binary_son@@ (Sonnet), and the thinking judges @@lat_binary_hait@@ and @@lat_binary_sont@@, with a long tail: the 95th percentile was @@lat_binary_hait_p95@@ for Haiku and @@lat_binary_sont_p95@@ for Sonnet. The thinking judges are therefore about @@ratio_binary_hait@@ to @@ratio_binary_sont@@ times slower than Jev per verdict. For rubric scoring Jev's single request (@@lat_rubric_jev@@) compares with @@lat_rubric_hait@@ (Haiku) and @@lat_rubric_sont@@ (Sonnet) for the thinking judges, about @@ratio_rubric_sont@@ to @@ratio_rubric_hait@@ times slower.

Part of the one-shot gap comes from my protocol (the slowest of five parallel calls). Per single call, one-shot Haiku takes @@call_binary_hai@@ and Sonnet @@call_binary_son@@ at the median, so Jev is still @@callratio_binary_hai@@ to @@callratio_binary_son@@ times faster per call. Jev also processes far fewer input tokens per binary item: @@tok_jev@@, against @@tok_cl@@ for the five one-shot Claude calls.

On cost, a thinking call is expensive because the thinking tokens are billed as output: @@costcall_binary_sont@@ per binary call for Sonnet with thinking against @@costcall_binary_son@@ one-shot (about @@costcall_ratio_son@@ times), and @@costcall_binary_hait@@ against @@costcall_binary_hai@@ for Haiku (about @@costcall_ratio_hai@@ times). Per item the picture is muddied by the different sample counts: the one-shot arms make five calls per binary item and the thinking arms one. The table above gives the cost per 1,000 items for what I actually ran. Jev's own pricing is not included, because I do not have it.

## Understanding the results

**These are the numbers from one run.** The conclusions above hold for this setup. They may change on a different run, and they will very likely change with different models:

- **Different runs.** The Claude models were sampled at default temperature. For the one-shot arms the five votes were identical on @@hai_unan@@ (Haiku) and @@son_unan@@ (Sonnet) of binary items, so run-to-run variation there is probably small. But the thinking arms use a single sample of a long, adaptive chain of thought, which I expect to vary more, and I did not repeat any run to measure it. I also did not check how repeatable Jev's output is across runs. The confidence intervals reflect which items were sampled from each dataset. They do not capture run-to-run variation.
- **Different models and settings.** The thinking arms were run at one effort level (`max`), chosen because lower effort made Sonnet skip thinking on many items. Another effort level, a newer or older Claude model, a different Jev version, or a different prompt wording could shift every number. I did not tune the instruction wording for any judge.
- **One sample for the thinking arms.** Because of cost, the thinking arms have a single sample, so no probabilities and a coarser rubric score than the one-shot arms (see the setup).
- **Different data.** The wrong answers are synthetic or dataset-made: near-miss numbers, single-edit code mutants, and benchmark hallucinations. The mistakes of a real model under evaluation may be easier or harder to catch. GSM8K, MBPP, HaluEval and HelpSteer2 are public, so some contamination of either system's training data is possible.
- **Failed items.** Some thinking verdicts and scores are missing (failure table). For binary I show accuracy with failures excluded and counted as wrong, and the ranking holds either way. The missing Sonnet rubric responses are a budget accident, not a model property.
- **Many comparisons.** I looked at several sources, criteria and metrics. The pooled comparisons are the main results; the per-source and per-criterion differences are exploratory and not corrected for multiple comparisons.
- **Label noise.** HelpSteer2 ratings come from human annotators who do not always agree, and the level descriptions I wrote are my paraphrase of the scale.
- **Latency.** Times are client-side wall-clock measurements at a concurrency of four, including network time. They are indicative, not server-side benchmarks.

**What the results do and do not say.** They say that, for pointwise binary correctness, a one-pass Jev can match a same-generation one-shot LLM and beat a smaller one, at a fraction of the latency and with well-calibrated probabilities. They also say that LLM judges that think are clearly more accurate than Jev here (about 11 points, with far fewer false accepts), at about @@ratio_binary_hait@@ to @@ratio_binary_sont@@ times the latency per verdict and with a failure mode of their own, and that Jev's mistakes lean toward accepting wrong answers. They do not show how any of this behaves on real model outputs, and they do not say how a thinking judge would compare at a different effort level.

## Conclusion

Speed and cost only matter once a judge is correct enough, so I measured correctness first. Against one-shot LLM judges, Jev 1.13 did well: @@jev_acc@@ accuracy, level with Claude Sonnet 5.5 (@@son_acc@@) and well ahead of Claude Haiku 5.5 (@@hai_acc@@), while answering in about a third of a second. But once the LLM judges are allowed to think, they win on correctness: @@hait_acc@@ for Haiku and @@sont_acc@@ for Sonnet, with false-accept rates of about 2% where Jev's is @@jev_far@@. That is the answer to the question I started with. If the only thing that matters is whether the verdict is right, a thinking LLM judge beat Jev on this benchmark, and Jev is not yet a drop-in replacement for it.

What Jev offers instead is speed and probabilities you can use. Its verdicts arrive about @@ratio_binary_hait@@ to @@ratio_binary_sont@@ times faster than the thinking judges' (and it never ran out of output budget), and its confidence is well calibrated, so it can say when it is unsure. That suggests a practical design: let Jev decide the cases it is confident about and send the rest to a thinking judge. I have not tested that cascade, so it remains a hypothesis. For rubric scoring on a 0 to 4 scale, no judge I tested was good, thinking barely helped, and Jev was level with the thinking judges.

My reading: Jev looks like a credible fast first-pass judge for correctness decisions, and not yet a replacement where false accepts are expensive or where the judgments are graded. What is still open: the confidence-based cascade, other effort levels and more samples for the thinking arms, repeated runs and other Jev versions, and real outputs from the system you actually want to evaluate.

## A note on the bill

This experiment cost me about $60 of my own money: $57 for everything up to the final rerun and $3 more for that rerun. That makes it a very expensive way to confirm that thinking is not free. Only about @@spend_final@@ of it is visible in the final result files, as Claude usage at list prices. The rest went on things those files do not show: test and pilot runs, the first thinking runs that I discarded, and archived runs from attempts I had to throw away, such as the one where I politely asked the models to reply with one word and they politely ignored me. Jev's share is not itemized here. The budget also left its mark on the data: my API balance hit zero 21 rubric responses before the end of Sonnet's thinking run, which cost me a top-up and a rerun to fill in. If this article saved you from running the same experiment, feedback is very welcome; it is the only reimbursement program available. :')

## References

1. LangChain documentation, *TypeSafe (Jev) integration*: https://docs.langchain.com/oss/python/integrations/providers/typesafe
2. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
3. Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K).
4. Austin, J., et al. (2021). *Program Synthesis with Large Language Models* (MBPP).
5. Li, J., et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.*
6. Wang, Z., et al. (2024). *HelpSteer2: Open-source dataset for training top-performing reward models.*
