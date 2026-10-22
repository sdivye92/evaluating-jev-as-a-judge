== Binary correctness ==
false-accept = incorrect marked correct; false-reject = correct marked incorrect
AUROC / ECE are shown only for judges that return graded probabilities
source    judge                            n  err        accuracy [95% CI]    FAR    FRR    MCC  AUROC    ECE
gsm8k     (length-only baseline AUROC)   400                          0.51
gsm8k     jev-1.13.0                     400    0     80.8% [77.0, 84.8]  23.5%  15.0%   0.62   0.89  0.055   (majority 50%)
gsm8k     claude-haiku-5-5:system1       400    0     62.7% [59.2, 66.2]  12.5%  62.0%   0.29   0.66  0.322   (majority 50%)
gsm8k     claude-sonnet-5-5:system1      400    0     76.2% [72.0, 80.0]  12.5%  35.0%   0.54   0.78  0.226   (majority 50%)
gsm8k     claude-haiku-5-5:reasoning     397    3     98.5% [97.2, 99.5]   0.0%   3.0%   0.97      –      –   (majority 50%)
gsm8k     claude-sonnet-5-5:reasoning    400    0     98.8% [97.5, 99.8]   0.0%   2.5%   0.98      –      –   (majority 50%)
halueval  (length-only baseline AUROC)   400                          0.51
halueval  jev-1.13.0                     400    0     93.8% [91.0, 95.8]  10.0%   2.5%   0.88   0.98  0.021   (majority 50%)
halueval  claude-haiku-5-5:system1       400    0     91.5% [88.5, 94.5]   8.0%   9.0%   0.83   0.93  0.078   (majority 50%)
halueval  claude-sonnet-5-5:system1      400    0     93.8% [91.2, 96.0]   4.0%   8.5%   0.88   0.94  0.061   (majority 50%)
halueval  claude-haiku-5-5:reasoning     400    0     94.8% [92.2, 97.0]   6.0%   4.5%   0.90      –      –   (majority 50%)
halueval  claude-sonnet-5-5:reasoning    399    1     95.7% [93.5, 97.5]   5.0%   3.5%   0.91      –      –   (majority 50%)
mbpp      (length-only baseline AUROC)   300                          0.50
mbpp      jev-1.13.0                     300    0     84.0% [80.3, 87.7]  14.0%  18.0%   0.68   0.94  0.066   (majority 50%)
mbpp      claude-haiku-5-5:system1       300    0     80.0% [76.0, 84.0]   8.0%  32.0%   0.62   0.82  0.184   (majority 50%)
mbpp      claude-sonnet-5-5:system1      300    0     91.7% [88.7, 94.7]   3.3%  13.3%   0.84   0.93  0.078   (majority 50%)
mbpp      claude-haiku-5-5:reasoning     296    4     98.3% [96.9, 99.7]   0.0%   3.4%   0.97      –      –   (majority 51%)
mbpp      claude-sonnet-5-5:reasoning    299    1     99.7% [99.0, 100.0]   0.0%   0.7%   0.99      –      –   (majority 50%)
ALL       (length-only baseline AUROC)  1100                          0.50
ALL       jev-1.13.0                    1100    0     86.4% [84.5, 88.4]  16.0%  11.3%   0.73   0.94  0.019   (majority 50%)
ALL       claude-haiku-5-5:system1      1100    0     77.9% [75.7, 80.2]   9.6%  34.5%   0.58   0.81  0.196   (majority 50%)
ALL       claude-sonnet-5-5:system1     1100    0     86.8% [84.7, 88.9]   6.9%  19.5%   0.74   0.88  0.125   (majority 50%)
ALL       claude-haiku-5-5:reasoning    1093    7     97.1% [96.0, 98.1]   2.2%   3.7%   0.94      –      –   (majority 50%)
ALL       claude-sonnet-5-5:reasoning   1098    2     97.9% [97.0, 98.7]   1.8%   2.4%   0.96      –      –   (majority 50%)

== Paired differences, all binary items pooled: jev-1.13.0 minus each other judge ==
95% CI from a bootstrap that resamples whole questions and recomputes both judges on the same resample.
Accuracy: negative = base is worse. False-accept / false-reject: positive = base is worse.  * = CI excludes 0
vs                               n                   accuracy               false-accept               false-reject
claude-haiku-5-5:system1      1100    +8.5 pts [+6.1, +10.7]*     +6.4 pts [+2.9, +9.6]*  -23.3 pts [-27.1, -19.1]*
claude-sonnet-5-5:system1     1100     -0.5 pts [-2.6, +2.0]     +9.1 pts [+5.8, +12.0]*    -8.2 pts [-12.0, -4.4]*
claude-haiku-5-5:reasoning    1093   -10.7 pts [-12.7, -8.8]*  +13.8 pts [+10.7, +16.7]*    +7.6 pts [+4.8, +10.4]*
claude-sonnet-5-5:reasoning   1098   -11.6 pts [-13.5, -9.6]*  +14.2 pts [+11.1, +17.1]*    +8.9 pts [+6.4, +11.5]*

-- Latency, binary (ms per judged item, as seen by the client; includes network and concurrency) --
judge                           mean     p50     p95     p99  calls  tokens in/out  errors
jev-1.13.0                       345     309     506     612    1.0     469/20           0
claude-haiku-5-5:system1        1564    1430    2456    4909    5.0    2671/60           0
claude-sonnet-5-5:system1       2082    1844    3399    8635    5.0    2671/60           0
claude-haiku-5-5:reasoning      6414    3911   19603   46512    1.0     535/980          7
claude-sonnet-5-5:reasoning     6597    4439   15933   42641    1.0     535/537          2

== Rubric scoring (HelpSteer2 human ratings 0-4) ==
criterion    judge                            n   exact    MAE    QWK  Spearman  predicted-level histogram
helpfulness  majority-class baseline        300   42.3%   1.07
helpfulness  jev-1.13.0                     300   33.0%   0.99   0.44      0.46  [14, 56, 89, 108, 33] (gold [22, 22, 38, 91, 127])
helpfulness  claude-haiku-5-5:system1       300   31.7%   0.91   0.46      0.43  [6, 53, 77, 119, 45] (gold [22, 22, 38, 91, 127])
helpfulness  claude-sonnet-5-5:system1      298   29.5%   1.06   0.41      0.43  [22, 65, 93, 91, 27] (gold [21, 22, 38, 91, 126])
helpfulness  claude-haiku-5-5:reasoning     300   32.0%   1.02   0.41      0.44  [23, 48, 108, 96, 25] (gold [22, 22, 38, 91, 127])
helpfulness  claude-sonnet-5-5:reasoning    298   29.5%   1.05   0.40      0.43  [19, 61, 101, 96, 21] (gold [21, 22, 38, 91, 126])
correctness  majority-class baseline        300   47.3%   0.97
correctness  jev-1.13.0                     300   31.0%   1.00   0.40      0.44  [10, 48, 82, 110, 50] (gold [17, 27, 29, 85, 142])
correctness  claude-haiku-5-5:system1       300   32.3%   1.00   0.41      0.40  [18, 44, 75, 113, 50] (gold [17, 27, 29, 85, 142])
correctness  claude-sonnet-5-5:system1      298   32.6%   1.09   0.40      0.46  [35, 60, 69, 78, 56] (gold [16, 27, 29, 85, 141])
correctness  claude-haiku-5-5:reasoning     300   33.7%   0.96   0.41      0.41  [15, 35, 88, 101, 61] (gold [17, 27, 29, 85, 142])
correctness  claude-sonnet-5-5:reasoning    297   37.7%   0.89   0.47      0.47  [18, 45, 75, 100, 59] (gold [16, 27, 29, 85, 140])
coherence    majority-class baseline        300   73.3%   0.35
coherence    jev-1.13.0                     300   51.7%   0.60   0.30      0.29  [0, 3, 29, 116, 152] (gold [2, 3, 14, 61, 220])
coherence    claude-haiku-5-5:system1       300   49.3%   0.65   0.33      0.31  [0, 17, 42, 99, 142] (gold [2, 3, 14, 61, 220])
coherence    claude-sonnet-5-5:system1      298   37.6%   0.79   0.30      0.35  [3, 23, 39, 143, 90] (gold [2, 3, 14, 61, 218])
coherence    claude-haiku-5-5:reasoning     300   46.0%   0.62   0.30      0.30  [0, 4, 34, 143, 119] (gold [2, 3, 14, 61, 220])
coherence    claude-sonnet-5-5:reasoning    298   50.7%   0.55   0.38      0.35  [0, 6, 27, 132, 133] (gold [2, 3, 14, 61, 218])

== Paired differences in rubric MAE: jev-1.13.0 minus each other judge ==
Per-item absolute error on the expected score; positive = base is worse.  * = 95% bootstrap CI excludes 0
vs                               n                helpfulness                correctness                  coherence         mean of 3 criteria
claude-haiku-5-5:system1       300      +0.07 [+0.01, +0.13]*      +0.01 [-0.06, +0.07]       -0.06 [-0.12, -0.00]*      +0.01 [-0.04, +0.05] 
claude-sonnet-5-5:system1      298      -0.09 [-0.15, -0.03]*      -0.09 [-0.17, -0.02]*      -0.20 [-0.25, -0.13]*      -0.13 [-0.17, -0.07]*
claude-haiku-5-5:reasoning     300      -0.03 [-0.09, +0.03]       +0.04 [-0.02, +0.11]       -0.02 [-0.08, +0.03]       -0.00 [-0.04, +0.04] 
claude-sonnet-5-5:reasoning    297      -0.08 [-0.14, -0.01]*      +0.10 [+0.04, +0.17]*      +0.04 [-0.01, +0.10]       +0.02 [-0.02, +0.07] 

-- Latency, rubric (ms per judged item, as seen by the client; includes network and concurrency) --
judge                           mean     p50     p95     p99  calls  tokens in/out  errors
jev-1.13.0                       355     314     511     611    1.0    1101/49           0
claude-haiku-5-5:system1        2981    2089    8506    9542   15.0   17282/135          0
claude-sonnet-5-5:system1       2438    2279    3501    5296   15.0   17309/135          2
claude-haiku-5-5:reasoning     17550   14904   40940   54696    3.0    3459/5982         0
claude-sonnet-5-5:reasoning    15108   12681   31949   69703    3.0    3462/3059         3