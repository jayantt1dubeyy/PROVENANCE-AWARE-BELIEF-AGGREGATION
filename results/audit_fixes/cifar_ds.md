# CIFAR-10H Dawid–Skene audit fix — summary

**Files:** fixed script `code/audit_fixes/real_world_cifar10h_ds.py` (copy of
`code/real_world_cifar10h.py`; the original is untouched), per-image outputs
`results/audit_fixes/real_world_cifar10h.csv` and
`results/audit_fixes/real_world_cifar10h_5judg.csv`, numbers in
`results/audit_fixes/cifar_ds.json`.
**Data:** `cifar10h-raw.zip` downloaded from
https://github.com/jcpeterson/cifar-10h (the paper's documented source),
extracted to `data/cifar10h-raw.csv`, passed via `CIFAR10H_CSV`.
**Runtime:** 20 s wall for the whole study (both regimes).

## What changed in the code (function level)

Only `predict_dawid_skene` changed (plus `DS_ITERS` 50 → 100, an explicit
`DS_TOL = 1e-6`, an audit note in the docstring, and output paths redirected
to `results/audit_fixes/`).

- **Before:** the label posterior was computed **once** from vote fractions
  and never updated; the loop only recomputed confusion matrices from that
  frozen posterior (identical from iteration 2 onward), and predictions came
  from a separate uniform-prior plug-in pass. That is a one-shot estimator,
  not Dawid–Skene EM — yet the paper calls it "transductive Dawid–Skene EM".
- **After (genuine DS EM):** posterior initialised from vote fractions;
  each iteration runs an **M-step** (confusion matrices from the current
  posterior, keeping the original estimation-half counts as pseudocounts in
  every M-step, so the warm start is unchanged) followed by an **E-step**
  (each image's posterior recomputed from the current confusion matrices
  and the class prevalence estimated from the current posterior). Iterates
  until max |Δcmat| < 1e-6 (cap 100). Predictions are the argmax of the
  final posterior.

Every other method, the seed, and the 50/50 split are untouched — and they
prove it: MajorityVote, AccuracyWeighted, BayesDirichlet and Oracle
reproduce the banked per-image correctness **exactly** (0 disagreements in
both regimes).

## New numbers vs banked (5,000 eval images)

Full regime (~51 judgments/image; EM converged in 5 iterations):

| method | banked | fixed run |
|---|---|---|
| MajorityVote | 0.9924 | 0.9924 |
| AccuracyWeighted | 0.9926 | 0.9926 |
| BayesDirichlet | 0.9930 | 0.9930 |
| DawidSkene | 0.9928 (one-shot, mislabelled) | **0.9928 (genuine EM)** |
| Oracle | 0.9960 (+0.0036 vs MV) | 0.9960 (+0.0036 vs MV) |

5-judgment regime (EM converged in 92 iterations):

| method | banked | fixed run |
|---|---|---|
| MajorityVote | 0.9844 | 0.9844 |
| AccuracyWeighted | 0.9846 | 0.9846 |
| BayesDirichlet | 0.9842 | 0.9842 |
| DawidSkene | 0.9846 (one-shot, mislabelled) | **0.9850 (genuine EM)** |
| Oracle | 0.9878 (+0.0034 vs MV) | 0.9878 (+0.0034 vs MV) |

**Interpretation.** In the full regime genuine EM lands on exactly the same
predictions as the one-shot column (0 per-image disagreements): with ~51
judgments per image and the estimation-half pseudocounts, the posterior
barely moves. In the 5-judgment regime EM flips 2 images, both wrong→right
(+0.0004). So the paper's CIFAR-10H null result survives the fix — genuine
Dawid–Skene still adds nothing over BayesDirichlet here (full regime: DS
0.9928 vs BD 0.9930, the same −0.0002 the audit noted) — but the claim
"Dawid–Skene adds nothing" is now supported by an actual Dawid–Skene
implementation, which it was not before.

## Descriptive statistics — verdict: all four VERIFIED

Computed from the raw CSV. The file has 539,910 rows including 25,710
attention checks (10 per annotator); excluding them, exactly as the study
loader does:

| statistic | paper | computed from raw data |
|---|---|---|
| total judgments | 514,200 | **514,200** ✓ |
| annotators | 2,571 | **2,571** ✓ |
| judgments per annotator | exactly 200 | **exactly 200 for every annotator** (210 trials each incl. attention checks) ✓ |
| mean worker accuracy | 0.949 | **0.9487** ✓ |

The numbers the audit marked UNVERIFIED are all correct as printed.
