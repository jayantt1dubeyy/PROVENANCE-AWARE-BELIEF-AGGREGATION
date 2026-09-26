# CtxLogReg: stronger discriminative baseline — notes

## Question (reviewer request)
Is Bayes only beating weak baselines? Compare against a sufficiently
expressive source x context meta-learner with the same information
budget: per-source logistic regression on (message, context) trained on
the same K = 10 support set, in the reviewer's suggested form
logit = w0 + m*(v0 + v1*a1 + v2*a2 + v12*a1*a2) -- a context-dependent
weight on the message learned from the support set. No generative
convention model, no reliability model, no posterior averaging.

## Protocol
Exact ExpB per-source evaluation (30 episodes x 6 sources, K = 10,
Q = 30, rho ~ U(0.99, 1.0), r ~ U(0.7, 0.95), s_eval = seed*1000+1,
seeds 0-9), numpy-only (IRLS, deterministic), torch-free. Same RNG
stream as results/expB_ablation.csv, so the comparison is
apples-to-apples.

## Results (mean +- 95% CI over 10 seeds)

| lambda | CtxLogReg       |
|--------|-----------------|
| 0.01   | 0.639 +- 0.007  |
| 0.1    | 0.637 +- 0.007  |
| 1.0    | 0.603 +- 0.008  |

Published ExpB comparators: Bayes 0.758 +- 0.009, ContextConditioned
0.730 +- 0.008, MetaAttn (AttentionCalibrator) 0.723 +- 0.008,
SignedScalar 0.625 +- 0.009.

## Reading
- The reviewer's requested stronger baseline reaches 0.637 +- 0.007 --
  below ContextConditioned (0.730) and Bayes (0.758) with non-overlapping
  95% CIs, and barely above SignedScalar (0.625). The result is not a
  regularisation artefact: lambda = 0.01 and 0.1 are identical, and a
  K-sweep (5 seeds) shows the model class is adequate given data
  (K=10: 0.635, K=30: 0.704, K=50: 0.712) -- it is a small-support-data
  limitation.
- Interpretation: at K = 10, learning a context-dependent weight from
  scratch cannot match either the hand-crafted Laplace context-matching
  estimator or the structured Bayesian model. Expressiveness without the
  right inductive bias does not help; the convention structure plus
  uncertainty marginalisation is doing real work beyond a flexible
  discriminative fit.
- A saturated 8-dim P(Y | m, a1, a2) fit was also tried; it overfits badly
  at K = 10 (worse than the reviewer's form) and was dropped.

## Files
- code/ctxlogreg_baseline.py  (--run-all)
- results/expB_ctxlogreg.csv  (columns: seed, lambda, method, acc)
