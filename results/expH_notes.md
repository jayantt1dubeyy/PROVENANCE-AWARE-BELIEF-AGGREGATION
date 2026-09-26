# Experiment H: correlated sources — notes

## Question
The Bayesian aggregator assumes sources are conditionally independent
given the truth. Does violating that assumption hurt Bayes
disproportionately, or does convention inference still dominate?

## Design
- Consensus-style protocol (cf. Experiment E): N = 10 sources (5 pairs)
  report on one shared truth Z per query; group decision by summing
  per-source evidence. Correlation only matters under combination, so a
  per-source protocol (ExpB) could not test this.
- Otherwise ExpB data-generating process: 8-convention family,
  rho ~ U(0.99, 1.0), r ~ U(0.7, 0.95), K = 10 support per source,
  seeds 0-9, n_trials = 60, Q = 10 queries per trial (600 group
  decisions per seed).
- Correlated arm: within-pair *query-time errors* correlated via a
  Gaussian copula. Copula parameter 0.8 was calibrated by simulation to
  yield achieved binary error correlation ~0.5 (dichotomisation at the
  per-source error thresholds attenuates; measured 0.44-0.52 across
  seeds, mean ~0.47). Independence arm: same protocol, independent
  errors (control). Paired seeds, so per-seed deltas are meaningful.
- Support data are independent in both arms: the misspecification is
  isolated to the combination step, exactly where the model assumes
  factorisation.

## Methods (all numpy, torch-free)
Bayes (per-source grid posterior, summed posterior-predictive log-odds),
ContextConditioned (numpy reimplementation; global temperature factors out
of the combination sign), SignedScalar (numpy reimplementation; global
tanh(raw_w) > 0 factors out), naive (majority vote of raw messages),
Oracle (true (c, rho, r), summed oracle log-odds -- also
independence-assuming, hence misspecified in the correlated arm; reference
only). MetaAttn could not be re-run: torch model, torch unavailable here.

## Verification
The numpy ContextConditioned / SignedScalar reimplementations were checked
against published ExpB per-source numbers (results/expB_ablation.csv):
max abs diff 8.26e-09 over 10 seeds x 2 methods. Not bit-identical only
because the published numbers average 30 per-episode float32 means
(torch) while this script accumulates one float64 correct/total; the
decision rule is identical.

## Results (mean +- 95% CI over 10 seeds; deltas paired per seed)

| method             | independence   | correlated     | delta (paired)   |
|--------------------|----------------|----------------|------------------|
| Bayes              | 0.971 +- 0.005 | 0.958 +- 0.006 | -0.013 +- 0.008  |
| ContextConditioned | 0.962 +- 0.006 | 0.950 +- 0.005 | -0.012 +- 0.006  |
| SignedScalar       | 0.847 +- 0.009 | 0.844 +- 0.014 | -0.004 +- 0.017  |
| naive              | 0.449 +- 0.013 | 0.457 +- 0.018 | +0.008 +- 0.015  |
| Oracle             | 0.994 +- 0.002 | 0.983 +- 0.004 | -0.011 +- 0.003  |

## Reading
- Correlation costs every method ~1-1.3 points, *including the Oracle*.
  The drop is the problem getting harder (correlated errors = less
  independent evidence), not Bayes-specific misspecification damage:
  Bayes's paired delta (-0.013 +- 0.008) matches ContextConditioned
  (-0.012 +- 0.006) and the Oracle (-0.011 +- 0.003).
- Ranking is preserved in both arms:
  Bayes > ContextConditioned > SignedScalar > naive.
- Answer: the independence misspecification does **not** hurt Bayes
  disproportionately; convention inference still dominates. This is now
  an out-of-model test: the model is no longer exactly the
  data-generating process, and the headline ordering survives.

## Files
- code/exp_correlated.py  (--verify, --run-all)
- results/expH_correlated.csv  (columns: seed, arm, method, acc)
