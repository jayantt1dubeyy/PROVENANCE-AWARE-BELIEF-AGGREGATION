# Reproducibility Checklist

Following AAMAS reproducibility guidelines.

## 1. Scope

All experimental results in the paper can be reproduced by running:

```bash
cd code
python pvi_experiments.py
python make_figures.py
```

The reported tables and figures correspond to the CSVs in results/ and the
PNGs/PDFs in figures/.

2. Randomness

· Per-experiment seeds: 0–9 (ten seeds). The canonical pipeline is
  code/run_all_n10.py, which runs every experiment at n = 10.
· RNG implementation: Python's random.Random throughout, wrapped in
  the SafeRNG class. This avoids known issues with numpy.Generator on
  recent Python/NumPy combinations and guarantees cross-platform determinism.
· PyTorch: torch.manual_seed(seed) is called before every model
  initialization via the make_model factory. No cuDNN kernels requiring
  deterministic flags are used; results are identical on CPU and GPU.
· Bayesian aggregator: fully deterministic. The grid over (rho, r) is
  fixed and enumerated in full. No sampling.
· Torch-free sensitivity drivers (code/ci_check.py, code/sens_bayes.py,
  code/new_baselines.py, code/exp_correlated.py,
  code/ctxlogreg_baseline.py) reimplement the pure-NumPy Bayes path and were
  verified bit-identical to the published Bayes columns (max abs diff
  0.0) before running variants. The NumPy ContextConditioned /
  SignedScalar reimplementations in code/exp_correlated.py reproduce the
  published ExpB columns to 1e-8 (decision-identical; residual is
  float32-vs-float64 episode-mean aggregation noise).

3. Data

No external datasets. All data is generated at runtime from the source model
described in Section 3 of the paper. Generation parameters (reliability,
stability, and convention ranges) are specified in pvi_experiments.py.

4. Environment

· Python 3.9+
· numpy >= 1.21, < 2.0
· torch >= 1.13
· pandas >= 1.3
· matplotlib >= 3.5
· seaborn >= 0.12
· scipy >= 1.7

Tested on:

· Google Colab (Python 3.10, numpy 1.24, torch 2.0)
· Local Linux (Python 3.11, numpy 1.26, torch 2.1)

5. Computational resources

· 10-seed run (code/run_all_n10.py): a few hours on a CPU-only machine;
  neural calibrators dominate runtime.
· Torch-free sensitivity sweeps (code/sens_bayes.py,
  code/new_baselines.py, code/exp_correlated.py,
  code/ctxlogreg_baseline.py): minutes on CPU.
· Peak memory: <2 GB

6. Statistical methodology

Ten seeds per experiment. Tables report mean ± standard deviation;
Appendix B.2.1 additionally reports 95% confidence intervals
(t_0.975,9 = 2.262). The statistics are exploratory: enough to separate
stable gaps from seed noise, not confirmatory. Overlapping CIs are
reported as indistinguishable (e.g. ContextConditioned vs MetaAttn in
Experiment B).

7. Baselines

All baselines receive identical information in every experiment. Specifically:

· NaiveMean and MajorityVote see only raw messages.
· NoSupport, UnsignedScalar, SignedScalar, ContextConditioned,
  MetaMean, MetaAttn see messages plus context and the support set
  where applicable.
· Bayes receives the same support set as the neural calibrators.
· Oracle knows the true (c, rho, r) and applies the Bayes-optimal rule.
· MetaAttn+Dec / MetaAttn+DecCtx (Experiment B only) are diagnostic
  variants that additionally receive privileged features: the message
  bit decoded with the true convention (Dec), plus a one-hot encoding
  of the true convention index (DecCtx). They test whether the
  bottleneck is representational, not whether such features are
  available in practice.
· EMDecoder (Experiment B protocol, seeds 0–9; code/new_baselines.py):
  fits (convention, rho, r) per source by EM on the K = 10 support set
  under the Section 3 generative model, then decides with the MAP
  convention and a plug-in log-odds weight. A diagnostic variant
  (EMDecoder_plugin) plugs the EM point estimates into the exact oracle
  log-odds formula, isolating point-estimate vs posterior-averaging
  with the decision rule held fixed.
· BayesSharedR (Experiment D protocol, seeds 0–9; code/new_baselines.py):
  identical to the Experiment D Bayes run except the reliability grid
  is restricted to a single shared r = 0.80, disabling per-source
  reliability inference while convention and stability are still
  inferred. Results: results/sens_emdecoder.csv, results/sens_sharedr.csv.
· CtxLogReg (Experiment B protocol, seeds 0–9;
  code/ctxlogreg_baseline.py): per-source L2 logistic regression on the
  same K = 10 support triples, logit = w0 + m*(v0 + v1*a1 + v2*a2 +
  v12*a1*a2) (a context-dependent weight on the message, fit by
  deterministic IRLS). Lambda sweep {0.01, 0.1, 1.0}; headline
  lambda = 0.1. Results: results/expB_ctxlogreg.csv,
  notes in results/expB_ctxlogreg_notes.md.
· Experiment H: correlated sources (consensus protocol, N = 10 in five
  pairs, K = 10, seeds 0–9; code/exp_correlated.py): within-pair
  query-time errors correlated ~0.5 via a calibrated Gaussian copula
  (copula parameter 0.8; achieved binary error correlation 0.44–0.52),
  paired-seed independence control arm. Compares Bayes,
  ContextConditioned, SignedScalar, naive majority vote, and Oracle
  (all NumPy). MetaAttn not re-run (torch unavailable). Results:
  results/expH_correlated.csv, notes in results/expH_notes.md.

8. Oracle definition

Oracle: knows the ground-truth parameters (c_s, rho_s, r_s) for every
source and applies the Bayes-optimal aggregation rule directly. No learning;
no inference. Used as a reference upper bound in all tables.

This is stated identically in paper/sections/formal.tex,
paper/sections/experiments.tex, and the code (experiment_B,
experiment_E).

9. Smoke tests

python smoke_test.py verifies:

1. Posterior shape is (|C|, |RHO_GRID|, |R_GRID|).
2. Posterior normalizes to 1.
3. Log-odds is finite.
4. Prediction is binary.
5. Bayesian accuracy exceeds 0.7 on a clean synthetic case.
6. RNG reproduces exactly under a fixed seed.

These run in under ten seconds and are also executed at the start of main()
in pvi_experiments.py.

10. Known limitations

· The Bayesian grid is finite. Continuous rho and r would require a
  different inference procedure.
· The convention family is fixed at 8 members. Adaptive expansion is not
  implemented.
· Sources are assumed conditionally independent given Y.
· The benchmark is synthetic. Deployment on real multi-agent systems is
  future work.

These match Section 8 of the paper.

11. Reporting of negative results

· An early bilinear-interaction hypothesis was falsified during
  development: all models produced identical outputs on held-out
  conventions, indicating an identifiability failure rather than an
  architectural one. This negative result informs the choice of the
  finite-grid Bayesian aggregator over neural alternatives (see
  paper Section 6).

12. Independent reproduction notes

An independent reproduction pass (September 2026, CPU, torch 2.14)
found and fixed the following before the results reported in the paper
were generated:

· Neural models crashed on integer input tensors (Long vs Float dtype);
  the models now cast inputs to float at the forward boundary.
· ContextConditioned initially collapsed to the signed-scalar solution
  (identical predictions): its MLP saw only the scalar support
  consistency, not per-context statistics, so it could not learn
  context-dependent signs. It was reimplemented to estimate the
  context-specific agreement rate P(m == z | context) from
  context-matching support examples (Laplace-smoothed) and set the
  weight sign from it; the reimplemented baseline reaches 0.736
  (vs 0.616 signed), restoring the intended representation hierarchy.
· eval_neural omitted the batch dimension expected by the models.
· The Experiment E oracle voted with the sources' unobserved true
  readings; it now applies the exact Bayes-optimal rule with the known
  (c, rho, r) on the observed messages, consistent with the definition
  in Section 8.
· The K = 0 support-size condition was evaluated with one support
  example (the encoders require a nonempty support set); the sweep now
  starts at K = 1.
· The MetaAttn+Dec / MetaAttn+DecCtx representation-ablation variants
  described in the paper were missing from the code and were
  re-implemented as documented in Section 7 above.
· The Bayesian predictor computed the posterior *expectation* of the
  parameter-conditional log-odds, E_post[log P(M|Y=1,theta)/P(M|Y=0,theta)],
  rather than the exact posterior-predictive log-odds,
  log(E_post[P(M|Y=1,theta)]/E_post[P(M|Y=0,theta)]). The two differ by
  Jensen's inequality; only the latter is the Bayes rule. bayes_log_odds
  was corrected to the predictive form, verified against brute-force
  enumeration (code/test_predictive_fix.py), and all experiments were
  re-run from scratch with the corrected estimator.

All numbers, tables, and figures in the paper are generated from the
corrected code by running code/run_all_n10.py followed by
code/make_figures.py from the repository root.

13. Contact

For questions about reproduction, open an issue on the repository.
