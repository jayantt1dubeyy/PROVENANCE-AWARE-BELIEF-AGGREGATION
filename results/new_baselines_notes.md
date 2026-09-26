# New baselines/ablations — findings note

Driver: `code/new_baselines.py` (torch-free, numpy-only). Verified bit-identical
reproduction of the published Bayes numbers before running variants
(max abs diff 0.0 vs `results/expB_ablation.csv` and `results/expD_heterogeneity.csv`).
Nothing committed; no `.tex` touched.

## Experiment 1 — EMDecoder baseline (ExpB protocol, seeds 0–9)

Per source, EM fits (convention, rho, r) on the K=10 support set under the §3
generative model (E-step: posterior over the 8 conventions; M-step: closed-form
rho-hat = E[use_conv], r-hat = E[reading correct]; 3 restarts, best observed LL
kept; ≤20 steps, tol 1e-6). Two decision rules from the fit:
- **EMDecoder** (as specified): MAP convention decode + log-odds weight
  log(r-hat/(1-r-hat)).
- **EMDecoder_plugin** (diagnostic): EM point estimates (c*, rho-hat, r-hat)
  plugged into the exact oracle log-odds formula — isolates point-estimate vs
  averaging, holding the decision rule fixed.

| Method | mean ± 95% CI (10 seeds) |
|---|---|
| EMDecoder | 0.683 ± 0.009 |
| EMDecoder_plugin | 0.732 ± 0.007 |
| Bayes (posterior-predictive) | 0.758 ± 0.011 |
| ContextConditioned | 0.730 ± 0.010 |
| MetaAttn+Dec | 0.822 ± 0.006 |
| Oracle | 0.825 ± 0.006 |

MAP-convention recovery rate is only ~0.39 — from K=10 noisy triples EM
frequently commits to the wrong convention, which is precisely what averaging
protects against.

**Implication for §6.6 ("posterior averaging matters"):** supported, and now
decomposable. The 7.5pp Bayes–EMDecoder gap splits into ~4.9pp from dropping
rho (hard decode: 0.732 → 0.683) and ~2.6pp from point-estimate vs averaging
(0.758 → 0.732, non-overlapping CIs). Note EMDecoder_plugin ≈
ContextConditioned (CIs overlap), so a well-fit parametric point estimate
matches the best scalar baseline but still falls short of averaging.

**Suggested framing:** "Replacing posterior-predictive averaging with an EM
point estimate of the same generative model costs 2.6pp (0.758 → 0.732,
non-overlapping 95% CIs); hard MAP-decoding costs a further 4.9pp. The
generative structure alone does not explain Bayes' performance — averaging
over convention uncertainty is doing independent work."

## Experiment 2 — BayesSharedR ablation (ExpD protocol, seeds 0–9)

Identical ExpD protocol (mean-matched E[r]=0.80; homogeneous/moderate/
heterogeneous spreads) with the reliability grid restricted to a single
shared r=0.80 — convention and rho posteriors still inferred, per-source
reliability inference disabled. Same data seeds as published ExpD.

| Condition | BayesSharedR | per-source Bayes (published) | diff |
|---|---|---|---|
| homogeneous (r∈[0.79,0.81]) | 0.728 ± 0.010 | 0.728 ± 0.010 | +0.0002 |
| moderate (r∈[0.70,0.90]) | 0.728 ± 0.010 | 0.727 ± 0.012 | +0.0016 |
| heterogeneous (r∈[0.65,0.95]) | 0.732 ± 0.007 | 0.731 ± 0.008 | +0.0013 |

**Implication for §6.4 ("flatness reflects per-source reliability inference"):**
not supported — report plainly. Disabling per-source reliability inference
changes nothing (all diffs ≤0.002, far inside CIs); both variants are flat.
The ExpD flatness therefore does not come from reliability inference. The
likely mechanism: with per-query decisions, the sign of the predictive
log-odds is driven by convention identification, which is robust to
reliability misspecification — pinning r at its mean costs nothing.

**Suggested framing:** "An ablation pinning reliability to its mean (r=0.80)
reproduces the flat heterogeneity profile exactly (Δ ≤ 0.002), so the
flatness in ExpD should not be attributed to per-source reliability
inference; it reflects robustness of convention identification to
reliability misspecification. We soften the §6.4 mechanism claim
accordingly."

## Files
- `results/sens_emdecoder.csv` — per-seed EMDecoder / EMDecoder_plugin
- `results/sens_sharedr.csv` — per-seed BayesSharedR × 3 conditions
- `code/new_baselines.py` — driver (`--verify`, `--run-all`)
