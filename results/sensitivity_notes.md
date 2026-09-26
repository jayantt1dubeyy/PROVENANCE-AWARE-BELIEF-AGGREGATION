# Sensitivity & confidence-interval notes (26 Sep 2026)

Supplements the deliberately-deferred analyses: 95% CIs over the ten seeds,
grid-resolution sensitivity of the Bayesian aggregator, and prior-strength
(temperature) sensitivity. Paper `.tex` files were NOT modified.

## 1. 95% confidence intervals over 10 seeds

t(0.975, df=9) = 2.262. Note the paper's tables report mean ± sample **std**,
not CIs; CIs are ~0.72× narrower. "Overlap" below means the 95% CIs overlap.

### ExpA — Bayes rho sweep
| rho | mean ± 95% CI |
|---|---|
| 0.5 | 0.625 ± 0.008 |
| 0.6 | 0.640 ± 0.007 |
| 0.7 | 0.659 ± 0.007 |
| 0.8 | 0.678 ± 0.007 |
| 0.9 | 0.714 ± 0.008 |
| 1.0 | 0.755 ± 0.009 |

Claim "Bayes climbs monotonically 0.625→0.755": HOLDS (adjacent CIs do not overlap).
At rho=1.0: Bayes 0.755±0.009 vs SignedScalar 0.621±0.009, MetaAttn 0.696±0.020,
MetaMean 0.711±0.012 — Bayes CI strictly above all: HOLDS.

### ExpB — ablation
| method | mean ± 95% CI |
|---|---|
| ContentOnly | 0.537 ± 0.024 |
| UnsignedScalar | 0.550 ± 0.004 |
| SignedScalar | 0.625 ± 0.010 |
| ContextConditioned | 0.730 ± 0.010 |
| AttentionCalibrator | 0.722 ± 0.009 |
| MetaAttn+Dec | 0.822 ± 0.006 |
| MetaAttn+DecCtx | 0.822 ± 0.006 |
| Bayes | 0.758 ± 0.011 |
| Oracle | 0.825 ± 0.006 |

- "MetaAttn+Dec 0.822 ≈ Oracle 0.825, within noise": HOLDS (CIs overlap).
- "Bayes 0.758 below oracle ceiling": HOLDS (Bayes CI strictly below Oracle CI).
- "ContextConditioned (0.730) edges out MetaAttn (0.722)": CIs OVERLAP — the edge
  is a point-estimate statement only, not statistically significant at 95%.
  The paper's wording ("edges out") is modest and does not claim significance,
  so this is a caveat, not a refutation.

### ExpC — K sweep
| K | BayesUniv | BayesFam |
|---|---|---|
| 1 | 0.588 ± 0.010 | 0.647 ± 0.013 |
| 2 | 0.644 ± 0.007 | 0.687 ± 0.011 |
| 5 | 0.711 ± 0.010 | 0.741 ± 0.010 |
| 10 | 0.767 ± 0.007 | 0.786 ± 0.007 |
| 20 | 0.793 ± 0.008 | 0.805 ± 0.007 |
| 50 | 0.820 ± 0.004 | 0.822 ± 0.004 |

Claim "Bayes 0.65→0.82, converging to the E[r]=0.825 ceiling": HOLDS.
Fam > Univ at small K (non-overlapping), CIs overlap at K=50 — consistent with
convergence.

### ExpD — heterogeneity (Bayes "flat ~0.73")
| condition | mean ± 95% CI |
|---|---|
| homogeneous | 0.728 ± 0.010 |
| moderate | 0.727 ± 0.012 |
| heterogeneous | 0.731 ± 0.008 |

HOLDS — all three CIs mutually overlap; no evidence of a heterogeneity effect.

### ExpE — consensus at N=12
| method | mean ± 95% CI |
|---|---|
| Bayes | 0.955 ± 0.014 |
| NaiveMean | 0.573 ± 0.044 |
| MajorityVote | 0.573 ± 0.044 |
| MetaAttn | 0.937 ± 0.024 |
| Oracle | 0.995 ± 0.006 |

Claim "+38.2pp over naive, 4.0pp off oracle": HOLDS (Bayes CI strictly above
naive, strictly below oracle).

### ExpF — posterior diagnostic
| metric | mean ± 95% CI |
|---|---|
| top-1 convention accuracy | 0.828 ± 0.029 |
| mass > 0.5 on true convention | 0.722 ± 0.023 |

Consistent with the paper's 0.828±0.040 / 0.722±0.032 (sample stds).

### ExpG — misspecification
| method | mean ± 95% CI |
|---|---|
| BayesFull | 0.756 ± 0.009 |
| BayesHalf | 0.685 ± 0.008 |
| in-candidate | 0.767 ± 0.011 |
| out-of-candidate | 0.599 ± 0.014 |
| MetaAttn | 0.731 ± 0.007 |
| Oracle | 0.824 ± 0.005 |

- "7pp drop 0.756→0.685": HOLDS (no CI overlap).
- "in-candidate 0.767 vs 0.756 within noise": HOLDS (overlap).
- "out-of-candidate 0.599, still well above chance": HOLDS.

## 2. Grid-resolution sensitivity

Driver: `code/sens_bayes.py` (torch-free, copies the pure-numpy Bayes path
verbatim; default-grid output is **bit-identical** to `results/*.csv` Bayes
columns, verified max abs diff = 0.0). Only Bayes-dependent quantities were
re-run — neural baselines do not touch the grid.

| grid | rho points | r points | cells |
|---|---|---|---|
| default | 0.5–1.0 step 0.1 (6) | 0.55–0.95 step 0.05 (9) | 8×6×9 = 432 |
| coarse | 0.5, 0.7, 0.9 (3) | 0.6, 0.7, 0.8, 0.9 (4) | 8×3×4 = 96 |
| fine | 0.5–1.0 step 0.05 (11) | 0.55–0.95 step 0.025 (17) | 8×11×17 = 1496 |

Results (mean ± 95% CI, 10 seeds):

**ExpA Bayes sweep** (rho 0.5→1.0):
- default: 0.625±0.008, 0.640±0.007, 0.659±0.007, 0.678±0.007, 0.714±0.008, 0.755±0.009
- coarse:  0.628±0.007, 0.640±0.006, 0.657±0.007, 0.675±0.007, 0.710±0.008, 0.752±0.009
- fine:    0.625±0.008, 0.640±0.007, 0.659±0.007, 0.679±0.007, 0.714±0.007, 0.756±0.009

**ExpB Bayes**: default 0.758±0.011, coarse 0.755±0.010, fine 0.758±0.011.

**ExpE at N=12** (Bayes / NaiveMean / Oracle):
- default: 0.955±0.014 / 0.573±0.044 / 0.995±0.006
- coarse:  0.955±0.018 / 0.573±0.044 / 0.995±0.006
- fine:    0.958±0.016 / 0.573±0.044 / 0.995±0.006

Verdict: **STABLE.** Even a 4.5× coarser grid shifts every headline number by
≤0.004, inside the CIs. Monotonicity (ExpA), the Bayes–oracle gap (ExpB),
and the +38pp consensus margin (ExpE) are unchanged. Grid resolution is not
a driver of any reported conclusion.

## 3. Prior-strength (likelihood-temperature) sensitivity

The paper's posterior is uniform-prior over the grid. Temperature T scales
the log-likelihood: post ∝ lik^(1/T). T=1 is the paper; T<1 sharpens the
posterior (weaker effective prior); T>1 flattens it toward the uniform prior
(stronger prior).

**ExpA Bayes sweep** (rho 0.5→1.0):
- T=0.5: 0.617±0.008, 0.636±0.007, 0.661±0.009, 0.682±0.007, 0.717±0.007, 0.758±0.007
- T=1.0: 0.625±0.008, 0.640±0.007, 0.659±0.007, 0.678±0.007, 0.714±0.008, 0.755±0.009
- T=2.0: 0.629±0.007, 0.643±0.006, 0.654±0.008, 0.671±0.008, 0.704±0.008, 0.745±0.009
- T=5.0: 0.635±0.007, 0.641±0.004, 0.644±0.009, 0.653±0.008, 0.674±0.007, 0.707±0.010

**ExpB Bayes**: T=0.5 → 0.758±0.010; T=1 → 0.758±0.011; T=2 → 0.745±0.010;
T=5 → 0.704±0.009.

**ExpE at N=12, Bayes vs NaiveMean 0.573±0.044 vs Oracle 0.995±0.006**:
- T=0.5 → 0.950±0.013; T=1 → 0.955±0.014; T=2 → 0.947±0.021; T=5 → 0.918±0.028.

Verdict: **STABLE within a wide band, graceful degradation at the extreme.**
T=0.5 and T=2.0 leave every headline conclusion intact (ordering Bayes >
naive baselines, Bayes < oracle, monotonic ExpA sweep). Only at T=5.0
(a pathologically strong uniform prior) do magnitudes attenuate: ExpB Bayes
falls to 0.704 (still above SignedScalar 0.625, below oracle 0.825) and the
ExpE oracle gap widens from ~4.0pp to ~7.7pp while the +34.5pp margin over
naive survives. The paper's quantitative claims ("4.0pp off oracle") are
calibrated to the stated uniform-prior model; they are not prior-fragile.

## Runtimes (CPU, numpy-only Bayes paths, 10 seeds each)

| run | wall time |
|---|---|
| default-grid replication (A+B+E) | 38 s |
| grid coarse | 52 s |
| grid fine | 65 s |
| prior T=0.5 / 2.0 / 5.0 | ~38 s each |

No paper `.tex` files were modified and no PDF was rebuilt.
