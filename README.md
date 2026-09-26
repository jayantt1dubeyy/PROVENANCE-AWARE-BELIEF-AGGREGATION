# Provenance-Aware Belief Aggregation in Multi-Agent Communication

Code and data for the paper *"Provenance-Aware Belief Aggregation in
Multi-Agent Communication Under Convention Heterogeneity"* (Jayant Dubey).

## Overview

Multi-agent systems commonly assume that messages can be aggregated using
source-reliability weights. This assumption fails when sources use different
reporting conventions — mappings from internal readings to messages that can
invert, condition on context, or otherwise transform the underlying signal.

This repository contains:

- Structural limitation theorems (nonnegative-weight insufficiency, scalar
  insufficiency under context-dependent conventions, equal-reliability
  optimum, posterior consistency) and their proofs.
- A finite-grid Bayesian latent-convention aggregator.
- Eight experiments (A–H) plus representation ablations.
- All baselines: content-only, unsigned scalar, signed scalar,
  context-conditioned scalar, context-dependent logistic regression
  (CtxLogReg), attention calibrator, mean-pool calibrator,
  EM decoder (MAP point estimate), Bayesian aggregator, and
  reliability-weighted oracle.

## Quick start

Run from the **repository root** so that `results/` and `figures/` are
created at the top level.

```bash
# Install dependencies
pip install -r requirements.txt

# Verify core components (~10 seconds)
python code/smoke_test.py

# Check the posterior-predictive fix against brute-force enumeration
python code/test_predictive_fix.py

# Run the full experimental suite (10 seeds; ~2 h on T4 GPU, ~5-6 h on CPU)
python code/run_all_n10.py

# Regenerate LaTeX table rows and figures from the CSVs
python code/gen_tables.py
python code/make_figures.py

# Supplementary: decision accuracy conditioned on convention recovery
python code/cond_acc_analysis.py
```

Results are written to `results/*.csv`. Figures are written to
`figures/*.png` and `figures/*.pdf`. The compiled manuscript is at
`paper/main.pdf`; sources are in `paper/sections/`.

## Repository structure

```
.
├── code/
│   ├── pvi_experiments.py     # Full experimental suite (experiments A–G)
│   ├── run_all_n10.py         # Run the full suite with N_SEEDS=10
│   ├── run_cg_n10.py          # Re-run experiments C–G only (e.g. after an interrupted run)
│   ├── run_B_to_F.py          # Re-run experiments B–F only
│   ├── run_B_only.py          # Re-run experiment B only
│   ├── new_baselines.py       # EMDecoder and BayesSharedR baselines
│   ├── ctxlogreg_baseline.py  # CtxLogReg: per-source context-dependent logistic weights
│   ├── exp_correlated.py      # Experiment H: correlated source errors (independence misspecification)
│   ├── sens_bayes.py          # Grid-resolution and likelihood-temperature sensitivity
│   ├── ci_check.py            # 95% CI checks over the ten seeds
│   ├── cond_acc_analysis.py   # Supplementary: P(correct decision | convention recovery)
│   ├── gen_tables.py          # Generate LaTeX table rows from results CSVs
│   ├── make_figures.py        # Figure generation (PNG + PDF)
│   ├── smoke_test.py          # Standalone smoke test
│   └── test_predictive_fix.py # Brute-force test of the posterior-predictive log-odds
├── results/                   # CSVs produced by the suite (10 seeds)
├── figures/                   # PNG and PDF figures
├── paper/
│   ├── main.pdf               # Compiled manuscript
│   ├── main.tex               # LaTeX manuscript
│   ├── references.bib         # Bibliography
│   └── sections/              # Section files
├── notebooks/
│   └── pvi_colab.ipynb        # Colab runner
├── requirements.txt
├── environment.yml
├── REPRODUCIBILITY.md
└── README.md
```

## Conventions

The benchmark uses eight involutive conventions:

| ID | Convention            | Context-dependent? |
|----|-----------------------|:------------------:|
| 0  | identity              | No                 |
| 1  | NOT                   | No                 |
| 2  | XOR with a1           | Yes                |
| 3  | XOR with a2           | Yes                |
| 4  | XOR with (a1 AND a2)  | Yes                |
| 5  | XOR with (a1 OR a2)   | Yes                |
| 6  | XOR with (a1 XOR a2)  | Yes                |
| 7  | XOR with NOT (a1 AND a2) | Yes             |

All eight satisfy the involution property f_c(f_c(x, a), a) = x, which the
Bayesian aggregator relies on to recover the underlying reading from an
observed message.

## Reproducing the paper

The paper reports results with `N_SEEDS = 10`. To reproduce the submitted
tables exactly, run `python code/run_all_n10.py` from the repository root,
then `python code/gen_tables.py` and `python code/make_figures.py` to
regenerate the LaTeX table rows and figures from the new CSVs.

## Reproducibility

- **Randomness:** all sampling uses Python's `random.Random` wrapped in a
  `SafeRNG` class. Each experiment seeds its own RNG via `SafeRNG(seed)` at
  the top of the experiment loop. PyTorch seeds are set via
  `torch.manual_seed(seed)` before every model initialisation.
- **Seeds:** per-experiment seeds 0 through 9.
- **Determinism of the Bayesian aggregator:** the grid over (ρ, r) is
  fixed and enumerated in full. No randomness.
- **Smoke tests:** `python code/smoke_test.py` verifies posterior
  normalization, posterior shape, finite log-odds, binary predictions,
  above-chance accuracy on a clean synthetic case, and RNG reproducibility.
- **Posterior-predictive fix:** `code/test_predictive_fix.py` checks the
  corrected posterior-predictive log-odds against explicit enumeration.
  See REPRODUCIBILITY.md for details of the correction.
- **CUDA nondeterminism:** not required for reproducibility. The suite
  produces the same numbers on CPU and GPU.

See REPRODUCIBILITY.md for the full checklist.

## Compute

- Reference environment: Google Colab T4 GPU
- Peak memory: < 2 GB
- Runtime (10 seeds): ~2 hours on T4, ~5–6 hours on CPU (observed Sep 2026)

## Citation

```bibtex
@inproceedings{dubey2026provenance,
  title={Provenance-Aware Belief Aggregation in Multi-Agent Communication
         Under Convention Heterogeneity},
  author={Dubey, Jayant},
  booktitle={International Conference on Autonomous Agents and Multiagent
             Systems (AAMAS)},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
