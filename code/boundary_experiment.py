"""Experiment I: the stability boundary (torch-free).

Characterizes when per-source Bayesian decoding helps vs hurts relative
to the population-average mapping (analytic equivalent of NoSupport):
sweeps rho x K, 30 seeds, paired comparison.

Writes: results/expI_boundary.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from gate_experiment import (SafeRNG, sample_source, generate,
                             bayes_post, bayes_log_odds, N_CONV)
from adaptive_experiment import POP_LO, CANDS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'results')


def run_cell(K, rho, seed, n_ep=20, Q=30):
    rng = SafeRNG(seed)
    cb = cp = tot = 0
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, None, (rho - 0.01, rho + 0.01),
                                (0.7, 0.95))
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            post = bayes_post(sm[0], sm[1], sm[2], sm[3], CANDS)
            for i in range(Q):
                m, a1, a2, z = (int(qm[0][i]), int(qm[1][i]),
                                int(qm[2][i]), int(qm[3][i]))
                pb = 1 if bayes_log_odds(post, m, a1, a2, CANDS) > 0 else 0
                pp = 1 if POP_LO[(m, a1, a2)] > 0 else 0
                cb += (pb == z); cp += (pp == z); tot += 1
    return cb / tot, cp / tot


def main():
    t0 = time.time()
    seeds = list(range(30))
    rhos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Ks = [2, 5, 10, 20, 50]
    rows = []
    for seed in seeds:
        for K in Ks:
            for rho in rhos:
                # offset seed per cell for independent streams
                b, p = run_cell(K, rho, seed * 1000 + K * 37 + int(rho * 100))
                rows.append({'seed': seed, 'K': K, 'rho': rho,
                             'method': 'Bayes', 'acc': b})
                rows.append({'seed': seed, 'K': K, 'rho': rho,
                             'method': 'PopMap', 'acc': p})
        print(f'  seed {seed} done ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'expI_boundary.csv')
    df.to_csv(out, index=False)
    piv = df.groupby(['K', 'rho', 'method'])['acc'].mean().unstack()
    print('\n' + piv.round(4).to_string())
    print(f'\nTOTAL {time.time()-t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
