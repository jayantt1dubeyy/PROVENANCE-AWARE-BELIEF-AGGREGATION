"""Experiment I: adaptive strategy selection (torch-free).

At low convention stability (rho~0.5), per-source Bayesian decoding loses
to the population-average mapping (analytic equivalent of NoSupport).
This script implements global split-half CV strategy selection:
fit per-source Bayes posteriors on half the pooled support, compare
held-out log-likelihood vs the fixed population mapping, and use the
winner on queries.

Writes: results/expI_adaptive.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from gate_experiment import (SafeRNG, apply_convention, sample_source,
                             generate, bayes_post, bayes_log_odds,
                             RHO_GRID, N_CONV)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'results')


def build_pop_lo():
    """Analytic population mapping: log-odds table over (m,a1,a2).

    P(Y=1|m,a) averaged over c~U(8), rho~U(0.5,1), r~U(0.7,0.95)
    (the training distribution of NoSupport). Matches NoSupport's
    banked 0.6485 at rho=0.5 within noise (analytic: 0.654).
    """
    def pop_log_odds(m, a1, a2):
        A = np.array([apply_convention(c, m, a1, a2) for c in range(N_CONV)])
        rhos = np.linspace(0.5, 1.0, 11)
        rs = np.linspace(0.7, 0.95, 6)
        p1 = ((1 - rs)[None, None, :]
              + (rhos[:, None] * A[:, None, None] + (1 - rhos[:, None]) * m)
              * (2 * rs[None, None, :] - 1))
        p0 = ((1 - rs)[None, None, :]
              + (rhos[:, None] * (1 - A[:, None, None])
                 + (1 - rhos[:, None]) * (1 - m))
              * (2 * rs[None, None, :] - 1))
        return float(np.log(p1.mean() / p0.mean()))
    return {(m, a1, a2): pop_log_odds(m, a1, a2)
            for m in (0, 1) for a1 in (0, 1) for a2 in (0, 1)}


POP_LO = build_pop_lo()
CANDS = list(range(N_CONV))


def cv_choose(supports, K, rng):
    """Split-half CV: returns True to use Bayes, False for population map."""
    h = K // 2
    ll_b = 0.0
    ll_p = 0.0
    for (m_s, a1_s, a2_s, z_s) in supports:
        post = bayes_post(m_s[:h], a1_s[:h], a2_s[:h], z_s[:h], CANDS)
        for i in range(h, K):
            m, a1, a2, z = (int(m_s[i]), int(a1_s[i]),
                            int(a2_s[i]), int(z_s[i]))
            lo_b = bayes_log_odds(post, m, a1, a2, CANDS)
            ll_b += z * lo_b - np.logaddexp(0, lo_b)
            lo_p = POP_LO[(m, a1, a2)]
            ll_p += z * lo_p - np.logaddexp(0, lo_p)
    return ll_b > ll_p


def run_episode(K, rho_range, rel_range, rng, Q=30):
    """One episode: 6 sources. Returns (acc_bayes, acc_pop, acc_adapt, chose_bayes)."""
    supports = []
    queries = []
    for _ in range(6):
        src = sample_source(rng, None, rho_range, rel_range)
        supports.append(generate(src, K, rng))
        queries.append(generate(src, Q, rng))

    use_bayes = cv_choose(supports, K, rng)

    # fit full posteriors for the Bayes predictions
    posts = [bayes_post(m_s, a1_s, a2_s, z_s, CANDS)
             for (m_s, a1_s, a2_s, z_s) in supports]

    cb = cp = ca = tot = 0
    for (post, qm) in zip(posts, queries):
        for i in range(Q):
            m, a1, a2, z = (int(qm[0][i]), int(qm[1][i]),
                            int(qm[2][i]), int(qm[3][i]))
            pb = 1 if bayes_log_odds(post, m, a1, a2, CANDS) > 0 else 0
            pp = 1 if POP_LO[(m, a1, a2)] > 0 else 0
            pa = pb if use_bayes else pp
            cb += (pb == z); cp += (pp == z); ca += (pa == z); tot += 1
    return cb / tot, cp / tot, ca / tot, int(use_bayes)


def main():
    t0 = time.time()
    seeds = list(range(30))
    rhos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    K = 10
    rows = []
    for seed in seeds:
        rng = SafeRNG(seed)
        for rho in rhos:
            accs = []
            ch = []
            for ep in range(30):
                b, p, a, c = run_episode(K, (rho - 0.01, rho + 0.01),
                                        (0.7, 0.95), rng)
                accs.append((b, p, a)); ch.append(c)
            b = np.mean([x[0] for x in accs]); p = np.mean([x[1] for x in accs])
            a = np.mean([x[2] for x in accs])
            rows.append({'seed': seed, 'rho': rho, 'method': 'Bayes', 'acc': b})
            rows.append({'seed': seed, 'rho': rho, 'method': 'PopMap', 'acc': p})
            rows.append({'seed': seed, 'rho': rho, 'method': 'Adaptive', 'acc': a,
                         'chose_bayes': np.mean(ch)})
        print(f'  seed {seed} done ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'expI_adaptive.csv')
    df.to_csv(out, index=False)
    print('\n' + df.groupby(['rho', 'method'])['acc'].mean().unstack().round(4).to_string())
    print('\nP(choose Bayes) by rho:')
    print(df[df['method'] == 'Adaptive'].groupby('rho')['chose_bayes'].mean().round(2).to_string())
    print(f'\nTOTAL {time.time()-t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
