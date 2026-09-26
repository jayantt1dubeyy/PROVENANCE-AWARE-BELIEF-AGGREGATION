"""Bayes-only sensitivity driver (torch-free).

Replicates the Bayes columns of ExpA / ExpB / ExpE under alternative
(a) latent-convention grids and (b) likelihood temperatures (prior strength).

The Bayesian functions are copied verbatim from code/pvi_experiments.py
(pure-numpy path; identical RNG usage) so default-grid output must match
results/*.csv exactly.

Usage: python3 code/sens_bayes.py --grid default|coarse|fine --temp 1.0
       --tag myrun   (run from the repository root)
Writes: results/sens_<tag>_expA.csv, results/sens_<tag>_expB.csv,
        results/sens_<tag>_expE.csv
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ---------------- configuration (set by CLI) ----------------
RHO_GRID = None
R_GRID = None
TEMP = 1.0  # likelihood temperature: post ~ lik^(1/T); T>1 = stronger prior

# ---------------- copied numpy-only machinery ----------------
N_CONV = 8


class SafeRNG:
    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def random(self, size=None):
        if size is None:
            return float(self._rng.random())
        return np.array([self._rng.random() for _ in range(int(size))])

    def uniform(self, low, high, size=None):
        low, high = float(low), float(high)
        if size is None:
            return float(self._rng.uniform(low, high))
        return np.array([self._rng.uniform(low, high) for _ in range(int(size))])

    def randint(self, low, high, size=None):
        low, high = int(low), int(high)
        if size is None:
            return int(self._rng.randint(low, high - 1))
        return np.array([self._rng.randint(low, high - 1) for _ in range(int(size))])

    def choice(self, arr):
        return arr[self._rng.randrange(len(arr))]


def apply_convention(c, signal, a1, a2):
    if c == 0: return signal
    if c == 1: return 1 - signal
    if c == 2: return signal ^ a1
    if c == 3: return signal ^ a2
    if c == 4: return signal ^ (a1 & a2)
    if c == 5: return signal ^ (a1 | a2)
    if c == 6: return signal ^ (a1 ^ a2)
    if c == 7: return signal ^ (1 - (a1 & a2))
    raise ValueError(f'Unknown convention {c}')


def sample_source(rng, pool=None, rho_range=(0.6, 1.0), rel_range=(0.7, 0.95)):
    c = rng.choice(pool) if pool is not None else rng.randint(0, N_CONV)
    rho = float(rng.uniform(*rho_range))
    r = float(rng.uniform(*rel_range))
    return int(c), float(rho), float(r)


def generate(src, n, rng):
    c, rho, r = src
    Z = rng.randint(0, 2, size=n)
    a1 = rng.randint(0, 2, size=n)
    a2 = rng.randint(0, 2, size=n)
    correct = rng.random(n) < r
    R = np.where(correct, Z, 1 - Z)
    use_conv = rng.random(n) < rho
    M = np.where(use_conv, apply_convention(c, R, a1, a2), R)
    return (M.astype(np.int64), a1.astype(np.int64),
            a2.astype(np.int64), Z.astype(np.int64))


def bayes_post(m_sup, a1_sup, a2_sup, z_sup, candidates):
    K = len(m_sup)
    nc = len(candidates)
    A = np.zeros((nc, K))
    for i, c in enumerate(candidates):
        A[i] = (apply_convention(c, m_sup, a1_sup, a2_sup) == z_sup).astype(float)
    B = (m_sup == z_sup).astype(float)

    rho_b = RHO_GRID[None, :, None, None]
    r_b = R_GRID[None, None, :, None]
    A_exp = A[:, None, None, :]
    B_exp = B[None, None, None, :]

    X = rho_b * A_exp + (1 - rho_b) * B_exp
    term = np.clip((1 - r_b) + X * (2 * r_b - 1), 1e-12, None)
    log_lik = np.log(term).sum(axis=-1)

    lp = (log_lik - log_lik.max()) / TEMP
    post = np.exp(lp)
    post /= post.sum()
    return post


def bayes_log_odds(post, m_q, a1_q, a2_q, candidates):
    nc = len(candidates)
    post = np.asarray(post).reshape(nc, len(RHO_GRID), len(R_GRID))

    A_q = np.array([apply_convention(c, np.array([m_q]),
                                      np.array([a1_q]),
                                      np.array([a2_q]))[0]
                    for c in candidates])
    B_q = 1.0 if m_q == 1 else 0.0

    rho_b = RHO_GRID[None, :, None]
    r_b = R_GRID[None, None, :]

    X1 = rho_b * A_q[:, None, None] + (1 - rho_b) * B_q
    p1 = (1 - r_b) + X1 * (2 * r_b - 1)

    X0 = rho_b * (1 - A_q)[:, None, None] + (1 - rho_b) * (1 - B_q)
    p0 = (1 - r_b) + X0 * (2 * r_b - 1)

    p1_bar = float((post * p1).sum())
    p0_bar = float((post * p0).sum())
    return float(np.log(max(p1_bar, 1e-12)) - np.log(max(p0_bar, 1e-12)))


def oracle_log_odds(m_q, a1_q, a2_q, c, rho, r):
    A_q = float(apply_convention(c, np.array([m_q]),
                                 np.array([a1_q]),
                                 np.array([a2_q]))[0])
    B_q = 1.0 if m_q == 1 else 0.0
    X1 = rho * A_q + (1 - rho) * B_q
    p1 = (1 - r) + X1 * (2 * r - 1)
    X0 = rho * (1 - A_q) + (1 - rho) * (1 - B_q)
    p0 = (1 - r) + X0 * (2 * r - 1)
    return float(np.log(max(p1, 1e-12)) - np.log(max(p0, 1e-12)))


def eval_bayes(K, pool, rho_range, rel_range, seed, n_ep=30, Q=30,
               candidates=None):
    if candidates is None:
        candidates = list(range(N_CONV))
    rng = SafeRNG(seed)
    correct, total = 0, 0
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, pool, rho_range, rel_range)
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            post = bayes_post(sm[0], sm[1], sm[2], sm[3], candidates)
            for i in range(Q):
                lo = bayes_log_odds(post, int(qm[0][i]), int(qm[1][i]),
                                     int(qm[2][i]), candidates)
                pred = 1 if lo > 0 else 0
                correct += int(pred == int(qm[3][i]))
                total += 1
    return correct / total


def consensus_bayes_only(n_agents, n_trials=60, K=10, seed=0):
    """ExpE consensus loop without the neural method (torch-free)."""
    rng = SafeRNG(seed)
    methods = ['NaiveMean', 'MajorityVote', 'Bayes', 'Oracle']
    correct = {m: 0 for m in methods}
    for _ in range(n_trials):
        Z_true = int(rng.random() < 0.5)
        raw_msgs, params, supports, queries = [], [], [], []
        for _a in range(n_agents):
            src = sample_source(rng, None, (0.7, 1.0), (0.7, 0.95))
            c, rho, r = src
            sm = generate(src, K, rng)
            a1_q = int(rng.random() < 0.5)
            a2_q = int(rng.random() < 0.5)
            correct_read = rng.random() < r
            sig = Z_true if correct_read else 1 - Z_true
            if rng.random() < rho:
                m_q = int(apply_convention(c, np.array([sig]),
                                            np.array([a1_q]),
                                            np.array([a2_q]))[0])
            else:
                m_q = sig
            raw_msgs.append(m_q)
            params.append((c, rho, r))
            supports.append(sm)
            queries.append((m_q, a1_q, a2_q))
        correct['NaiveMean'] += int((np.mean(raw_msgs) > 0.5) == Z_true)
        correct['MajorityVote'] += int((np.sum(raw_msgs) > n_agents / 2) == Z_true)
        lo_total = 0.0
        for j in range(n_agents):
            sup = supports[j]
            q = queries[j]
            post = bayes_post(sup[0], sup[1], sup[2], sup[3],
                               list(range(N_CONV)))
            lo_total += bayes_log_odds(post, int(q[0]), int(q[1]),
                                        int(q[2]), list(range(N_CONV)))
        correct['Bayes'] += int((lo_total > 0) == Z_true)
        lo_oracle = 0.0
        for j in range(n_agents):
            q = queries[j]
            c_j, rho_j, r_j = params[j]
            lo_oracle += oracle_log_odds(int(q[0]), int(q[1]),
                                         int(q[2]), c_j, rho_j, r_j)
        correct['Oracle'] += int((lo_oracle > 0) == Z_true)
    return {k: v / n_trials for k, v in correct.items()}


# ---------------- sweeps ----------------
def run_expA(seeds):
    rows = []
    for seed in seeds:
        for rho in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            acc = eval_bayes(10, None, (rho - 0.01, rho + 0.01),
                             (0.7, 0.95), seed * 100 + int(rho * 100))
            rows.append({'seed': seed, 'rho': rho, 'method': 'Bayes', 'acc': acc})
    return pd.DataFrame(rows)


def run_expB(seeds):
    rows = []
    for seed in seeds:
        acc = eval_bayes(10, None, (0.99, 1.0), (0.7, 0.95), seed * 1000 + 1)
        rows.append({'seed': seed, 'method': 'Bayes', 'acc': acc})
    return pd.DataFrame(rows)


def run_expE(seeds):
    rows = []
    for seed in seeds:
        for n_agents in [2, 3, 4, 6, 8, 12]:
            res = consensus_bayes_only(n_agents=n_agents, n_trials=60, K=10,
                                       seed=seed * 100 + n_agents)
            for method, acc in res.items():
                rows.append({'seed': seed, 'n_agents': n_agents,
                             'method': method, 'acc': acc})
    return pd.DataFrame(rows)


GRIDS = {
    'default': (np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])),
    'coarse': (np.array([0.5, 0.7, 0.9]),
               np.array([0.6, 0.7, 0.8, 0.9])),
    'fine': (np.arange(0.5, 1.0001, 0.05),
             np.arange(0.55, 0.9501, 0.025)),
}


def main():
    global RHO_GRID, R_GRID, TEMP
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='default', choices=list(GRIDS))
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    RHO_GRID, R_GRID = GRIDS[args.grid]
    TEMP = args.temp
    tag = args.tag or f"{args.grid}_T{args.temp:g}"
    print(f'grid={args.grid} ({len(RHO_GRID)}x{len(R_GRID)} cells), TEMP={TEMP}, tag={tag}', flush=True)
    seeds = list(range(10))
    os.makedirs('results', exist_ok=True)

    t0 = time.time()
    dfA = run_expA(seeds)
    dfA.to_csv(f'results/sens_{tag}_expA.csv', index=False)
    print(f'ExpA done ({time.time()-t0:.0f}s)', flush=True)

    t0 = time.time()
    dfB = run_expB(seeds)
    dfB.to_csv(f'results/sens_{tag}_expB.csv', index=False)
    print(f'ExpB done ({time.time()-t0:.0f}s)', flush=True)

    t0 = time.time()
    dfE = run_expE(seeds)
    dfE.to_csv(f'results/sens_{tag}_expE.csv', index=False)
    print(f'ExpE done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
