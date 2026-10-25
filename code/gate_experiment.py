"""Experiment I: the informativeness gate (torch-free).

For each source, test H0: M _|_ Y | A on the K support triples with an
exact permutation test (alpha=0.05). Aggregate only sources that reject;
a source that fails to reject contributes log-odds 0 (prior). If no
source is kept, predict by fair coin (matches the paper's sign(0) rule).

The permutation null (M,A) _|_ Y is exact here: by design A _|_ Y, and H0
asserts M _|_ Y | A.

Mirrors eval_bayes (per-source decoding accuracy, 30 episodes x 6
sources x Q=30 queries) so results are directly comparable to the
banked Experiment A numbers. Runs both Bayes and GatedBayes on the
IDENTICAL stream (same sources/support/queries) for paired comparison.

Usage: python3 code/gate_experiment.py  (run from repo root)
Writes: results/expI_gate.csv
"""
import os
import sys
import time
import random
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- verbatim numpy machinery from pvi_experiments.py ----------

class SafeRNG:
    """Deterministic RNG wrapper around random.Random."""
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


N_CONV = 8


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


RHO_GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
R_GRID = np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])


def bayes_post(m_sup, a1_sup, a2_sup, z_sup, candidates):
    """Posterior over (c, rho, r). Shape: (len(candidates), |RHO_GRID|, |R_GRID|)."""
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

    lp = log_lik - log_lik.max()
    post = np.exp(lp)
    post /= post.sum()
    return post


def bayes_log_odds(post, m_q, a1_q, a2_q, candidates):
    """Posterior-predictive log-odds for a query given the posterior."""
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


# ---------- the informativeness gate ----------

def _batch_cond_mi(counts, K):
    """Plug-in I(M;Y|A) for a batch of (B,16) count tables.

    Index j = a*4 + m*2 + y, a=(a1,a2) in 0..3.
    I = sum_{a,m,y} (n_amy/K) log( n_amy*n_a / (n_am*n_ay) ).
    """
    B = counts.shape[0]
    n_amy = counts.reshape(B, 4, 2, 2)
    n_a = n_amy.sum(axis=(2, 3), keepdims=True)
    n_am = n_amy.sum(axis=3, keepdims=True)
    n_ay = n_amy.sum(axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(n_amy > 0, n_amy * n_a / (n_am * n_ay + 1e-300), 1.0)
        contrib = np.where(n_amy > 0, (n_amy / K) * np.log(ratio), 0.0)
    return contrib.sum(axis=(1, 2, 3))


def mi_gate_keep(m_sup, a1_sup, a2_sup, z_sup, seed, alpha=0.05, B=199):
    """Exact permutation test of H0: M _|_ Y | A. Returns (keep, p_value)."""
    m_sup = np.asarray(m_sup, dtype=np.int64)
    a = (np.asarray(a1_sup, dtype=np.int64) * 2
         + np.asarray(a2_sup, dtype=np.int64))
    y = np.asarray(z_sup, dtype=np.int64)
    K = len(m_sup)
    base = a * 4 + m_sup * 2  # + y gives the joint index

    j0 = base + y
    c0 = np.bincount(j0, minlength=16).reshape(1, 16).astype(float)
    i_obs = _batch_cond_mi(c0, K)[0]

    g = np.random.default_rng(seed)
    Yp = np.stack([g.permutation(y) for _ in range(B)])          # (B, K)
    Jp = base[None, :] + Yp                                      # (B, K)
    Cb = (Jp[:, :, None] == np.arange(16)[None, None, :]).sum(axis=1).astype(float)
    i_perm = _batch_cond_mi(Cb, K)

    p = (1.0 + float((i_perm >= i_obs).sum())) / (1.0 + B)
    return bool(p < alpha), float(p)


# ---------- evaluation (mirrors eval_bayes, paired) ----------

def run_paired(K, rho_range, rel_range, seed, n_ep=30, Q=30,
               candidates=None, alpha=0.05):
    """Run Bayes and GatedBayes on the identical stream.

    Returns dict with accuracies and the fraction of sources kept.
    """
    if candidates is None:
        candidates = list(range(N_CONV))
    rng = SafeRNG(seed)
    res = {'bayes': [0, 0], 'gated': [0, 0], 'kept': [0, 0]}
    for ep in range(n_ep):
        for s in range(6):
            src = sample_source(rng, None, rho_range, rel_range)
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            m_sup, a1_sup, a2_sup, z_sup = sm
            post = bayes_post(m_sup, a1_sup, a2_sup, z_sup, candidates)
            keep, _ = mi_gate_keep(m_sup, a1_sup, a2_sup, z_sup,
                                   seed=seed * 100003 + ep * 1009 + s * 7,
                                   alpha=alpha)
            res['kept'][0] += int(keep)
            res['kept'][1] += 1
            for i in range(Q):
                lo = bayes_log_odds(post, int(qm[0][i]), int(qm[1][i]),
                                    int(qm[2][i]), candidates)
                pred_b = 1 if lo > 0 else (rng.random() < 0.5)
                res['bayes'][0] += int(pred_b == int(qm[3][i]))
                res['bayes'][1] += 1
                if keep:
                    pred_g = pred_b
                else:
                    pred_g = (rng.random() < 0.5)
                res['gated'][0] += int(pred_g == int(qm[3][i]))
                res['gated'][1] += 1
    return {
        'bayes': res['bayes'][0] / res['bayes'][1],
        'gated': res['gated'][0] / res['gated'][1],
        'keep_frac': res['kept'][0] / res['kept'][1],
    }


def main():
    t0 = time.time()
    seeds = list(range(30))
    rhos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for seed in seeds:
        for rho in rhos:
            r = run_paired(10, (rho - 0.01, rho + 0.01), (0.7, 0.95), seed)
            rows.append({'seed': seed, 'rho': rho, 'method': 'Bayes',
                         'acc': r['bayes']})
            rows.append({'seed': seed, 'rho': rho, 'method': 'GatedBayes',
                         'acc': r['gated'], 'keep_frac': r['keep_frac']})
        print(f'  seed {seed} done ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'expI_gate.csv')
    df.to_csv(out, index=False)

    piv = df.groupby(['rho', 'method'])['acc'].mean().unstack()
    print('\n' + piv.round(4).to_string())
    print('\nkeep_frac by rho:')
    print(df[df['method'] == 'GatedBayes'].groupby('rho')['keep_frac'].mean().round(3).to_string())
    # paired deltas
    b = df[df['method'] == 'Bayes'].set_index(['seed', 'rho'])['acc']
    g = df[df['method'] == 'GatedBayes'].set_index(['seed', 'rho'])['acc']
    d = (g - b).groupby('rho')
    print('\npaired GatedBayes - Bayes (mean, 95% t-CI):')
    for rho, vals in d:
        v = vals.values
        m = v.mean()
        se = v.std(ddof=1) / np.sqrt(len(v))
        from scipy import stats as _st
        h = se * _st.t.ppf(0.975, len(v) - 1)
        print(f'  rho={rho}: {m:+.4f} [{m-h:+.4f}, {m+h:+.4f}]')
    print(f'\nTOTAL {time.time()-t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
