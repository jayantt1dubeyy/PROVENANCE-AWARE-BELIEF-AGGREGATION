"""Two new baseline/ablation experiments (torch-free, numpy-only).

Experiment 1 (EMDecoder): per-source EM fit of (convention, rho, r) on the
K=10 support set under the section-3 generative model, then MAP-convention
decoding with log-odds weights log(r-hat/(1-r-hat)). Tests whether posterior
*averaging* matters vs just having the right generative structure.
Runs the EXACT ExpB protocol (seeds 0-9, s_eval = seed*1000+1).

Experiment 2 (BayesSharedR): identical ExpD protocol (mean-matched
E[r]=0.80, three heterogeneity spreads, s_eval = seed*10+off) but with the
reliability grid restricted to a single shared r=0.80 (convention/rho
posterior still inferred; per-source reliability inference disabled).

Verification: --verify reproduces the published Bayes ExpB/ExpD numbers
bit-identically before any variant runs.

Usage (from repo root):
  python3 code/new_baselines.py --verify
  python3 code/new_baselines.py --run-all
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

N_CONV = 8
RHO_GRID_DEFAULT = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
R_GRID_DEFAULT = np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])


# ---------------- RNG / generative model (copied verbatim) ----------------
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


# ---------------- Bayes machinery (grids passed explicitly) ----------------
def bayes_post(m_sup, a1_sup, a2_sup, z_sup, candidates, rho_grid, r_grid):
    K = len(m_sup)
    nc = len(candidates)
    A = np.zeros((nc, K))
    for i, c in enumerate(candidates):
        A[i] = (apply_convention(c, m_sup, a1_sup, a2_sup) == z_sup).astype(float)
    B = (m_sup == z_sup).astype(float)

    rho_b = rho_grid[None, :, None, None]
    r_b = r_grid[None, None, :, None]
    A_exp = A[:, None, None, :]
    B_exp = B[None, None, None, :]

    X = rho_b * A_exp + (1 - rho_b) * B_exp
    term = np.clip((1 - r_b) + X * (2 * r_b - 1), 1e-12, None)
    log_lik = np.log(term).sum(axis=-1)

    lp = log_lik - log_lik.max()
    post = np.exp(lp)
    post /= post.sum()
    return post


def bayes_log_odds(post, m_q, a1_q, a2_q, candidates, rho_grid, r_grid):
    nc = len(candidates)
    post = np.asarray(post).reshape(nc, len(rho_grid), len(r_grid))

    A_q = np.array([apply_convention(c, np.array([m_q]),
                                      np.array([a1_q]),
                                      np.array([a2_q]))[0]
                    for c in candidates])
    B_q = 1.0 if m_q == 1 else 0.0

    rho_b = rho_grid[None, :, None]
    r_b = r_grid[None, None, :]

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


def eval_bayes_grid(K, pool, rho_range, rel_range, seed, candidates,
                    rho_grid, r_grid, n_ep=30, Q=30):
    """Identical RNG consumption order to pvi_experiments.eval_bayes."""
    rng = SafeRNG(seed)
    correct, total = 0, 0
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, pool, rho_range, rel_range)
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            post = bayes_post(sm[0], sm[1], sm[2], sm[3], candidates,
                              rho_grid, r_grid)
            for i in range(Q):
                lo = bayes_log_odds(post, int(qm[0][i]), int(qm[1][i]),
                                    int(qm[2][i]), candidates, rho_grid, r_grid)
                pred = 1 if lo > 0 else 0
                correct += int(pred == int(qm[3][i]))
                total += 1
    return correct / total


# ---------------- Experiment 1: EM decoder ----------------
def _em_run(m_sup, a1_sup, a2_sup, z_sup, B, A, init, n_steps, tol):
    K = len(m_sup)
    rho, r = init
    q = np.full(N_CONV, 1.0 / N_CONV)
    prev_ll = -np.inf
    ll = prev_ll
    for _ in range(n_steps):
        rc = min(max(r, 1e-6), 1 - 1e-6)
        rhoc = min(max(rho, 1e-6), 1 - 1e-6)
        # per-(c,k) likelihood
        lik = ((1 - rhoc) * (rc ** B)[None, :] * ((1 - rc) ** (1 - B))[None, :]
               + rhoc * (rc ** A) * ((1 - rc) ** (1 - A)))
        lik = np.clip(lik, 1e-300, None)
        # observed-data log-likelihood (uniform prior over c)
        ll = float(np.logaddexp.reduce(np.log(1.0 / N_CONV) + np.log(lik),
                                       axis=0).sum())
        # E-step
        log_q = np.log(lik).sum(axis=1)
        log_q -= log_q.max()
        q = np.exp(log_q)
        q /= q.sum()
        # expected use_conv and reading-correctness
        w = rhoc * (rc ** A) * ((1 - rc) ** (1 - A)) / lik
        v = (B[None, :] * (1 - rhoc) * (rc ** B)[None, :]
             * ((1 - rc) ** (1 - B))[None, :]
             + A * rhoc * (rc ** A) * ((1 - rc) ** (1 - A))) / lik
        rho = float((q[:, None] * w).sum() / K)
        r = float((q[:, None] * v).sum() / K)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    c_star = int(np.argmax(q))
    return ll, c_star, float(np.clip(rho, 1e-6, 1 - 1e-6)), \
        float(np.clip(r, 1e-6, 1 - 1e-6)), q


def em_fit(m_sup, a1_sup, a2_sup, z_sup, n_steps=20, tol=1e-6):
    """EM for (convention c in 0..7 [discrete], rho, r) on the support set.

    Complete-data view: per triple k, latent use_conv u_k in {0,1} and
    reading-correct e_k in {0,1}; P(m_k | z_k, c, rho, r) =
      (1-rho) * r^B_k (1-r)^(1-B_k) + rho * r^A_ck (1-r)^(1-A_ck),
    with B_k = [m_k==z_k], A_ck = [decode_c(m_k)==z_k].
    E-step: posterior q over the 8 conventions.
    M-step (closed form): rho = E[u], r = E[e].
    Multiple restarts; keeps the highest observed-data log-likelihood.
    Returns (c_star MAP, rho_hat, r_hat, q).
    """
    K = len(m_sup)
    B = (m_sup == z_sup).astype(float)
    A = np.zeros((N_CONV, K))
    for c in range(N_CONV):
        A[c] = (apply_convention(c, m_sup, a1_sup, a2_sup) == z_sup).astype(float)

    best = None
    for init in [(0.9, 0.8), (0.5, 0.6), (0.99, 0.9)]:
        run = _em_run(m_sup, a1_sup, a2_sup, z_sup, B, A, init, n_steps, tol)
        if best is None or run[0] > best[0]:
            best = run
    _, c_star, rho, r, q = best
    return c_star, rho, r, q


def eval_emdecoder(K, pool, rho_range, rel_range, seed, n_ep=30, Q=30):
    """Same RNG consumption order as eval_bayes (identical data)."""
    rng = SafeRNG(seed)
    correct = {'EMDecoder': 0, 'EMDecoder_plugin': 0}
    total = 0
    cstar_hit, cstar_n = 0, 0
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, pool, rho_range, rel_range)
            c_true = src[0]
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            c_star, rho_hat, r_hat, q = em_fit(sm[0], sm[1], sm[2], sm[3])
            cstar_hit += int(c_star == c_true)
            cstar_n += 1
            w = float(np.log(r_hat / (1 - r_hat)))
            for i in range(Q):
                m_q, a1_q, a2_q = int(qm[0][i]), int(qm[1][i]), int(qm[2][i])
                z_q = int(qm[3][i])
                D = int(apply_convention(
                    c_star, np.array([m_q]), np.array([a1_q]),
                    np.array([a2_q]))[0])
                lo = w * (2 * D - 1)
                correct['EMDecoder'] += int((1 if lo > 0 else 0) == z_q)
                lo2 = oracle_log_odds(m_q, a1_q, a2_q, c_star, rho_hat, r_hat)
                correct['EMDecoder_plugin'] += int((1 if lo2 > 0 else 0) == z_q)
                total += 1
    accs = {k: v / total for k, v in correct.items()}
    return accs, cstar_hit / cstar_n


def run_exp1_emdecoder(seeds):
    rows = []
    for seed in seeds:
        s_eval = seed * 1000 + 1
        accs, cstar = eval_emdecoder(10, None, (0.99, 1.0), (0.7, 0.95),
                                     s_eval)
        for method, acc in accs.items():
            rows.append({'seed': seed, 'method': method, 'acc': acc})
        print(f'  seed {seed}: EMDecoder={accs["EMDecoder"]:.4f} '
              f'plugin={accs["EMDecoder_plugin"]:.4f} '
              f'MAP-conv-hit={cstar:.3f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('results/sens_emdecoder.csv', index=False)
    return df


# ---------------- Experiment 2: BayesSharedR (ExpD protocol) ----------------
REL_CONDITIONS = {
    'homogeneous': (0.79, 0.81),
    'moderate': (0.70, 0.90),
    'heterogeneous': (0.65, 0.95),
}
REL_OFFSET = {'homogeneous': 0, 'moderate': 1, 'heterogeneous': 2}


def run_exp2_sharedr(seeds):
    rows = []
    r_shared = np.array([0.80])
    for seed in seeds:
        for name, rel_range in REL_CONDITIONS.items():
            off = REL_OFFSET[name]
            s_eval = seed * 10 + off
            acc = eval_bayes_grid(10, None, (0.99, 1.0), rel_range, s_eval,
                                  list(range(N_CONV)), RHO_GRID_DEFAULT,
                                  r_shared)
            rows.append({'seed': seed, 'rel': name, 'method': 'BayesSharedR',
                         'acc': acc})
        print(f'  seed {seed} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('results/sens_sharedr.csv', index=False)
    return df


# ---------------- verification ----------------
def verify():
    ok = True
    print('Verifying ExpB Bayes reproduction...', flush=True)
    df = pd.read_csv('results/expB_ablation.csv')
    pub = df[df.method == 'Bayes'].set_index('seed')['acc']
    diffs = []
    for seed in range(10):
        acc = eval_bayes_grid(10, None, (0.99, 1.0), (0.7, 0.95),
                              seed * 1000 + 1, list(range(N_CONV)),
                              RHO_GRID_DEFAULT, R_GRID_DEFAULT)
        diffs.append(abs(acc - pub.loc[seed]))
    print(f'  max abs diff vs published ExpB Bayes: {max(diffs):.2e}')
    ok &= max(diffs) == 0.0

    print('Verifying ExpD Bayes reproduction...', flush=True)
    dfd = pd.read_csv('results/expD_heterogeneity.csv')
    diffs = []
    for seed in range(10):
        for name, rel_range in REL_CONDITIONS.items():
            s_eval = seed * 10 + REL_OFFSET[name]
            acc = eval_bayes_grid(10, None, (0.99, 1.0), rel_range, s_eval,
                                  list(range(N_CONV)), RHO_GRID_DEFAULT,
                                  R_GRID_DEFAULT)
            pub_acc = dfd[(dfd.seed == seed) & (dfd.rel == name)
                          & (dfd.method == 'Bayes')]['acc'].iloc[0]
            diffs.append(abs(acc - pub_acc))
    print(f'  max abs diff vs published ExpD Bayes: {max(diffs):.2e}')
    ok &= max(diffs) == 0.0
    print('VERIFICATION:', 'PASS (bit-identical)' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--run-all', action='store_true')
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if verify() else 1)
    if args.run_all:
        if not verify():
            print('Refusing to run: verification failed', flush=True)
            sys.exit(1)
        t0 = time.time()
        print('Experiment 1: EMDecoder (ExpB protocol)', flush=True)
        run_exp1_emdecoder(list(range(10)))
        print(f'Exp1 done ({time.time()-t0:.0f}s)', flush=True)
        t0 = time.time()
        print('Experiment 2: BayesSharedR (ExpD protocol)', flush=True)
        run_exp2_sharedr(list(range(10)))
        print(f'Exp2 done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
