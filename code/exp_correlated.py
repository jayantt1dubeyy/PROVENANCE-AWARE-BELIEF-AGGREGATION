"""Experiment H: correlated sources (out-of-model robustness test).

The Bayesian aggregator assumes sources are conditionally independent given
the truth. This experiment breaks that assumption on purpose: sources are
paired, and within each pair the query-time *errors* are correlated via a
Gaussian copula calibrated so the achieved within-pair binary error
correlation is ~0.5 (copula parameter 0.8; dichotomisation attenuates),
follows the ExpB data-generating process (same 8-convention family,
rho ~ U(0.99, 1.0), r ~ U(0.7, 0.95), K = 10 support per source,
seeds 0-9).

Because correlation only matters when sources are *combined*, the protocol
is consensus-style (cf. Experiment E): N = 10 sources (5 pairs) report on
one shared truth Z per query; each method sums per-source evidence into a
single group decision. Two arms are run with paired seeds:

  independence : errors independent (control)
  correlated   : within-pair error correlation ~0.5 (misspecified arm)

Support data stay independent in both arms, so the misspecification is
isolated to the combination step -- exactly where the model assumes
factorisation.

Methods (all numpy, torch-free):
  Bayes               per-source grid posterior + sum of posterior-predictive log-odds
  ContextConditioned  per-source Laplace context-matching + sum of logits
                      (global temperature factors out of the sign; dropped)
  SignedScalar        per-source support sign + sum of signed messages
                      (global tanh(raw_w) > 0 factors out; verified below)
  naive               majority vote of raw messages
  Oracle              true (c, rho, r) + sum of oracle log-odds
                      (also independence-assuming, hence misspecified in the
                      correlated arm -- reported as reference only)

MetaAttn could not be re-evaluated under this protocol: it is a torch
model and torch is unavailable in this environment. Its ExpB accuracy
under independence was 0.722 +/- 0.013.

Verification (--verify): the numpy ContextConditioned / SignedScalar /
naive reimplementations are checked against the published ExpB per-source
numbers (results/expB_ablation.csv) before any correlated run.

Usage (from repo root):
  python3 code/exp_correlated.py --verify
  python3 code/exp_correlated.py --run-all        # both arms, seeds 0-9
"""
import argparse
import math
import os
import sys
import time
import random
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

N_CONV = 8
RHO_GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
R_GRID = np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
N_AGENTS = 10
N_TRIALS = 60
Q_PER_TRIAL = 10
K_SUP = 10
# ERR_CORR is the *Gaussian-copula* parameter, calibrated (see notes) so the
# achieved within-pair *binary error* correlation is ~0.5. Dichotomising at
# the per-source error thresholds attenuates the correlation, hence the
# copula parameter must be larger than the target binary correlation.
ERR_CORR = 0.8


# ---------------- RNG / generative model (verbatim copies) ----------------
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


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _std_normal(rng):
    """Box-Muller from the SafeRNG uniform stream."""
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    r_ = math.sqrt(-2.0 * math.log(u1))
    return r_ * math.cos(2.0 * math.pi * u2)


# ---------------- Bayes machinery (verbatim copies) ----------------
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


# ---------------- numpy scalar baselines ----------------
def ctxcond_logit(m_sup, a1_sup, a2_sup, z_sup, m_q, a1_q, a2_q):
    """Laplace-smoothed context-matching logit (temperature factored out)."""
    match = (a1_sup == a1_q) & (a2_sup == a2_q)
    agree = (m_sup == z_sup)
    cons_q = (np.sum(agree & match) + 0.5) / (np.sum(match) + 1.0)
    return float((2 * cons_q - 1) * (2 * m_q - 1))


def signeds_scalar_contrib(m_sup, z_sup, m_q):
    """Signed-scalar contribution (positive global scale factored out)."""
    cons = float(np.mean(m_sup == z_sup))
    return float((2 * cons - 1) * (2 * m_q - 1))


# ---------------- consensus trial ----------------
def run_arm(correlated, seed):
    """One seed of the consensus protocol. Returns dict method -> acc and,
    for the correlated arm, the empirical within-pair error correlation."""
    rng = SafeRNG(seed)
    correct = {'Bayes': 0, 'ContextConditioned': 0, 'SignedScalar': 0,
               'naive': 0, 'Oracle': 0}
    total = 0
    err_pairs = []  # (e_A, e_B) for empirical correlation check
    cands = list(range(N_CONV))
    for _ in range(N_TRIALS):
        srcs = [sample_source(rng, None, (0.99, 1.0), (0.7, 0.95))
                for _ in range(N_AGENTS)]
        sups = [generate(src, K_SUP, rng) for src in srcs]
        posts = [bayes_post(s[0], s[1], s[2], s[3], cands, RHO_GRID, R_GRID)
                 for s in sups]
        for _ in range(Q_PER_TRIAL):
            Z = int(rng.random() < 0.5)
            lo_b, lo_c, lo_s, votes, lo_o = 0.0, 0.0, 0.0, 0, 0.0
            for p in range(N_AGENTS // 2):
                A, B = 2 * p, 2 * p + 1
                (cA, rhoA, rA), (cB, rhoB, rB) = srcs[A], srcs[B]
                a1A, a2A = rng.randint(0, 2), rng.randint(0, 2)
                a1B, a2B = rng.randint(0, 2), rng.randint(0, 2)
                if correlated:
                    w1, w2 = _std_normal(rng), _std_normal(rng)
                    u2 = ERR_CORR * w1 + math.sqrt(1 - ERR_CORR ** 2) * w2
                    okA = _phi(w1) < rA
                    okB = _phi(u2) < rB
                else:
                    okA = rng.random() < rA
                    okB = rng.random() < rB
                err_pairs.append((0 if okA else 1, 0 if okB else 1))
                msgs = []
                for (j, c, rho, ok, a1, a2) in (
                        (A, cA, rhoA, okA, a1A, a2A),
                        (B, cB, rhoB, okB, a1B, a2B)):
                    sig = Z if ok else 1 - Z
                    m = int(apply_convention(c, np.array([sig]),
                                             np.array([a1]),
                                             np.array([a2]))[0]) \
                        if rng.random() < rho else sig
                    msgs.append((j, m, a1, a2))
                    s = sups[j]
                    lo_b += bayes_log_odds(posts[j], m, a1, a2, cands,
                                           RHO_GRID, R_GRID)
                    lo_c += ctxcond_logit(s[0], s[1], s[2], s[3], m, a1, a2)
                    lo_s += signeds_scalar_contrib(s[0], s[3], m)
                    votes += m
                    lo_o += oracle_log_odds(m, a1, a2, c, rho,
                                            srcs[j][2])
            correct['Bayes'] += int((lo_b > 0) == Z)
            correct['ContextConditioned'] += int((lo_c > 0) == Z)
            correct['SignedScalar'] += int((lo_s > 0) == Z)
            correct['naive'] += int((votes > N_AGENTS / 2) == Z)
            correct['Oracle'] += int((lo_o > 0) == Z)
            total += 1
    accs = {k: v / total for k, v in correct.items()}
    out = {'accs': accs}
    if correlated and err_pairs:
        e = np.array(err_pairs, dtype=float)
        out['emp_corr'] = float(np.corrcoef(e[:, 0], e[:, 1])[0, 1])
    return out


# ---------------- verification (per-source ExpB protocol) ----------------
def _eval_persource_numpy(K, pool, rho_range, rel_range, seed,
                          n_ep=30, Q=30):
    """Per-source ExpB protocol with numpy baselines (for verification)."""
    rng = SafeRNG(seed)
    correct = {'ContextConditioned': 0, 'SignedScalar': 0}
    total = 0
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, pool, rho_range, rel_range)
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            for i in range(Q):
                m_q, a1_q, a2_q = int(qm[0][i]), int(qm[1][i]), int(qm[2][i])
                z_q = int(qm[3][i])
                lo_c = ctxcond_logit(sm[0], sm[1], sm[2], sm[3],
                                     m_q, a1_q, a2_q)
                correct['ContextConditioned'] += int((lo_c > 0) == z_q)
                lo_s = signeds_scalar_contrib(sm[0], sm[3], m_q)
                correct['SignedScalar'] += int((lo_s > 0) == z_q)
                total += 1
    return {k: v / total for k, v in correct.items()}


def verify():
    print('Verifying numpy baselines vs published ExpB numbers...',
          flush=True)
    df = pd.read_csv('results/expB_ablation.csv')
    ok = True
    for seed in range(10):
        got = _eval_persource_numpy(10, None, (0.99, 1.0), (0.7, 0.95),
                                    seed * 1000 + 1)
        for method in got:
            pub = df[(df.seed == seed)
                     & (df.method == method)]['acc'].iloc[0]
            d = abs(got[method] - pub)
            flag = '' if d < 1e-6 else '  <-- MISMATCH'
            if d >= 1e-6:
                ok = False
            if seed == 0:
                print(f'  {method}: got {got[method]:.4f} pub {pub:.4f}'
                      f' diff {d:.2e}{flag}', flush=True)
    # full max diff
    diffs = []
    for seed in range(10):
        got = _eval_persource_numpy(10, None, (0.99, 1.0), (0.7, 0.95),
                                    seed * 1000 + 1)
        for method in got:
            pub = df[(df.seed == seed)
                     & (df.method == method)]['acc'].iloc[0]
            diffs.append(abs(got[method] - pub))
    print(f'  max abs diff over all seeds/methods: {max(diffs):.2e}',
          flush=True)
    # NOTE: not bit-identical at 1e-12 because the published numbers average
    # 30 per-episode float32 means (torch), while this script accumulates a
    # single float64 correct/total. The decision rule itself is identical;
    # the residual (~1e-9) is float32-vs-float64 aggregation noise.
    ok &= max(diffs) < 1e-6
    print('VERIFICATION:', 'PASS (decision-identical)' if ok else 'FAIL')
    return ok


def run_all(seeds):
    rows = []
    for seed in seeds:
        for arm, corr in (('independence', False), ('correlated', True)):
            out = run_arm(corr, seed)
            for method, acc in out['accs'].items():
                rows.append({'seed': seed, 'arm': arm, 'method': method,
                             'acc': acc})
            extra = (f" emp_err_corr={out['emp_corr']:.3f}"
                     if 'emp_corr' in out else '')
            accs = ' '.join(f'{m}={a:.3f}'
                            for m, a in out['accs'].items())
            print(f'  seed {seed} {arm}: {accs}{extra}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('results/expH_correlated.csv', index=False)
    return df


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
        print('Experiment H: correlated sources (both arms)', flush=True)
        run_all(list(range(10)))
        print(f'ExpH done ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
