"""
Provenance-Aware Belief Aggregation in Multi-Agent Communication
Under Convention Heterogeneity.

Full experimental suite. Reproduces all tables and figures in the paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
N_SEEDS = 3  # Set to 10 for stronger statistical power.

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f'Device: {DEVICE}')
print(f'Seeds: {N_SEEDS}')
print(f'Results: {RESULTS_DIR}/')


# ============================================================
# SAFE RNG
# ============================================================
class SafeRNG:
    """Deterministic RNG wrapper around random.Random.
    Avoids known issues with numpy Generator on recent Python/NumPy."""
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


# ============================================================
# CONVENTIONS
# ============================================================
N_CONV = 8
TRAIN_CONVS = [0, 1, 2, 3]
HELD_CONVS = [4, 5, 6, 7]


def apply_convention(c, signal, a1, a2):
    """Apply convention c. Conventions 0-7 are involutions in the first
    argument; c=8 is the 'truthful-iff-science' rule used by agent E in
    the LLM benchmark (negated xor-topic), also an involution."""
    if c == 0: return signal
    if c == 1: return 1 - signal
    if c == 2: return signal ^ a1
    if c == 3: return signal ^ a2
    if c == 4: return signal ^ (a1 & a2)
    if c == 5: return signal ^ (a1 | a2)
    if c == 6: return signal ^ (a1 ^ a2)
    if c == 7: return signal ^ (1 - (a1 & a2))
    if c == 8: return signal ^ (1 - a2)
    raise ValueError(f'Unknown convention {c}')


# ============================================================
# SOURCE MODEL
# ============================================================
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


# ============================================================
# BAYESIAN GRID INFERENCE
# ============================================================
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
    """Posterior-predictive log-odds for a query given the posterior.

    Computes log( E_post[P(M_q | Y=1, theta)] / E_post[P(M_q | Y=0, theta)] ),
    the log-odds of the posterior-predictive distribution. This is the exact
    Bayesian predictive quantity -- the log of the ratio of posterior-
    averaged likelihoods -- not the posterior expectation of the
    parameter-conditional log-odds (those differ by Jensen's inequality).
    Defensive reshape.
    """
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


def bayes_predict_single(m_sup, a1_sup, a2_sup, z_sup,
                          m_q, a1_q, a2_q, candidates=None):
    if candidates is None:
        candidates = list(range(N_CONV))
    post = bayes_post(m_sup, a1_sup, a2_sup, z_sup, candidates)
    lo = bayes_log_odds(post, m_q, a1_q, a2_q, candidates)
    return 1 if lo > 0 else 0


def oracle_log_odds(m_q, a1_q, a2_q, c, rho, r):
    """Exact log-odds with known (c, rho, r): the Bayes-optimal rule.

    This is bayes_log_odds with a degenerate (point-mass) posterior over
    the true parameters. It uses only the observed message, context, and
    the known parameters -- never the source's unobserved reading.
    """
    A_q = float(apply_convention(c, np.array([m_q]),
                                 np.array([a1_q]),
                                 np.array([a2_q]))[0])
    B_q = 1.0 if m_q == 1 else 0.0

    X1 = rho * A_q + (1 - rho) * B_q
    p1 = (1 - r) + X1 * (2 * r - 1)

    X0 = rho * (1 - A_q) + (1 - rho) * (1 - B_q)
    p0 = (1 - r) + X0 * (2 * r - 1)

    return float(np.log(max(p1, 1e-12)) - np.log(max(p0, 1e-12)))


# ============================================================
# NEURAL MODELS
# ============================================================
class MetaAttn(nn.Module):
    def __init__(self, d=64, sup_in=4, qry_in=3):
        super().__init__()
        self.sup_in, self.qry_in = sup_in, qry_in
        self.sup = nn.Sequential(nn.Linear(sup_in, d), nn.ReLU(),
                                  nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.attn = nn.Linear(d, 1)
        self.qry = nn.Sequential(nn.Linear(qry_in, d), nn.ReLU(),
                                  nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, s, q):
        s, q = s.float(), q.float()
        B, N, K, _ = s.shape
        Q_ = q.shape[2]
        s_enc = self.sup(s.reshape(B * N, K, self.sup_in))
        w = F.softmax(self.attn(s_enc), dim=1)
        s_agg = (s_enc * w).sum(dim=1)
        q_enc = self.qry(q.reshape(B * N, Q_, self.qry_in))
        combined = torch.cat([q_enc, s_agg.unsqueeze(1).expand(-1, Q_, -1)], dim=-1)
        return self.head(combined).reshape(B, N, Q_, 2)


class MetaMean(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.sup = nn.Sequential(nn.Linear(4, d), nn.ReLU(),
                                  nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.qry = nn.Sequential(nn.Linear(3, d), nn.ReLU(),
                                  nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, s, q):
        s, q = s.float(), q.float()
        B, N, K, _ = s.shape
        Q_ = q.shape[2]
        s_agg = self.sup(s.reshape(B * N, K, 4)).mean(dim=1)
        q_enc = self.qry(q.reshape(B * N, Q_, 3))
        combined = torch.cat([q_enc, s_agg.unsqueeze(1).expand(-1, Q_, -1)], dim=-1)
        return self.head(combined).reshape(B, N, Q_, 2)


class NoSupport(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, d), nn.ReLU(),
                                  nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, s, q):
        q = q.float()
        B, N, Q_, _ = q.shape
        return self.net(q.reshape(B * N * Q_, 3)).reshape(B, N, Q_, 2)


class UnsignedScalar(nn.Module):
    """Non-negative weight on message; directly tests Theorem 1."""
    def __init__(self):
        super().__init__()
        self.raw_w = nn.Parameter(torch.zeros(1))

    def forward(self, s, q):
        B, N, K, _ = s.shape
        Q_ = q.shape[2]
        m_sup = s[..., 0].float()
        z_sup = s[..., 3].float()
        cons = (m_sup == z_sup).float().mean(dim=-1)
        w = F.softplus(self.raw_w) * (2 * cons - 1)
        w = F.relu(w)
        m_q = q[..., 0].float()
        logit = w.unsqueeze(-1) * (2 * m_q - 1)
        return torch.stack([-logit, logit], dim=-1)


class SignedScalar(nn.Module):
    """Signed scalar weight; tests Theorem 2."""
    def __init__(self):
        super().__init__()
        self.raw_w = nn.Parameter(torch.zeros(1))

    def forward(self, s, q):
        B, N, K, _ = s.shape
        Q_ = q.shape[2]
        m_sup = s[..., 0].float()
        z_sup = s[..., 3].float()
        cons = (m_sup == z_sup).float().mean(dim=-1)
        w = torch.tanh(self.raw_w) * (2 * cons - 1)
        m_q = q[..., 0].float()
        logit = w.unsqueeze(-1) * (2 * m_q - 1)
        return torch.stack([-logit, logit], dim=-1)


class ContextConditioned(nn.Module):
    """Context-conditioned scalar weight.

    For each query context a, estimates the support agreement rate
    P(m == z | context == a) from context-matching support examples
    (Laplace-smoothed) and sets the weight sign from it. Only the
    support set is used; no privileged information.
    """
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, s, q):
        s = s.float()
        q = q.float()
        B, N, K, _ = s.shape
        Q_ = q.shape[2]
        m_sup = s[..., 0]                                # (B, N, K)
        z_sup = s[..., 3]                                # (B, N, K)
        a_sup = s[..., 1:3]                              # (B, N, K, 2)
        a_q = q[..., 1:]                                 # (B, N, Q, 2)
        match = (a_sup.unsqueeze(2) == a_q.unsqueeze(3)).all(-1).float()
        agree = (m_sup.unsqueeze(2) == z_sup.unsqueeze(2)).float()
        cons_q = ((agree * match).sum(-1) + 0.5) / (match.sum(-1) + 1.0)
        w = self.temp * (2 * cons_q - 1)                  # (B, N, Q)
        m_q = q[..., 0]                                  # (B, N, Q)
        logit = w * (2 * m_q - 1)
        return torch.stack([-logit, logit], dim=-1)


# ============================================================
# DATA HELPERS
# ============================================================
def sample_batch(rng, n_sources, K, Q, pool=None,
                 rho_range=(0.6, 1.0), rel_range=(0.7, 0.95),
                 feat_mode='raw'):
    """Draw a meta-learning batch.

    feat_mode='raw':    support (m, a1, a2, z), query (m, a1, a2).
    feat_mode='dec':    additionally the bit decoded with the TRUE
                        convention (privileged diagnostic feature).
    feat_mode='decctx': additionally a one-hot convention index
                        (privileged diagnostic feature).
    """
    S, Qs, Z = [], [], []
    for _ in range(n_sources):
        src = sample_source(rng, pool, rho_range, rel_range)
        c = src[0]
        sm = generate(src, K, rng)
        qm = generate(src, Q, rng)
        m_s = torch.tensor(sm[0]); a1_s = torch.tensor(sm[1])
        a2_s = torch.tensor(sm[2]); z_s = torch.tensor(sm[3])
        m_q = torch.tensor(qm[0]); a1_q = torch.tensor(qm[1])
        a2_q = torch.tensor(qm[2])
        if feat_mode == 'raw':
            s = torch.stack([m_s, a1_s, a2_s, z_s], dim=-1)
            q = torch.stack([m_q, a1_q, a2_q], dim=-1)
        else:
            # Diagnostic: decode with the true convention.
            d_s = torch.tensor(apply_convention(c, sm[0], sm[1], sm[2]))
            d_q = torch.tensor(apply_convention(c, qm[0], qm[1], qm[2]))
            s = torch.stack([m_s, a1_s, a2_s, z_s, d_s], dim=-1)
            q = torch.stack([m_q, a1_q, a2_q, d_q], dim=-1)
            if feat_mode == 'decctx':
                oh = F.one_hot(torch.tensor(c), num_classes=8).float()
                s = torch.cat([s.float(),
                               oh.unsqueeze(0).expand(s.shape[0], -1)], dim=-1)
                q = torch.cat([q.float(),
                               oh.unsqueeze(0).expand(q.shape[0], -1)], dim=-1)
        S.append(s)
        Qs.append(q)
        Z.append(torch.tensor(qm[3]))
    return torch.stack(S), torch.stack(Qs), torch.stack(Z)


def make_model(ModelClass, seed, **kwargs):
    """Factory: seeds and creates a model on the correct device."""
    torch.manual_seed(seed)
    return ModelClass(**kwargs).to(DEVICE)


def train_neural(model, n_steps=2000, batch=16, K=10, Q=20, lr=1e-3,
                 pool=None, rho_range=(0.6, 1.0), rel_range=(0.7, 0.95),
                 seed=0, feat_mode='raw'):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = SafeRNG(seed)
    for _ in range(n_steps):
        Ss, Qs, Zs = [], [], []
        for _ in range(batch):
            s, q, z = sample_batch(rng, 4, K, Q, pool, rho_range, rel_range,
                                   feat_mode=feat_mode)
            Ss.append(s); Qs.append(q); Zs.append(z)
        S_b = torch.stack(Ss).to(DEVICE)
        Q_b = torch.stack(Qs).to(DEVICE)
        Z_b = torch.stack(Zs).to(DEVICE)
        logits = model(S_b, Q_b)
        loss = F.cross_entropy(logits.reshape(-1, 2), Z_b.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_neural(model, K, pool, rho_range, rel_range, seed, n_ep=30, Q=30,
                feat_mode='raw'):
    model.eval()
    rng = SafeRNG(seed)
    accs = []
    for _ in range(n_ep):
        s, q, z = sample_batch(rng, 6, K, Q, pool, rho_range, rel_range,
                               feat_mode=feat_mode)
        preds = model(s.unsqueeze(0).to(DEVICE),
                      q.unsqueeze(0).to(DEVICE)).argmax(-1).cpu()
        accs.append((preds == z).float().mean().item())
    return float(np.mean(accs))


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


# ============================================================
# SMOKE TESTS
# ============================================================
def smoke_test():
    """Verify core components on small synthetic cases."""
    print('Running smoke tests...')
    rng = SafeRNG(0)

    # 1. Posterior shape
    src = (0, 1.0, 0.85)
    sm = generate(src, 20, rng)
    post = bayes_post(sm[0], sm[1], sm[2], sm[3], list(range(N_CONV)))
    assert post.shape == (N_CONV, len(RHO_GRID), len(R_GRID)), \
        f'Posterior shape {post.shape}'

    # 2. Posterior normalization
    assert abs(post.sum() - 1.0) < 1e-9, f'Posterior sum {post.sum()}'

    # 3. Log-odds is finite
    lo = bayes_log_odds(post, int(sm[0][0]), int(sm[1][0]),
                         int(sm[2][0]), list(range(N_CONV)))
    assert np.isfinite(lo), f'Log-odds {lo}'

    # 4. Prediction is binary
    pred = bayes_predict_single(sm[0], sm[1], sm[2], sm[3],
                                 int(sm[0][0]), int(sm[1][0]), int(sm[2][0]))
    assert pred in (0, 1), f'Prediction {pred}'

    # 5. Accuracy on a clean synthetic case
    rng2 = SafeRNG(1)
    correct, total = 0, 0
    for _ in range(20):
        src = (1, 1.0, 0.9)  # always-inverted, high reliability
        sm = generate(src, 30, rng2)
        qm = generate(src, 20, rng2)
        post = bayes_post(sm[0], sm[1], sm[2], sm[3], list(range(N_CONV)))
        for i in range(20):
            pred = bayes_predict_single(sm[0], sm[1], sm[2], sm[3],
                                         int(qm[0][i]), int(qm[1][i]),
                                         int(qm[2][i]))
            correct += int(pred == int(qm[3][i]))
            total += 1
    acc = correct / total
    assert acc > 0.7, f'Bayes accuracy {acc} on clean case'

    # 6. RNG reproducibility
    rng3, rng4 = SafeRNG(42), SafeRNG(42)
    x = rng3.uniform(0, 1, size=20)
    y = rng4.uniform(0, 1, size=20)
    assert np.array_equal(x, y), 'RNG not reproducible'

    print(f'  All smoke tests passed (accuracy {acc:.3f})')


# ============================================================
# EXPERIMENT A — Convention stability sweep
# ============================================================
def experiment_A(seeds):
    print('\n' + '=' * 60)
    print('EXPERIMENT A: Convention stability sweep')
    print('=' * 60)
    rho_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for seed in seeds:
        print(f'  seed {seed}')
        # Train on the full rho range so the eval grid (incl. rho=0.5)
        # is in-distribution.
        m_attn = train_neural(make_model(MetaAttn, seed), seed=seed,
                              rho_range=(0.5, 1.0))
        m_mean = train_neural(make_model(MetaMean, seed), seed=seed,
                              rho_range=(0.5, 1.0))
        m_nos  = train_neural(make_model(NoSupport, seed), seed=seed,
                              rho_range=(0.5, 1.0))
        m_uns  = train_neural(make_model(UnsignedScalar, seed), seed=seed,
                              rho_range=(0.5, 1.0))
        m_sig  = train_neural(make_model(SignedScalar, seed), seed=seed,
                              rho_range=(0.5, 1.0))

        for rho in rho_values:
            rho_range = (rho - 0.01, rho + 0.01)
            rel_range = (0.7, 0.95)
            s_eval = seed * 100 + int(rho * 100)

            for name, m in [('MetaAttn', m_attn), ('MetaMean', m_mean),
                            ('NoSupport', m_nos), ('UnsignedScalar', m_uns),
                            ('SignedScalar', m_sig)]:
                acc = eval_neural(m, 10, None, rho_range, rel_range, s_eval)
                rows.append({'seed': seed, 'rho': rho,
                             'method': name, 'acc': acc})

            acc_b = eval_bayes(10, None, rho_range, rel_range, s_eval)
            rows.append({'seed': seed, 'rho': rho, 'method': 'Bayes', 'acc': acc_b})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/expA_rho_sweep.csv', index=False)
    print('\n' + df.groupby(['rho', 'method'])['acc'].mean().unstack().round(4).to_string())
    return df


# ============================================================
# EXPERIMENT B — Representation ablation
# ============================================================
def experiment_B(seeds):
    print('\n' + '=' * 60)
    print('EXPERIMENT B: Representation ablation')
    print('=' * 60)
    rows = []
    for seed in seeds:
        print(f'  seed {seed}')
        m_ctx  = train_neural(make_model(ContextConditioned, seed), seed=seed)
        m_sig  = train_neural(make_model(SignedScalar, seed), seed=seed)
        m_uns  = train_neural(make_model(UnsignedScalar, seed), seed=seed)
        m_attn = train_neural(make_model(MetaAttn, seed), seed=seed)
        m_nos  = train_neural(make_model(NoSupport, seed), seed=seed)
        # Diagnostic representation ablation: privileged decoded features.
        m_dec  = train_neural(make_model(MetaAttn, seed, sup_in=5, qry_in=4),
                              feat_mode='dec', seed=seed)
        m_dctx = train_neural(make_model(MetaAttn, seed, sup_in=13, qry_in=12),
                              feat_mode='decctx', seed=seed)

        rho_range = (0.99, 1.0)
        rel_range = (0.7, 0.95)
        s_eval = seed * 1000 + 1

        results = {
            'ContentOnly':         eval_neural(m_nos,  10, None, rho_range, rel_range, s_eval),
            'UnsignedScalar':      eval_neural(m_uns,  10, None, rho_range, rel_range, s_eval),
            'SignedScalar':        eval_neural(m_sig,  10, None, rho_range, rel_range, s_eval),
            'ContextConditioned':  eval_neural(m_ctx,  10, None, rho_range, rel_range, s_eval),
            'AttentionCalibrator': eval_neural(m_attn, 10, None, rho_range, rel_range, s_eval),
            'MetaAttn+Dec':        eval_neural(m_dec,  10, None, rho_range, rel_range, s_eval,
                                               feat_mode='dec'),
            'MetaAttn+DecCtx':     eval_neural(m_dctx, 10, None, rho_range, rel_range, s_eval,
                                               feat_mode='decctx'),
            'Bayes':               eval_bayes(10, None, rho_range, rel_range, s_eval),
        }

        # Oracle: knows true (c, rho, r); applies the Bayes-optimal rule.
        rng = SafeRNG(s_eval + 500)
        correct, total = 0, 0
        for _ in range(30):
            for _ in range(6):
                src = sample_source(rng, None, rho_range, rel_range)
                c, rho, r = src
                sm = generate(src, 10, rng)
                qm = generate(src, 30, rng)
                for i in range(30):
                    lo = oracle_log_odds(int(qm[0][i]), int(qm[1][i]),
                                         int(qm[2][i]), c, rho, r)
                    correct += int((lo > 0) == int(qm[3][i]))
                    total += 1
        results['Oracle'] = correct / total

        for name, acc in results.items():
            rows.append({'seed': seed, 'method': name, 'acc': acc})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/expB_ablation.csv', index=False)
    print('\n' + df.groupby('method')['acc'].agg(['mean', 'std']).round(4).to_string())
    return df


# ============================================================
# EXPERIMENT C — K-sweep on held-out conventions
# ============================================================
def experiment_C(seeds):
    print('\n' + '=' * 60)
    print('EXPERIMENT C: K-sweep on held-out conventions')
    print('=' * 60)
    K_values = [1, 2, 5, 10, 20, 50]
    rows = []
    for seed in seeds:
        print(f'  seed {seed}')
        m_attn = train_neural(make_model(MetaAttn, seed), pool=TRAIN_CONVS,
                              rho_range=(0.9, 1.0), seed=seed)
        m_mean = train_neural(make_model(MetaMean, seed), pool=TRAIN_CONVS,
                              rho_range=(0.9, 1.0), seed=seed)
        m_nos  = train_neural(make_model(NoSupport, seed), pool=TRAIN_CONVS,
                              rho_range=(0.9, 1.0), seed=seed)

        for K in K_values:
            s_eval = seed * 100 + K
            for name, m in [('MetaAttn', m_attn), ('MetaMean', m_mean),
                            ('NoSupport', m_nos)]:
                acc = eval_neural(m, K, HELD_CONVS, (0.99, 1.0),
                                  (0.7, 0.95), s_eval)
                rows.append({'seed': seed, 'K': K, 'method': name, 'acc': acc})

            acc_bu = eval_bayes(K, HELD_CONVS, (0.99, 1.0), (0.7, 0.95),
                                s_eval, candidates=list(range(N_CONV)))
            acc_bf = eval_bayes(K, HELD_CONVS, (0.99, 1.0), (0.7, 0.95),
                                s_eval, candidates=HELD_CONVS)
            rows.append({'seed': seed, 'K': K, 'method': 'BayesUniv', 'acc': acc_bu})
            rows.append({'seed': seed, 'K': K, 'method': 'BayesFam', 'acc': acc_bf})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/expC_ksweep.csv', index=False)
    print('\n' + df.groupby(['K', 'method'])['acc'].mean().unstack().round(4).to_string())
    return df


# ============================================================
# EXPERIMENT D — Reliability heterogeneity
# ============================================================
def experiment_D(seeds):
    print('\n' + '=' * 60)
    print('EXPERIMENT D: Reliability heterogeneity')
    print('=' * 60)
    # Mean-matched reliability conditions: E[r] = 0.80 in all three, so any
    # performance difference is attributable to heterogeneity (spread),
    # not to the mean reliability level.
    rel_conditions = {
        'homogeneous':   (0.79, 0.81),
        'moderate':      (0.70, 0.90),
        'heterogeneous': (0.65, 0.95),
    }
    rel_offset = {'homogeneous': 0, 'moderate': 1, 'heterogeneous': 2}
    rows = []
    for seed in seeds:
        for name, rel_range in rel_conditions.items():
            off = rel_offset[name]
            rho_range = (0.99, 1.0)

            m_attn = train_neural(make_model(MetaAttn, seed + off),
                                  pool=None, rho_range=rho_range,
                                  rel_range=rel_range, seed=seed + off)
            m_nos = train_neural(make_model(NoSupport, seed + off),
                                 pool=None, rho_range=rho_range,
                                 rel_range=rel_range, seed=seed + off)

            s_eval = seed * 10 + off
            acc_at = eval_neural(m_attn, 10, None, rho_range, rel_range, s_eval)
            acc_ns = eval_neural(m_nos, 10, None, rho_range, rel_range, s_eval)
            acc_b = eval_bayes(10, None, rho_range, rel_range, s_eval)

            rows.append({'seed': seed, 'rel': name, 'method': 'MetaAttn', 'acc': acc_at})
            rows.append({'seed': seed, 'rel': name, 'method': 'NoSupport', 'acc': acc_ns})
            rows.append({'seed': seed, 'rel': name, 'method': 'Bayes', 'acc': acc_b})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/expD_heterogeneity.csv', index=False)
    print('\n' + df.groupby(['rel', 'method'])['acc'].mean().unstack().round(4).to_string())
    return df


# ============================================================
# EXPERIMENT E — Multi-agent consensus
# ============================================================
def experiment_E(seeds):
    print('\n' + '=' * 60)
    print('EXPERIMENT E: Multi-agent consensus')
    print('=' * 60)

    def consensus(neural_attn, n_agents, n_trials=60, K=10, seed=0):
        rng = SafeRNG(seed)
        methods = ['NaiveMean', 'MajorityVote', 'MetaAttn', 'Bayes', 'Oracle']
        correct = {m: 0 for m in methods}

        with torch.no_grad():
            for _ in range(n_trials):
                Z_true = int(rng.random() < 0.5)
                supports, queries, raw_msgs, params = [], [], [], []

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

                    supports.append(torch.stack(
                        [torch.tensor(sm[0]), torch.tensor(sm[1]),
                         torch.tensor(sm[2]), torch.tensor(sm[3])], dim=-1))
                    queries.append(torch.tensor([[m_q, a1_q, a2_q]]))

                S_b = torch.stack(supports).unsqueeze(0).to(DEVICE)
                Q_b = torch.stack(queries).unsqueeze(0).to(DEVICE)

                correct['NaiveMean']    += int((np.mean(raw_msgs) > 0.5) == Z_true)
                correct['MajorityVote'] += int((np.sum(raw_msgs) > n_agents / 2) == Z_true)

                p_at = F.softmax(neural_attn(S_b, Q_b),
                                 dim=-1)[0, :, 0, 1].cpu().numpy()
                correct['MetaAttn'] += int((p_at.mean() > 0.5) == Z_true)

                lo_total = 0.0
                for j in range(n_agents):
                    sup = supports[j].numpy()
                    q = queries[j].numpy()[0]
                    post = bayes_post(sup[:, 0], sup[:, 1], sup[:, 2], sup[:, 3],
                                       list(range(N_CONV)))
                    lo_total += bayes_log_odds(post, int(q[0]), int(q[1]),
                                                int(q[2]), list(range(N_CONV)))
                correct['Bayes'] += int((lo_total > 0) == Z_true)

                # Oracle: knows true (c, rho, r) per agent; Bayes-optimal rule
                # on the observed messages (never the unobserved readings).
                lo_oracle = 0.0
                for j in range(n_agents):
                    q = queries[j].numpy()[0]
                    c_j, rho_j, r_j = params[j]
                    lo_oracle += oracle_log_odds(int(q[0]), int(q[1]),
                                                 int(q[2]), c_j, rho_j, r_j)
                correct['Oracle'] += int((lo_oracle > 0) == Z_true)

        return {k: v / n_trials for k, v in correct.items()}

    rows = []
    for seed in seeds:
        print(f'  seed {seed}')
        m_attn = train_neural(make_model(MetaAttn, seed), pool=None,
                              rho_range=(0.7, 1.0), seed=seed)
        for n_agents in [2, 3, 4, 6, 8, 12]:
            res = consensus(m_attn, n_agents=n_agents, n_trials=60, K=10,
                            seed=seed * 100 + n_agents)
            for method, acc in res.items():
                rows.append({'seed': seed, 'n_agents': n_agents,
                             'method': method, 'acc': acc})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/expE_consensus.csv', index=False)
    print('\n' + df.groupby(['n_agents', 'method'])['acc'].mean().unstack().round(4).to_string())
    return df


# ============================================================
# MAIN
# ============================================================
# ============================================================
# EXPERIMENT F — Posterior diagnostic
# ============================================================
def experiment_F(seeds, n_sources=200, K=20):
    print('\n' + '=' * 60)
    print('EXPERIMENT F: Posterior diagnostic')
    print('=' * 60)
    rows = []
    for seed in seeds:
        rng = SafeRNG(10000 + seed)
        top1, mass50 = 0, 0
        for _ in range(n_sources):
            src = sample_source(rng, None, (0.99, 1.0), (0.7, 0.95))
            c_true = src[0]
            sm = generate(src, K, rng)
            post = bayes_post(sm[0], sm[1], sm[2], sm[3],
                              list(range(N_CONV)))
            conv_mass = post.sum(axis=(1, 2))  # marginalize rho, r
            top1 += int(int(np.argmax(conv_mass)) == c_true)
            mass50 += int(conv_mass[c_true] > 0.50)
        rows.append({'seed': seed,
                     'top1_acc': top1 / n_sources,
                     'mass_gt_50': mass50 / n_sources})
    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/expF_posterior.csv', index=False)
    print('\n' + df.mean(numeric_only=True).round(4).to_string())
    return df


# ============================================================
# EXPERIMENT G — Candidate-set misspecification
# ============================================================
def experiment_G(seeds):
    """What happens when the true convention is outside the candidate set?

    Sources are drawn from all 8 conventions, but BayesHalf only considers
    candidates {0,1,2,3}. BayesFull (candidates 0-7) is the well-specified
    control. Tests graceful degradation under an incomplete convention
    family -- the realistic case where the modeller does not know every
    convention in the wild.
    """
    print('\n' + '=' * 60)
    print('EXPERIMENT G: Candidate-set misspecification')
    print('=' * 60)
    rows = []
    for seed in seeds:
        print(f'  seed {seed}')
        m_attn = train_neural(make_model(MetaAttn, seed), pool=None,
                              rho_range=(0.9, 1.0), seed=seed)
        s_eval = seed * 1000 + 77
        rho_range = (0.99, 1.0)
        rel_range = (0.7, 0.95)

        acc_full = eval_bayes(10, None, rho_range, rel_range, s_eval,
                              candidates=list(range(N_CONV)))
        acc_half = eval_bayes(10, None, rho_range, rel_range, s_eval,
                              candidates=[0, 1, 2, 3])
        acc_attn = eval_neural(m_attn, 10, None, rho_range, rel_range, s_eval)

        # Oracle reference.
        rng = SafeRNG(s_eval + 500)
        correct, total = 0, 0
        for _ in range(30):
            for _ in range(6):
                src = sample_source(rng, None, rho_range, rel_range)
                c, rho, r = src
                qm = generate(src, 30, rng)
                for i in range(30):
                    lo = oracle_log_odds(int(qm[0][i]), int(qm[1][i]),
                                         int(qm[2][i]), c, rho, r)
                    correct += int((lo > 0) == int(qm[3][i]))
                    total += 1
        acc_or = correct / total

        # Accuracy split by whether the true convention was in the
        # restricted candidate set.
        rng2 = SafeRNG(s_eval + 501)
        in_acc, out_acc, in_n, out_n = 0, 0, 0, 0
        for _ in range(30):
            for _ in range(6):
                src = sample_source(rng2, None, rho_range, rel_range)
                c, rho, r = src
                sm = generate(src, 10, rng2)
                qm = generate(src, 30, rng2)
                post = bayes_post(sm[0], sm[1], sm[2], sm[3], [0, 1, 2, 3])
                for i in range(30):
                    lo = bayes_log_odds(post, int(qm[0][i]), int(qm[1][i]),
                                        int(qm[2][i]), [0, 1, 2, 3])
                    ok = int((lo > 0) == int(qm[3][i]))
                    if c in (0, 1, 2, 3):
                        in_acc += ok; in_n += 1
                    else:
                        out_acc += ok; out_n += 1
        rows.append({'seed': seed, 'method': 'BayesFull', 'acc': acc_full})
        rows.append({'seed': seed, 'method': 'BayesHalf', 'acc': acc_half})
        rows.append({'seed': seed, 'method': 'MetaAttn', 'acc': acc_attn})
        rows.append({'seed': seed, 'method': 'Oracle', 'acc': acc_or})
        rows.append({'seed': seed, 'method': 'BayesHalf_in',
                     'acc': in_acc / in_n})
        rows.append({'seed': seed, 'method': 'BayesHalf_out',
                     'acc': out_acc / out_n})

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/expG_misspec.csv', index=False)
    print('\n' + df.groupby('method')['acc'].agg(['mean', 'std']).round(4).to_string())
    return df


def main():
    smoke_test()
    seeds = list(range(N_SEEDS))
    start = datetime.now()

    dfA = experiment_A(seeds)
    dfB = experiment_B(seeds)
    dfC = experiment_C(seeds)
    dfD = experiment_D(seeds)
    dfE = experiment_E(seeds)
    dfF = experiment_F(seeds)
    dfG = experiment_G(seeds)

    elapsed = (datetime.now() - start).total_seconds()
    print(f'\nTotal runtime: {elapsed / 60:.1f} minutes')
    with open(f'{RESULTS_DIR}/run_summary.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'seeds': seeds,
            'runtime_seconds': elapsed,
            'device': str(DEVICE),
            'n_seeds': N_SEEDS,
        }, f, indent=2)
    print(f'\nAll results saved in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
