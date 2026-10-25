"""
Review-driven upgrade: 100-seed reruns of headline experiments (A, B, C, E)
with calibration metrics (NLL, Brier, ECE) and stronger baselines.

New baselines (review item 3):
  - DeepAttn: high-capacity universal neural decoder (d=256, deeper MLPs).
    Same information as MetaAttn; tests whether Bayes wins only because
    the neural baselines are too small.
  - TransformerMeta: small transformer encoder with full self-attention
    over the support+query sequence; a proper meta-learner baseline.
  - BetaBinomial: nonparametric direct estimator of P(z|m,a) via
    Beta-Binomial with backoff. No convention structure at all; tests
    whether the convention model helps beyond flexible nonparametrics.

Calibration (review item 2/9): every probabilistic method reports
accuracy, NLL, Brier score and ECE alongside accuracy.

Paired-seed design: all methods share seeds within an experiment, so
paired differences with 95% CIs are valid (see analyze_n100.py).

Usage:
  python3 code/upgrade_n100.py            # full 100-seed run (~8-10h, 2 workers)
  python3 code/upgrade_n100.py --quick    # 2-seed smoke of Exp B (~10 min)
Run from the repository root.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit
from multiprocessing import Pool

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'code')
from pvi_experiments import (
    SafeRNG, sample_source, generate, apply_convention,
    bayes_post, bayes_log_odds, oracle_log_odds,
    MetaAttn, MetaMean, NoSupport, UnsignedScalar, SignedScalar,
    ContextConditioned, make_model, train_neural, sample_batch,
    N_CONV, TRAIN_CONVS, HELD_CONVS, DEVICE, RESULTS_DIR,
)

# ============================================================
# NEW BASELINES
# ============================================================

class DeepAttn(nn.Module):
    """High-capacity universal neural decoder.

    Same (support, query) interface and information as MetaAttn, but
    d=256 with deeper MLPs. If Bayes still wins, it is not because the
    neural baselines were too small.
    """
    def __init__(self, d=256, sup_in=4, qry_in=3):
        super().__init__()
        self.sup_in, self.qry_in = sup_in, qry_in

        def mlp(inp):
            return nn.Sequential(
                nn.Linear(inp, d), nn.ReLU(),
                nn.Linear(d, d), nn.ReLU(),
                nn.Linear(d, d), nn.ReLU(),
                nn.Linear(d, d))

        self.sup = mlp(sup_in)
        self.qry = mlp(qry_in)
        self.attn = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(),
                                  nn.Linear(d, d), nn.ReLU(),
                                  nn.Linear(d, 2))

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


class TransformerMeta(nn.Module):
    """Transformer meta-learner over the support+query sequence.

    Full self-attention between every support example and every query;
    strictly more expressive interaction than single-head attention
    pooling (MetaAttn) or mean pooling (MetaMean).
    """
    def __init__(self, d=64, nhead=4, nlayers=2, sup_in=4, qry_in=3):
        super().__init__()
        self.sup_emb = nn.Linear(sup_in, d)
        self.qry_emb = nn.Linear(qry_in, d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4 * d,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                                  nn.Linear(d, 2))

    def forward(self, s, q):
        s, q = s.float(), q.float()
        B, N, K, _ = s.shape
        Q_ = q.shape[2]
        BN = B * N
        s_emb = self.sup_emb(s.reshape(BN, K, -1))
        q_emb = self.qry_emb(q.reshape(BN, Q_, -1))
        x = torch.cat([s_emb, q_emb], dim=1)          # (BN, K+Q, d)
        h = self.enc(x)[:, K:, :]                      # query positions
        return self.head(h).reshape(B, N, Q_, 2)


def betabinomial_probs(m_sup, a1_sup, a2_sup, z_sup, m_q, a1_q, a2_q):
    """Nonparametric P(z=1 | m, a): Beta-Binomial on exact (m,a1,a2)
    matches, backing off to (m,) matches, then the global rate.
    No convention structure whatsoever.
    """
    m_sup = np.asarray(m_sup); a1_sup = np.asarray(a1_sup)
    a2_sup = np.asarray(a2_sup); z_sup = np.asarray(z_sup)
    out = np.zeros(len(m_q))
    for i in range(len(m_q)):
        mask = ((m_sup == m_q[i]) & (a1_sup == a1_q[i])
                & (a2_sup == a2_q[i]))
        if mask.sum() == 0:
            mask = (m_sup == m_q[i])
        if mask.sum() == 0:
            mask = np.ones(len(m_sup), dtype=bool)
        n1 = z_sup[mask].sum()
        n = mask.sum()
        out[i] = (n1 + 1.0) / (n + 2.0)
    return out


# ============================================================
# CALIBRATION
# ============================================================

def calibration_metrics(probs, labels, n_bins=10):
    """Return dict with acc, nll, brier, ece."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    z = np.asarray(labels, dtype=float)
    acc = float(((p > 0.5) == (z > 0.5)).mean())
    nll = float(-(z * np.log(p) + (1 - z) * np.log(1 - p)).mean())
    brier = float(((p - z) ** 2).mean())
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (p > edges[b]) & (p <= edges[b + 1])
        if m.sum() > 0:
            ece += (m.mean() * abs(p[m].mean() - z[m].mean()))
    return {'acc': acc, 'nll': float(nll), 'brier': float(brier),
            'ece': float(ece)}


@torch.no_grad()
def eval_neural_full(model, K, pool, rho_range, rel_range, seed,
                     n_ep=30, Q=30, feat_mode='raw'):
    model.eval()
    rng = SafeRNG(seed)
    probs, labels = [], []
    for _ in range(n_ep):
        s, q, z = sample_batch(rng, 6, K, Q, pool, rho_range, rel_range,
                               feat_mode=feat_mode)
        p = F.softmax(model(s.unsqueeze(0).to(DEVICE),
                            q.unsqueeze(0).to(DEVICE)),
                      dim=-1)[0, :, :, 1].cpu().numpy()
        probs.append(p.reshape(-1))
        labels.append(np.asarray(z).reshape(-1))
    return calibration_metrics(np.concatenate(probs), np.concatenate(labels))


def eval_bayes_full(K, pool, rho_range, rel_range, seed, n_ep=30, Q=30,
                    candidates=None):
    if candidates is None:
        candidates = list(range(N_CONV))
    rng = SafeRNG(seed)
    probs, labels = [], []
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, pool, rho_range, rel_range)
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            post = bayes_post(sm[0], sm[1], sm[2], sm[3], candidates)
            for i in range(Q):
                lo = bayes_log_odds(post, int(qm[0][i]), int(qm[1][i]),
                                    int(qm[2][i]), candidates)
                probs.append(1.0 / (1.0 + np.exp(-lo)))
                labels.append(int(qm[3][i]))
    return calibration_metrics(np.array(probs), np.array(labels))


def eval_betabinomial_full(K, pool, rho_range, rel_range, seed, n_ep=30,
                           Q=30):
    rng = SafeRNG(seed)
    probs, labels = [], []
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, pool, rho_range, rel_range)
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            p = betabinomial_probs(sm[0], sm[1], sm[2], sm[3],
                                   qm[0], qm[1], qm[2])
            probs.append(p)
            labels.append(qm[3].astype(float))
    return calibration_metrics(np.concatenate(probs), np.concatenate(labels))


def eval_oracle_full(K, pool, rho_range, rel_range, seed, n_ep=30, Q=30):
    rng = SafeRNG(seed)
    probs, labels = [], []
    for _ in range(n_ep):
        for _ in range(6):
            src = sample_source(rng, pool, rho_range, rel_range)
            c, rho, r = src
            qm = generate(src, Q, rng)
            for i in range(Q):
                lo = oracle_log_odds(int(qm[0][i]), int(qm[1][i]),
                                     int(qm[2][i]), c, rho, r)
                probs.append(1.0 / (1.0 + np.exp(-lo)))
                labels.append(int(qm[3][i]))
    return calibration_metrics(np.array(probs), np.array(labels))


# ============================================================
# SEED RUNNERS (one seed = one unit of parallel work)
# ============================================================

def _row(seed, extra, method, met):
    r = {'seed': seed, 'method': method}
    r.update(extra)
    r.update(met)
    return r


def run_seed_A(seed):
    """Experiment A (rho sweep) with calibration + BetaBinomial."""
    rho_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    m_attn = train_neural(make_model(MetaAttn, seed), seed=seed,
                          rho_range=(0.5, 1.0))
    m_mean = train_neural(make_model(MetaMean, seed), seed=seed,
                          rho_range=(0.5, 1.0))
    m_nos = train_neural(make_model(NoSupport, seed), seed=seed,
                         rho_range=(0.5, 1.0))
    m_uns = train_neural(make_model(UnsignedScalar, seed), seed=seed,
                         rho_range=(0.5, 1.0))
    m_sig = train_neural(make_model(SignedScalar, seed), seed=seed,
                         rho_range=(0.5, 1.0))
    rows = []
    for rho in rho_values:
        rho_range = (rho - 0.01, rho + 0.01)
        rel_range = (0.7, 0.95)
        s_eval = seed * 100 + int(rho * 100)
        extra = {'rho': rho}
        for name, m in [('MetaAttn', m_attn), ('MetaMean', m_mean),
                        ('NoSupport', m_nos), ('UnsignedScalar', m_uns),
                        ('SignedScalar', m_sig)]:
            met = eval_neural_full(m, 10, None, rho_range, rel_range, s_eval)
            rows.append(_row(seed, extra, name, met))
        met = eval_betabinomial_full(10, None, rho_range, rel_range, s_eval)
        rows.append(_row(seed, extra, 'BetaBinomial', met))
        met = eval_bayes_full(10, None, rho_range, rel_range, s_eval)
        rows.append(_row(seed, extra, 'Bayes', met))
    return rows


def run_seed_B(seed):
    """Experiment B (representation ablation) with calibration +
    DeepAttn, TransformerMeta, BetaBinomial."""
    m_ctx = train_neural(make_model(ContextConditioned, seed), seed=seed)
    m_sig = train_neural(make_model(SignedScalar, seed), seed=seed)
    m_uns = train_neural(make_model(UnsignedScalar, seed), seed=seed)
    m_attn = train_neural(make_model(MetaAttn, seed), seed=seed)
    m_nos = train_neural(make_model(NoSupport, seed), seed=seed)
    m_dec = train_neural(make_model(MetaAttn, seed, sup_in=5, qry_in=4),
                         feat_mode='dec', seed=seed)
    m_dctx = train_neural(make_model(MetaAttn, seed, sup_in=13, qry_in=12),
                          feat_mode='decctx', seed=seed)
    m_deep = train_neural(make_model(DeepAttn, seed), seed=seed)
    m_tr = train_neural(make_model(TransformerMeta, seed), seed=seed)

    rho_range = (0.99, 1.0)
    rel_range = (0.7, 0.95)
    s_eval = seed * 1000 + 1
    extra = {}
    results = {
        'ContentOnly': eval_neural_full(m_nos, 10, None, rho_range,
                                        rel_range, s_eval),
        'UnsignedScalar': eval_neural_full(m_uns, 10, None, rho_range,
                                           rel_range, s_eval),
        'SignedScalar': eval_neural_full(m_sig, 10, None, rho_range,
                                         rel_range, s_eval),
        'ContextConditioned': eval_neural_full(m_ctx, 10, None, rho_range,
                                               rel_range, s_eval),
        'AttentionCalibrator': eval_neural_full(m_attn, 10, None, rho_range,
                                                rel_range, s_eval),
        'MetaAttn+Dec': eval_neural_full(m_dec, 10, None, rho_range,
                                         rel_range, s_eval, feat_mode='dec'),
        'MetaAttn+DecCtx': eval_neural_full(m_dctx, 10, None, rho_range,
                                            rel_range, s_eval,
                                            feat_mode='decctx'),
        'DeepAttn': eval_neural_full(m_deep, 10, None, rho_range,
                                     rel_range, s_eval),
        'TransformerMeta': eval_neural_full(m_tr, 10, None, rho_range,
                                            rel_range, s_eval),
        'BetaBinomial': eval_betabinomial_full(10, None, rho_range,
                                               rel_range, s_eval),
        'Bayes': eval_bayes_full(10, None, rho_range, rel_range, s_eval),
        'Oracle': eval_oracle_full(10, None, rho_range, rel_range, s_eval),
    }
    return [_row(seed, extra, name, met) for name, met in results.items()]


def run_seed_C(seed):
    """Experiment C (K-sweep, held-out conventions) with calibration."""
    K_values = [1, 2, 5, 10, 20, 50]
    m_attn = train_neural(make_model(MetaAttn, seed), pool=TRAIN_CONVS,
                          rho_range=(0.9, 1.0), seed=seed)
    m_mean = train_neural(make_model(MetaMean, seed), pool=TRAIN_CONVS,
                          rho_range=(0.9, 1.0), seed=seed)
    m_nos = train_neural(make_model(NoSupport, seed), pool=TRAIN_CONVS,
                         rho_range=(0.9, 1.0), seed=seed)
    m_deep = train_neural(make_model(DeepAttn, seed), pool=TRAIN_CONVS,
                          rho_range=(0.9, 1.0), seed=seed)
    rows = []
    for K in K_values:
        s_eval = seed * 100 + K
        extra = {'K': K}
        for name, m in [('MetaAttn', m_attn), ('MetaMean', m_mean),
                        ('NoSupport', m_nos), ('DeepAttn', m_deep)]:
            met = eval_neural_full(m, K, HELD_CONVS, (0.99, 1.0),
                                   (0.7, 0.95), s_eval)
            rows.append(_row(seed, extra, name, met))
        met = eval_betabinomial_full(K, HELD_CONVS, (0.99, 1.0),
                                    (0.7, 0.95), s_eval)
        rows.append(_row(seed, extra, 'BetaBinomial', met))
        met = eval_bayes_full(K, HELD_CONVS, (0.99, 1.0), (0.7, 0.95),
                              s_eval, candidates=list(range(N_CONV)))
        rows.append(_row(seed, extra, 'BayesUniv', met))
        met = eval_bayes_full(K, HELD_CONVS, (0.99, 1.0), (0.7, 0.95),
                              s_eval, candidates=HELD_CONVS)
        rows.append(_row(seed, extra, 'BayesFam', met))
    return rows


def run_seed_E(seed):
    """Experiment E (multi-agent consensus) with calibration."""
    def consensus(neural_attn, n_agents, n_trials=60, K=10, seed=0):
        rng = SafeRNG(seed)
        methods = ['NaiveMean', 'MajorityVote', 'MetaAttn', 'Nonparam',
                   'Bayes', 'Oracle']
        probs = {m: [] for m in methods}
        labels = []
        with torch.no_grad():
            for _ in range(n_trials):
                Z_true = int(rng.random() < 0.5)
                labels.append(Z_true)
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

                vote_frac = float(np.mean(raw_msgs))
                probs['NaiveMean'].append(vote_frac)
                probs['MajorityVote'].append(vote_frac)

                p_at = F.softmax(neural_attn(S_b, Q_b),
                                 dim=-1)[0, :, 0, 1].cpu().numpy()
                probs['MetaAttn'].append(float(p_at.mean()))

                p_np, lo_bayes, lo_or = [], [], []
                for j in range(n_agents):
                    sup = supports[j].numpy()
                    q = queries[j].numpy()[0]
                    p_np.append(betabinomial_probs(
                        sup[:, 0], sup[:, 1], sup[:, 2], sup[:, 3],
                        np.array([int(q[0])]), np.array([int(q[1])]),
                        np.array([int(q[2])]))[0])
                    post = bayes_post(sup[:, 0], sup[:, 1], sup[:, 2],
                                      sup[:, 3], list(range(N_CONV)))
                    # Sum per-agent log-odds, then ONE sigmoid: the original
                    # experiment's decision rule, now as a calibrated prob.
                    lo_bayes.append(bayes_log_odds(
                        post, int(q[0]), int(q[1]),
                        int(q[2]), list(range(N_CONV))))
                    c_j, rho_j, r_j = params[j]
                    lo_or.append(oracle_log_odds(int(q[0]), int(q[1]),
                                                int(q[2]), c_j, rho_j, r_j))
                # Nonparam stays a mean: it is a heuristic consensus of
                # per-agent estimates, with no independence model behind it.
                probs['Nonparam'].append(float(np.mean(p_np)))
                probs['Bayes'].append(float(expit(np.sum(lo_bayes))))
                probs['Oracle'].append(float(expit(np.sum(lo_or))))
        out = {}
        for m in methods:
            out[m] = calibration_metrics(np.array(probs[m]),
                                         np.array(labels))
        return out

    m_attn = train_neural(make_model(MetaAttn, seed), pool=None,
                          rho_range=(0.7, 1.0), seed=seed)
    rows = []
    for n_agents in [2, 3, 4, 6, 8, 12]:
        res = consensus(m_attn, n_agents=n_agents, n_trials=60, K=10,
                        seed=seed * 100 + n_agents)
        for method, met in res.items():
            rows.append(_row(seed, {'n_agents': n_agents}, method, met))
    return rows


# ============================================================
# MAIN
# ============================================================

_RUNNERS = {'A': run_seed_A, 'B': run_seed_B,
            'C': run_seed_C, 'E': run_seed_E}


def _run(job):
    tag, seed = job
    try:
        return (tag, seed, _RUNNERS[tag](seed), None)
    except Exception as e:  # noqa: BLE001 - one bad seed must not kill the run
        import traceback
        return (tag, seed, [], f'{e}\n{traceback.format_exc()[-500:]}')


def _init_worker():
    torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='2-seed smoke of Exp B only')
    ap.add_argument('--only', default='ABCE',
                    help='subset of experiments to run, e.g. AB or E')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--resume', action='store_true',
                    help='skip seeds already present in checkpoint CSVs')
    ap.add_argument('--max-jobs', type=int, default=None,
                    help='run at most N pending seed-jobs, then exit '
                         '(for chunked foreground execution)')
    ap.add_argument('--seeds', type=int, default=100,
                    help='number of seeds per experiment '
                         '(default 100; 30 is enough for ±2pp CIs)')
    args = ap.parse_args()

    names = {'A': 'expA_n100', 'B': 'expB_n100',
             'C': 'expC_n100', 'E': 'expE_n100'}

    if args.quick:
        jobs = [('B', s) for s in range(2)]
    else:
        jobs = []
        if 'A' in args.only:
            jobs += [('A', s) for s in range(args.seeds)]
        if 'B' in args.only:
            jobs += [('B', s) for s in range(args.seeds)]
        if 'C' in args.only:
            jobs += [('C', s) for s in range(args.seeds)]
        if 'E' in args.only:
            jobs += [('E', s) for s in range(args.seeds)]

    # Resume: drop jobs whose seed is already checkpointed on disk.
    if args.resume:
        kept = []
        for tag, seed in jobs:
            path = f'{RESULTS_DIR}/{names[tag]}.csv'
            if os.path.exists(path):
                done_seeds = set(
                    pd.read_csv(path, usecols=['seed'])['seed'].tolist())
                if seed in done_seeds:
                    continue
            kept.append((tag, seed))
        print(f'Resume: {len(jobs) - len(kept)} seeds already done, '
              f'{len(kept)} remaining.')
        jobs = kept

    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
        print(f'Chunked run: first {len(jobs)} pending seed-jobs.')

    t0 = time.time()
    print(f'Launching {len(jobs)} seed-jobs on {args.workers} workers...')
    errors = []
    done = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for tag, seed, rows, err in pool.imap_unordered(_run, jobs):
            done += 1
            if err:
                errors.append((tag, seed, err))
            elif rows:
                # Per-seed checkpoint: a dead run loses nothing completed.
                path = f'{RESULTS_DIR}/{names[tag]}.csv'
                df = pd.DataFrame(rows)
                df.to_csv(path, mode='a',
                          header=not os.path.exists(path), index=False)
            if done % 10 == 0 or done == len(jobs):
                el = (time.time() - t0) / 60
                print(f'  [{done}/{len(jobs)}] {el:.1f} min elapsed',
                      flush=True)

    for tag in names:
        path = f'{RESULTS_DIR}/{names[tag]}.csv'
        if os.path.exists(path):
            n = len(pd.read_csv(path, usecols=['seed']))
            print(f'{tag}: {n} rows checkpointed -> {path}')

    if errors:
        print(f'\n{len(errors)} seed-jobs failed:')
        for tag, seed, err in errors[:5]:
            print(f'  {tag} seed {seed}: {err[:200]}')
    print(f'\nDone in {(time.time()-t0)/3600:.2f} hours.')


if __name__ == '__main__':
    main()
