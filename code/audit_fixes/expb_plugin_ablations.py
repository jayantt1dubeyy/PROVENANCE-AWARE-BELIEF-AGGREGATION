"""Audit fix experiments, part 1: Exp B plug-in ablations + recovery rates.

Protocol: EXACT Exp B per-source evaluation (K=10 support, Q=30 queries,
n_ep=30 episodes x 6 sources, rho ~ U(0.99, 1.0), r ~ U(0.7, 0.95),
s_eval = seed*1000 + 1, seeds 0..29). One RNG stream per seed, generated
with the repo's own SafeRNG / sample_source / generate in the same
consumption order as pvi_experiments.eval_bayes and
new_baselines.eval_emdecoder, so every method below sees IDENTICAL
streams and the sanity methods must reproduce the banked CSVs.

Methods (all decisions from the same support set, per source):
  Bayes          repo grid posterior + posterior-predictive log-odds
                 (sanity: banked expB_n100.csv Bayes, 30 seeds, 0.7590).
  GridMAP        MAP cell (c, rho, r) of the grid posterior; queries
                 decided by the repo oracle_log_odds at that cell.
  PostMean       MAP convention c* (convention of the joint MAP cell);
                 rho-bar = E[rho | c*, support], r-bar = E[r | c*,
                 support] under the grid posterior; queries decided by
                 oracle_log_odds at (c*, rho-bar, r-bar).
  EMDecoder      repo em_fit point estimates; original rule
                 lo = log(r-hat/(1-r-hat)) * (2D - 1), D = MAP-decoded
                 bit (drops rho-hat) -- verbatim from eval_emdecoder.
                 (sanity: banked sens_emdecoder.csv, 10 seeds, 0.6829.)
  EMDecoder_plugin  same EM estimates, oracle_log_odds decision.
                 (sanity: banked 0.7316.)
  LogOddsScalar  NEW parameter-free scalar baseline mirroring
                 SignedScalar's decision form logit = w_s * (2 m_q - 1),
                 pred = 1 iff logit > 0, but with the per-source weight
                 set analytically (no learned global scale):
                   p_s   = support agreement rate mean(m_sup == z_sup)
                           (the same 'cons' quantity SignedScalar uses),
                   p     = clip(p_s, 0.05, 0.95),
                   w_s   = log(p / (1 - p)).

Also records, per source, whether the true convention is recovered by
(a) the grid posterior's MAP cell and (b) EM's MAP convention.

Usage (from repo root):
  .venv/bin/python code/audit_fixes/expb_plugin_ablations.py
Outputs: results/audit_fixes/expb_plugin_raw.json (merged by merge step
in run_all_audit_fixes.py; this script writes its own partial too).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
import numpy as np  # noqa: E402

from pvi_experiments import (  # noqa: E402
    N_CONV, R_GRID, RHO_GRID, SafeRNG, apply_convention, bayes_log_odds,
    bayes_post, generate, oracle_log_odds, sample_source,
)
from new_baselines import em_fit  # noqa: E402

CANDS = list(range(N_CONV))
K, Q, N_EP = 10, 30, 30
RHO_RANGE, REL_RANGE = (0.99, 1.0), (0.7, 0.95)
METHODS = ['Bayes', 'GridMAP', 'PostMean', 'EMDecoder', 'EMDecoder_plugin',
           'LogOddsScalar']
CLIP_LO, CLIP_HI = 0.05, 0.95


def run_seed(seed):
    rng = SafeRNG(seed * 1000 + 1)
    correct = {m: 0 for m in METHODS}
    total = 0
    grid_hit = em_hit = n_src = 0
    for _ in range(N_EP):
        for _ in range(6):
            src = sample_source(rng, None, RHO_RANGE, REL_RANGE)
            c_true = src[0]
            sm = generate(src, K, rng)
            qm = generate(src, Q, rng)
            m_sup, a1_sup, a2_sup, z_sup = sm

            # --- grid posterior (repo machinery) ---
            post = bayes_post(m_sup, a1_sup, a2_sup, z_sup, CANDS)
            ci, ri, gi = np.unravel_index(int(np.argmax(post)), post.shape)
            c_map = CANDS[ci]
            rho_map, r_map = float(RHO_GRID[ri]), float(R_GRID[gi])
            grid_hit += int(c_map == c_true)
            # posterior means of (rho, r) conditional on the MAP convention
            cond = post[ci] / post[ci].sum()
            rho_bar = float((cond * RHO_GRID[:, None]).sum())
            r_bar = float((cond * R_GRID[None, :]).sum())

            # --- EM fit (repo) ---
            c_em, rho_em, r_em, _q = em_fit(m_sup, a1_sup, a2_sup, z_sup)
            em_hit += int(c_em == c_true)
            n_src += 1
            w_em = float(np.log(r_em / (1 - r_em)))

            # --- LogOddsScalar weight (support-only) ---
            cons = float((m_sup == z_sup).mean())
            p_clip = min(max(cons, CLIP_LO), CLIP_HI)
            w_los = float(np.log(p_clip / (1 - p_clip)))

            for i in range(Q):
                m_q, a1_q, a2_q = int(qm[0][i]), int(qm[1][i]), int(qm[2][i])
                z_q = int(qm[3][i])
                total += 1
                lo = bayes_log_odds(post, m_q, a1_q, a2_q, CANDS)
                correct['Bayes'] += int((1 if lo > 0 else 0) == z_q)
                lo = oracle_log_odds(m_q, a1_q, a2_q, c_map, rho_map, r_map)
                correct['GridMAP'] += int((1 if lo > 0 else 0) == z_q)
                lo = oracle_log_odds(m_q, a1_q, a2_q, c_map, rho_bar, r_bar)
                correct['PostMean'] += int((1 if lo > 0 else 0) == z_q)
                D = int(apply_convention(c_em, np.array([m_q]),
                                         np.array([a1_q]),
                                         np.array([a2_q]))[0])
                lo = w_em * (2 * D - 1)
                correct['EMDecoder'] += int((1 if lo > 0 else 0) == z_q)
                lo = oracle_log_odds(m_q, a1_q, a2_q, c_em, rho_em, r_em)
                correct['EMDecoder_plugin'] += int((1 if lo > 0 else 0) == z_q)
                logit = w_los * (2 * m_q - 1)
                correct['LogOddsScalar'] += int((1 if logit > 0 else 0) == z_q)
    out = {'seed': seed,
           'acc': {m: correct[m] / total for m in METHODS},
           'grid_map_recovery': grid_hit / n_src,
           'em_map_recovery': em_hit / n_src,
           'n_sources': n_src}
    return out


def main():
    seeds = list(range(30))
    t0 = time.time()
    with Pool(2) as pool:
        results = pool.map(run_seed, seeds)
    elapsed = time.time() - t0
    results.sort(key=lambda r: r['seed'])
    out = {'experiment': 'expb_plugin_ablations',
           'protocol': {'K': K, 'Q': Q, 'n_ep': N_EP,
                        'rho_range': RHO_RANGE, 'rel_range': REL_RANGE,
                        's_eval': 'seed*1000+1', 'seeds': seeds},
           'runtime_seconds': elapsed,
           'per_seed': results}
    path = 'results/audit_fixes/expb_plugin_raw.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {path} in {elapsed:.1f}s')
    for m in METHODS:
        accs = [r['acc'][m] for r in results]
        print(f'  {m:18s} mean={np.mean(accs):.4f} std={np.std(accs, ddof=1):.4f}')
    for key in ['grid_map_recovery', 'em_map_recovery']:
        vals = [r[key] for r in results]
        print(f'  {key:18s} mean={np.mean(vals):.4f} std={np.std(vals, ddof=1):.4f}')


if __name__ == '__main__':
    main()
