"""Audit fix experiments, part 2: CtxLogReg K-sweep with data backing.

The paper quotes '0.63 -> 0.71' for K = 2 -> 50 at lambda = 0.1, 10
seeds, backed only by prose in an internal note (and that note's sweep
used 5 seeds and K in {10, 30, 50}). This runs the repo's own
eval_ctxlogreg (code/ctxlogreg_baseline.py, ExpB per-source protocol,
s_eval = seed*1000 + 1) for K in {2, 5, 10, 20, 50}, lambda = 0.1,
seeds 0..9, and records every per-seed accuracy.

Sanity: the K=10 rows must bit-match results/expB_ctxlogreg.csv at
lambda=0.1 (checked by run_all_audit_fixes.py).

Usage (from repo root):
  .venv/bin/python code/audit_fixes/ctxlogreg_ksweep.py
"""
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
import numpy as np  # noqa: E402

from ctxlogreg_baseline import eval_ctxlogreg  # noqa: E402

KS = [2, 5, 10, 20, 50]
LAM = 0.1
SEEDS = list(range(10))


def job(args):
    K, seed = args
    acc = eval_ctxlogreg(K, None, (0.99, 1.0), (0.7, 0.95),
                         seed * 1000 + 1, LAM)
    return {'K': K, 'seed': seed, 'acc': acc}


def main():
    t0 = time.time()
    jobs = [(K, s) for K in KS for s in SEEDS]
    with Pool(2) as pool:
        rows = pool.map(job, jobs)
    elapsed = time.time() - t0
    out = {'experiment': 'ctxlogreg_ksweep',
           'protocol': {'lambda': LAM, 'seeds': SEEDS, 'K_values': KS,
                        'rho_range': (0.99, 1.0), 'rel_range': (0.7, 0.95),
                        's_eval': 'seed*1000+1', 'n_ep': 30, 'Q': 30},
           'runtime_seconds': elapsed,
           'per_seed': sorted(rows, key=lambda r: (r['K'], r['seed']))}
    path = 'results/audit_fixes/ctxlogreg_ksweep_raw.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {path} in {elapsed:.1f}s')
    for K in KS:
        accs = [r['acc'] for r in rows if r['K'] == K]
        print(f'  K={K:3d}  mean={np.mean(accs):.4f} std={np.std(accs, ddof=1):.4f}')


if __name__ == '__main__':
    main()
