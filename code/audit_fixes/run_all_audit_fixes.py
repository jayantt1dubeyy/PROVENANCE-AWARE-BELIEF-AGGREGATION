"""Aggregate the audit-fix raw outputs into results/audit_fixes/
experiments.json: means with 95% t-CIs over seeds, paired deltas vs
Bayes with CIs and Cohen's dz, convention-recovery rates, the
CtxLogReg K-sweep, parameter counts, and bit-exactness sanity checks
against the banked CSVs.

Usage (from repo root, after the three part scripts have run):
  .venv/bin/python code/audit_fixes/run_all_audit_fixes.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

FIX = 'results/audit_fixes'


def summ(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
    return {'mean': mean, 'std': std, 'ci95_half_width': half,
            'ci95': [mean - half, mean + half], 'n': int(n)}


def paired(a, b):
    """Paired per-seed delta a-b with 95% t-CI and Cohen's dz."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    s = summ(d)
    s['cohens_dz'] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 \
        else float('inf')
    return s


def main():
    out = {}

    # ---------- Exp B plug-in ablations ----------
    raw = json.load(open(f'{FIX}/expb_plugin_raw.json'))
    per_seed = raw['per_seed']
    methods = list(per_seed[0]['acc'].keys())
    acc = {m: [r['acc'][m] for r in per_seed] for m in methods}
    expb = {'protocol': raw['protocol'],
            'runtime_seconds': raw['runtime_seconds'],
            'methods': {m: {**summ(acc[m]), 'per_seed': acc[m]}
                        for m in methods},
            'paired_vs_bayes': {
                m: paired(acc['Bayes'], acc[m])
                for m in methods if m != 'Bayes'},
            'convention_recovery': {
                'grid_posterior_map': summ(
                    [r['grid_map_recovery'] for r in per_seed]),
                'em_map': summ([r['em_map_recovery'] for r in per_seed]),
                'pooled': {
                    'grid_posterior_map': float(np.average(
                        [r['grid_map_recovery'] for r in per_seed],
                        weights=[r['n_sources'] for r in per_seed])),
                    'em_map': float(np.average(
                        [r['em_map_recovery'] for r in per_seed],
                        weights=[r['n_sources'] for r in per_seed])),
                    'n_sources_total': int(sum(
                        r['n_sources'] for r in per_seed))}}}

    # ---------- sanity checks vs banked CSVs ----------
    sanity = {}
    banked_b = pd.read_csv('results/expB_n100.csv')
    bb = banked_b[banked_b.method == 'Bayes'].sort_values('seed')
    mine = pd.Series(acc['Bayes'], index=[r['seed'] for r in per_seed])
    diffs = (mine.loc[bb.seed.values].values - bb.acc.values)
    sanity['bayes_vs_expB_n100_csv'] = {
        'max_abs_diff': float(np.abs(diffs).max()),
        'banked_mean': float(bb.acc.mean()),
        'reproduced': bool(np.abs(diffs).max() == 0.0)}
    banked_e = pd.read_csv('results/sens_emdecoder.csv')
    ss = banked_b[banked_b.method == 'SignedScalar'].sort_values('seed')
    d_ss = np.abs(np.array(acc['LogOddsScalar']) - ss.acc.values)
    sanity['logodds_scalar_vs_banked_signed_scalar'] = {
        'max_abs_diff': float(d_ss.max()),
        'per_seed_identical': bool(d_ss.max() == 0.0),
        'detail': ('Decision-identical on all 30 seeds. sample_batch '
                   'consumes the RNG in the same order as the per-source '
                   'eval loop (identical streams), and with the trained '
                   'tanh(raw_w) > 0 the SignedScalar decision reduces to '
                   'sign(2*cons - 1) * (2*m_q - 1) -- the same sign rule '
                   'as w = log(p/(1-p)) with p = cons (the [0.05, 0.95] '
                   'clip never changes a sign). The parameter-free '
                   'log-odds scalar therefore IS the trained SignedScalar '
                   'as a decision rule on this protocol.')}
    for meth in ['EMDecoder', 'EMDecoder_plugin']:
        be = banked_e[banked_e.method == meth].sort_values('seed')
        mine_s = pd.Series(acc[meth], index=[r['seed'] for r in per_seed])
        d = (mine_s.loc[be.seed.values].values - be.acc.values)
        sanity[f'{meth.lower()}_vs_sens_emdecoder_csv'] = {
            'max_abs_diff': float(np.abs(d).max()),
            'banked_mean_10_seeds': float(be.acc.mean()),
            'reproduced': bool(np.abs(d).max() == 0.0)}
    out['expb_plugin_ablations'] = expb

    # ---------- CtxLogReg K-sweep ----------
    rawc = json.load(open(f'{FIX}/ctxlogreg_ksweep_raw.json'))
    ks = {}
    for K in rawc['protocol']['K_values']:
        vals = [r['acc'] for r in rawc['per_seed'] if r['K'] == K]
        ks[str(K)] = {**summ(vals), 'per_seed': vals}
    out['ctxlogreg_ksweep'] = {'protocol': rawc['protocol'],
                               'runtime_seconds': rawc['runtime_seconds'],
                               'by_K': ks}
    banked_c = pd.read_csv('results/expB_ctxlogreg.csv')
    bc = banked_c[(banked_c['lambda'] == 0.1)].sort_values('seed')
    mine_c = [r['acc'] for r in sorted(
        [r for r in rawc['per_seed'] if r['K'] == 10],
        key=lambda r: r['seed'])]
    d = np.abs(np.array(mine_c) - bc.acc.values)
    sanity['ctxlogreg_K10_vs_expB_ctxlogreg_csv'] = {
        'max_abs_diff': float(d.max()),
        'banked_mean': float(bc.acc.mean()),
        'reproduced': bool(d.max() == 0.0),
        'detail': ('9 of 10 seeds bit-exact. Seed 5 alone differs by '
                   '9/5400 queries (mine 0.629259 vs banked 0.627593); '
                   'a serial rerun on this machine confirms the current '
                   'code produces 0.629259, so the banked seed-5 '
                   'lambda=0.1 row does not reproduce in the repo\'s '
                   'current environment (one source\'s IRLS fit lands '
                   'on the other side of a few borderline queries). '
                   'Effect on the 10-seed mean: +0.0002.')}

    # ---------- parameter counts ----------
    out['param_counts'] = json.load(open(f'{FIX}/param_counts_raw.json'))

    out['sanity_checks'] = sanity
    with open(f'{FIX}/experiments.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {FIX}/experiments.json')
    print(json.dumps(sanity, indent=1))


if __name__ == '__main__':
    main()
