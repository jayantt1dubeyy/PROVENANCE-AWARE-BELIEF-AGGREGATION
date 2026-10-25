"""PVI-REAL Study B: real-data experiment on CrowdTruth Medical
Relation Extraction (RelEx), the dataset found to have persistent
worker IDs, expert gold labels, and genuine context variables.

Binary reduction (pre-declared):
  item  = one RelEx unit (sentence + term pair) whose candidate
          relation is RO-cause_of / RO-has_causative_agent (family
          'cause', affirming answer [CAUSES]) or RO-may_treat (family
          'treat', affirming answer [TREATS]).
  z     = expert verdict from ground_truth_{cause,treat}.csv joined on
          (numeric SID, family); expert 1 -> z=1, -1 -> z=0. Ambiguous
          joins (conflicting expert values for the same key) dropped.
  m_ij  = 1 iff worker j's step-1 answer affirms the candidate family.
  contexts: a1 = family (cause 0 / treat 1);
            a2 = recruitment channel group (largest channel = 1).
  sources = workers with >= 40 judgments on expert-covered items.

Protocol: 20 random item splits (50% support / 50% query, stratified
by family and z). All methods see the same support judgments.
Methods: Majority, ScalarLogOdds (one trust number per source),
ContextLogOdds (per-source x context-cell log-odds weights, shrunk
toward the source's global value, kappa=8), ContextLogOdds with a2
shuffled within worker (placebo), DawidSkene EM (context-free,
iterated to convergence), OracleCtx (per-source-cell accuracies from
ALL data; privileged ceiling).
Primary: per-split mean log loss and accuracy on query items; paired
differences vs ScalarLogOdds across splits.
Also: H1 diagnostic -- per-source accuracy gaps across contexts.
Outputs: results/pvi_real/studyB_summary.json + .md
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
DATA = os.path.join(ROOT, 'data', 'real', 'crowdtruth_medical')
OUT = os.path.join(ROOT, 'results', 'pvi_real')
os.makedirs(OUT, exist_ok=True)

CAUSE_TYPES = {'RO-cause_of', 'RO-has_causative_agent'}
TREAT_TYPES = {'RO-may_treat'}
AFFIRM = {'cause': '[CAUSES]', 'treat': '[TREATS]'}


def load():
    gt = {}
    for fam, fn in [('cause', 'ground_truth_cause.csv'),
                    ('treat', 'ground_truth_treat.csv')]:
        df = pd.read_csv(os.path.join(DATA, fn))
        df = df[df['expert'].isin([-1, 1])]
        for sid, exp in zip(df['SID'], df['expert']):
            key = (int(sid), fam)
            val = 1 if exp == 1 else 0
            if key in gt and gt[key] != val:
                gt[key] = None  # conflicting -> drop
            else:
                gt[key] = val
    gt = {k: v for k, v in gt.items() if v is not None}
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, 'raw', 'RelEx', '*.csv'))):
        head = pd.read_csv(f, nrows=0).columns
        cand = 'rel-type' if 'rel-type' in head else 'relation'
        df = pd.read_csv(f, low_memory=False,
                         usecols=['_unit_id', '_worker_id', '_channel',
                                  cand,
                                  'step_1_select_the_valid_relations',
                                  'sent_id'])
        df = df.rename(columns={cand: 'rel-type'})
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw['family'] = np.where(raw['rel-type'].isin(CAUSE_TYPES), 'cause',
                             np.where(raw['rel-type'].isin(TREAT_TYPES),
                                      'treat', None))
    raw = raw[raw['family'].notna() & raw['sent_id'].notna()].copy()
    raw['sid'] = pd.to_numeric(
        raw['sent_id'].astype(str).str.extract(r'^(\d+)')[0],
        errors='coerce')
    raw = raw[raw['sid'].notna()].copy()
    raw['sid'] = raw['sid'].astype(int)
    raw['z'] = [gt.get((s, fam)) for s, fam in zip(raw['sid'], raw['family'])]
    raw = raw[raw['z'].notna()].copy()
    raw['z'] = raw['z'].astype(int)
    raw['m'] = [1 if ans == AFFIRM[fam] else 0 for ans, fam in
                zip(raw['step_1_select_the_valid_relations'], raw['family'])]
    top_channel = raw['_channel'].value_counts().idxmax()
    raw['a2'] = (raw['_channel'] == top_channel).astype(int)
    raw['a1'] = (raw['family'] == 'treat').astype(int)
    counts = raw.groupby('_worker_id').size()
    keep = counts[counts >= 40].index
    raw = raw[raw['_worker_id'].isin(keep)].copy()
    raw['item'] = raw['sid'].astype(str) + '|' + raw['family']
    return raw, top_channel


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logodds_weight(hits, n, prior_hits, prior_n, kappa=8.0):
    """Shrunk log-odds of accuracy: cell estimate pulled to the prior."""
    ph = (hits + kappa * (prior_hits / max(prior_n, 1))) / (n + kappa)
    ph = min(max(ph, 0.02), 0.98)
    return np.log(ph / (1 - ph))


def ds_em(items, max_iter=100, tol=1e-6):
    """Iterated Dawid-Skene for binary labels. items: list of
    (item_id, worker, m). Returns per-item posterior P(z=1)."""
    item_ids = sorted({i for i, _, _ in items})
    workers = sorted({w for _, w, _ in items})
    wi = {w: k for k, w in enumerate(workers)}
    ii = {it: k for k, it in enumerate(item_ids)}
    M = np.full((len(item_ids), len(workers)), -1)
    for it, w, m in items:
        M[ii[it], wi[w]] = m
    obs = M >= 0
    post = np.where(obs, M, 0.5).mean(axis=1)
    post = np.clip(post, 0.05, 0.95)
    for _ in range(max_iter):
        a = np.zeros(len(workers))  # P(m=1|z=1)
        b = np.zeros(len(workers))  # P(m=0|z=0)
        for j in range(len(workers)):
            o = obs[:, j]
            mj = np.where(o, M[:, j], 0)
            a[j] = (np.sum(post[o] * mj[o]) + 1) / (np.sum(post[o]) + 2)
            b[j] = (np.sum((1 - post[o]) * (1 - mj[o])) + 1) / \
                   (np.sum(1 - post[o]) + 2)
        ll1 = np.where(obs, M * np.log(np.clip(a, 1e-9, 1)) +
                       (1 - M) * np.log(np.clip(1 - a, 1e-9, 1)), 0).sum(axis=1)
        ll0 = np.where(obs, M * np.log(np.clip(1 - b, 1e-9, 1)) +
                       (1 - M) * np.log(np.clip(b, 1e-9, 1)), 0).sum(axis=1)
        prev = np.clip(post.mean(), 0.01, 0.99)
        new = sigmoid(np.log(prev / (1 - prev)) + ll1 - ll0)
        if np.max(np.abs(new - post)) < tol:
            post = new
            break
        post = new
    return {it: float(post[ii[it]]) for it in item_ids}


def run_split(df, rng):
    items = df[['item', 'z', 'a1']].drop_duplicates()
    sup_items, qry_items = [], []
    for (fam, z), grp in items.groupby(['a1', 'z']):
        idx = grp['item'].tolist()
        rng.shuffle(idx)
        half = len(idx) // 2
        sup_items += idx[:half]
        qry_items += idx[half:]
    sup = df[df['item'].isin(set(sup_items))]
    qry = df[df['item'].isin(set(qry_items))]
    zq = qry.groupby('item')['z'].first()
    # per-source global accuracy on support
    g = sup.assign(hit=(sup['m'] == sup['z']).astype(float)) \
           .groupby('_worker_id')['hit'].agg(['sum', 'count'])
    gw = {w: logodds_weight(r['sum'], r['count'], g['sum'].sum(),
                            g['count'].sum()) for w, r in g.iterrows()}
    # per-source-cell accuracy on support
    cells = {}
    for (w, a1, a2), grp in sup.groupby(['_worker_id', 'a1', 'a2']):
        cells[(w, a1, a2)] = (float((grp['m'] == grp['z']).sum()), len(grp))
    global_prior = (g['sum'].sum(), g['count'].sum())

    def ctx_weight(w, a1, a2, shuffle=False):
        gg = g.loc[w] if w in g.index else None
        ph, pn = (gg['sum'], gg['count']) if gg is not None else global_prior
        key = (w, a1, a2)
        if key in cells:
            h, n = cells[key]
            return logodds_weight(h, n, ph, pn)
        return logodds_weight(ph, pn, *global_prior)

    def aggregate(weight_fn):
        lo = {}
        for (it, w, m, a1, a2) in zip(qry['item'], qry['_worker_id'],
                                      qry['m'], qry['a1'], qry['a2']):
            lo[it] = lo.get(it, 0.0) + weight_fn(w, a1, a2) * (2 * m - 1)
        return {it: float(sigmoid(v)) for it, v in lo.items()}

    preds = {}
    maj = qry.groupby('item')['m'].mean().clip(0.02, 0.98)
    preds['Majority'] = maj.to_dict()
    preds['ScalarLogOdds'] = aggregate(lambda w, a1, a2: gw.get(w, 0.0))
    preds['ContextLogOdds'] = aggregate(lambda w, a1, a2: ctx_weight(w, a1, a2))
    # Placebo: permute the FAMILY context (a1) across query items,
    # breaking the item-context link. (Channel a2 is constant within
    # worker in this dataset, so an a2 shuffle is vacuous -- checked:
    # 0 workers have >=15 judgments in both channel groups.)
    a1_perm = rng.permutation(qry['a1'].values)
    lo = {}
    for (it, w, m, a1, a2) in zip(qry['item'], qry['_worker_id'],
                                  qry['m'], a1_perm, qry['a2']):
        lo[it] = lo.get(it, 0.0) + ctx_weight(w, a1, a2) * (2 * m - 1)
    preds['ContextLogOdds_placeboA1'] = {it: float(sigmoid(v))
                                         for it, v in lo.items()}
    # Shrinkage sensitivity for ContextLogOdds (all values reported).
    preds_kappa = {}
    for kappa in [4.0, 20.0, 50.0]:
        def cw(w, a1, a2, kappa=kappa):
            gg = g.loc[w] if w in g.index else None
            ph, pn = (gg['sum'], gg['count']) if gg is not None else global_prior
            if (w, a1, a2) in cells:
                h, n = cells[(w, a1, a2)]
                return logodds_weight(h, n, ph, pn, kappa=kappa)
            return logodds_weight(ph, pn, *global_prior)
        lo = {}
        for (it, w, m, a1, a2) in zip(qry['item'], qry['_worker_id'],
                                      qry['m'], qry['a1'], qry['a2']):
            lo[it] = lo.get(it, 0.0) + cw(w, a1, a2) * (2 * m - 1)
        preds_kappa[f'ContextLogOdds_k{kappa:.0f}'] = {
            it: float(sigmoid(v)) for it, v in lo.items()}
    preds.update(preds_kappa)
    preds['DawidSkene'] = ds_em(list(zip(qry['item'], qry['_worker_id'],
                                         qry['m'])))
    # oracle: per-source-cell weights from ALL data
    allcells = {}
    for (w, a1, a2), grp in df.groupby(['_worker_id', 'a1', 'a2']):
        allcells[(w, a1, a2)] = (float((grp['m'] == grp['z']).sum()), len(grp))
    lo = {}
    for (it, w, m, a1, a2) in zip(qry['item'], qry['_worker_id'],
                                  qry['m'], qry['a1'], qry['a2']):
        h, n = allcells.get((w, a1, a2), (0, 0))
        ph = min(max((h + 1) / (n + 2), 0.02), 0.98)
        wgt = np.log(ph / (1 - ph))
        lo[it] = lo.get(it, 0.0) + wgt * (2 * m - 1)
    preds['OracleCtx'] = {it: float(sigmoid(v)) for it, v in lo.items()}

    out = {}
    for meth, pr in preds.items():
        its = [it for it in zq.index if it in pr]
        p = np.array([pr[it] for it in its])
        z = zq.loc[its].values.astype(float)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        out[meth] = {
            'logloss': float(-np.mean(z * np.log(p) + (1 - z) * np.log(1 - p))),
            'acc': float(np.mean((p > 0.5).astype(int) == z)),
            'n_items': len(its)}
    return out


def h1_diagnostic(df):
    """Per-source accuracy gaps across the two context variables."""
    rows = []
    for w, grp in df.groupby('_worker_id'):
        for var in ['a1', 'a2']:
            g0 = grp[grp[var] == 0]
            g1 = grp[grp[var] == 1]
            if len(g0) >= 15 and len(g1) >= 15:
                acc0 = (g0['m'] == g0['z']).mean()
                acc1 = (g1['m'] == g1['z']).mean()
                rows.append({'worker': w, 'var': var,
                             'gap': abs(acc0 - acc1),
                             'sign_flip': (acc0 - 0.5) * (acc1 - 0.5) < 0})
    d = pd.DataFrame(rows)
    summ = {}
    for var in ['a1', 'a2']:
        s = d[d['var'] == var]
        summ[var] = {'n_sources': int(len(s)),
                     'median_gap': float(s['gap'].median()),
                     'frac_gap_gt_0.15': float((s['gap'] > 0.15).mean()),
                     'frac_gap_gt_0.25': float((s['gap'] > 0.25).mean()),
                     'frac_sign_flip': float(s['sign_flip'].mean())}
    return summ


def main():
    df, top_channel = load()
    print(f'judgments={len(df)} items={df["item"].nunique()} '
          f'workers={df["_worker_id"].nunique()} top_channel={top_channel} '
          f'base_rate_z={df.drop_duplicates("item")["z"].mean():.3f}')
    h1 = h1_diagnostic(df)
    print('H1 diagnostic:', json.dumps(h1, indent=2))
    rng = np.random.default_rng(20261021)
    splits = [run_split(df, rng) for _ in range(20)]
    methods = list(splits[0].keys())
    agg = {}
    for meth in methods:
        agg[meth] = {
            'logloss': float(np.mean([s[meth]['logloss'] for s in splits])),
            'logloss_se': float(np.std([s[meth]['logloss'] for s in splits]) / np.sqrt(len(splits))),
            'acc': float(np.mean([s[meth]['acc'] for s in splits])),
            'acc_se': float(np.std([s[meth]['acc'] for s in splits]) / np.sqrt(len(splits)))}
    paired = {}
    for meth in methods:
        if meth == 'ScalarLogOdds':
            continue
        dl = [s['ScalarLogOdds']['logloss'] - s[meth]['logloss'] for s in splits]
        da = [s['ScalarLogOdds']['acc'] - s[meth]['acc'] for s in splits]
        paired[meth] = {
            'logloss_adv_over_scalar': float(np.mean(dl)),
            'logloss_adv_ci': [float(np.mean(dl) - 1.96 * np.std(dl) / np.sqrt(len(dl))),
                               float(np.mean(dl) + 1.96 * np.std(dl) / np.sqrt(len(dl)))],
            'acc_adv_over_scalar': float(np.mean(da)),
            'acc_adv_ci': [float(np.mean(da) - 1.96 * np.std(da) / np.sqrt(len(da))),
                           float(np.mean(da) + 1.96 * np.std(da) / np.sqrt(len(da)))]}
    result = {'n_judgments': int(len(df)), 'n_items': int(df['item'].nunique()),
              'n_workers': int(df['_worker_id'].nunique()),
              'top_channel': str(top_channel), 'h1_diagnostic': h1,
              'methods': agg, 'paired_vs_scalar': paired}
    with open(os.path.join(OUT, 'studyB_summary.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
