"""PVI-REAL deep analysis.

Part 1 (Study A, from banked episode CSV): paired CIs + effect sizes
for PVI vs each comparator; oracle-gap closure by K; within-condition
heterogeneity slopes; ablation decomposition; PVI failure episodes.

Part 2 (Study B, CrowdTruth): temperature-scaling test for the
calibration gap (temperature chosen on a calibration half of support,
never on query), per-family gains, ECE, worker-influence (drop
top-gap / bottom-gap workers), worker-threshold robustness.

Outputs: results/pvi_real/deep_analysis.json and prints a summary.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studyB_crowdtruth import load, logodds_weight, sigmoid  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'results', 'pvi_real')
RNG = np.random.default_rng(99)


def ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return float(m), [float(m - 1.96 * se), float(m + 1.96 * se)], \
        float(m / (x.std(ddof=1) + 1e-12))  # mean, CI, Cohen dz


# ---------------------------------------------------------------- Part 1
def studyA():
    df = pd.read_csv(os.path.join(OUT, 'studyA_episodes.csv'))
    res = {}
    # paired PVI vs comparators per condition @K20 and @K40 (logloss: + = PVI better)
    paired = {}
    for cond in ['low', 'fixed', 'context', 'boundary', 'emergent']:
        paired[cond] = {}
        for K in [20, 40]:
            paired[cond][K] = {}
            piv = df[(df.condition == cond) & (df.K == K) &
                     (df.method == 'PVI')].sort_values('episode')
            for meth in ['Majority', 'SignedScalar', 'LogOddsScalar',
                         'ContextMatchFreq', 'PerContextTM', 'Oracle']:
                oth = df[(df.condition == cond) & (df.K == K) &
                         (df.method == meth)].sort_values('episode')
                dll = oth['logloss'].values - piv['logloss'].values
                dac = piv['acc'].values - oth['acc'].values
                m1, c1, d1 = ci95(dll)
                m2, c2, d2 = ci95(dac)
                paired[cond][K][meth] = {
                    'logloss_adv': m1, 'logloss_ci': c1, 'dz_ll': d1,
                    'acc_adv': m2, 'acc_ci': c2, 'dz_acc': d2}
    res['paired'] = paired
    # oracle-gap closure by K in context/emergent: (scalar-PVI)/(scalar-oracle)
    closure = {}
    for cond in ['context', 'emergent', 'boundary']:
        closure[cond] = {}
        for K in [0, 2, 5, 10, 20, 40]:
            def mean_ll(meth):
                s = df[(df.condition == cond) & (df.K == K) &
                       (df.method == meth)]
                return s['logloss'].mean()
            sc, pv, orc = mean_ll('LogOddsScalar'), mean_ll('PVI'), \
                mean_ll('Oracle')
            closure[cond][K] = float((sc - pv) / (sc - orc)) if sc > orc else None
    res['gap_closure'] = closure
    # within-condition heterogeneity slopes (episode level, K=20)
    slopes = {}
    for cond in ['context', 'emergent', 'boundary', 'fixed']:
        sub = df[(df.condition == cond) & (df.K == 20)]
        wide = sub.pivot_table(index='episode', columns='method',
                               values='logloss')
        het = sub[sub.method == 'PVI'].sort_values('episode')['het'].values
        adv = (wide['LogOddsScalar'] - wide['PVI']).values
        if np.std(het) > 1e-9:
            boots = []
            for _ in range(2000):
                idx = RNG.integers(0, len(het), len(het))
                if np.std(het[idx]) > 1e-9:
                    boots.append(np.polyfit(het[idx], adv[idx], 1)[0])
            slopes[cond] = {'slope': float(np.polyfit(het, adv, 1)[0]),
                            'ci': [float(np.percentile(boots, 2.5)),
                                   float(np.percentile(boots, 97.5))],
                            'het_range': [float(het.min()), float(het.max())]}
    res['within_slopes'] = slopes
    # ablation decomposition (context, K=20): share of PVI-vs-scalar
    # logloss advantage destroyed by each ablation
    sub = df[(df.condition == 'context') & (df.K == 20)]
    wide = sub.pivot_table(index='episode', columns='method', values='logloss')
    base = (wide['LogOddsScalar'] - wide['PVI']).mean()
    decomp = {}
    for abl in ['PVI_srcshuffle', 'PVI_ctxshuffle', 'PVI_placebo']:
        abl_adv = (wide['LogOddsScalar'] - wide[abl]).mean()
        decomp[abl] = {'adv_remaining': float(abl_adv),
                       'share_destroyed': float(1 - abl_adv / base)}
    res['ablation_decomp_context_K20'] = {'base_adv': float(base), **decomp}
    # failure episodes: where does PVI lose to LogOddsScalar (logloss)?
    fails = {}
    for cond in ['context', 'emergent', 'boundary']:
        sub = df[(df.condition == cond) & (df.K == 20)]
        wide = sub.pivot_table(index='episode', columns='method',
                               values='logloss')
        het = sub[sub.method == 'PVI'].sort_values('episode')['het']
        d = (wide['LogOddsScalar'] - wide['PVI'])
        fails[cond] = {'n_loss': int((d < 0).sum()), 'n': int(len(d)),
                       'het_mean_when_win': float(het[d.values > 0].mean()),
                       'het_mean_when_loss': float(het[d.values < 0].mean()) if (d < 0).any() else None}
    res['failures'] = fails
    return res


# ---------------------------------------------------------------- Part 2
def split_items(df, rng):
    items = df[['item', 'z', 'a1']].drop_duplicates()
    sup, qry = [], []
    for (_, _), grp in items.groupby(['a1', 'z']):
        idx = grp['item'].tolist()
        rng.shuffle(idx)
        sup += idx[:len(idx) // 2]
        qry += idx[len(idx) // 2:]
    return set(sup), set(qry)


def fit_weights(sup):
    g = sup.assign(hit=(sup['m'] == sup['z']).astype(float)) \
           .groupby('_worker_id')['hit'].agg(['sum', 'count'])
    gp = (g['sum'].sum(), g['count'].sum())
    gw = {w: logodds_weight(r['sum'], r['count'], *gp)
          for w, r in g.iterrows()}
    cells = {}
    for (w, a1, a2), grp in sup.groupby(['_worker_id', 'a1', 'a2']):
        cells[(w, a1, a2)] = (float((grp['m'] == grp['z']).sum()), len(grp))

    def cw(w, a1, a2, kappa=8.0):
        if w in g.index:
            ph, pn = g.loc[w, 'sum'], g.loc[w, 'count']
        else:
            ph, pn = gp
        if (w, a1, a2) in cells:
            h, n = cells[(w, a1, a2)]
            return logodds_weight(h, n, ph, pn, kappa=kappa)
        return logodds_weight(ph, pn, *gp)
    return gw, cw


def aggregate_lo(dfq, weight_fn):
    lo = {}
    for it, w, m, a1, a2 in zip(dfq['item'], dfq['_worker_id'], dfq['m'],
                                dfq['a1'], dfq['a2']):
        lo[it] = lo.get(it, 0.0) + weight_fn(w, a1, a2) * (2 * m - 1)
    return lo


def metrics(lo, zmap, temp=1.0):
    its = list(lo.keys())
    p = np.clip(sigmoid(np.array([lo[i] for i in its]) / temp), 1e-9, 1 - 1e-9)
    z = np.array([zmap[i] for i in its], float)
    ll = -np.mean(z * np.log(p) + (1 - z) * np.log(1 - p))
    acc = np.mean((p > 0.5).astype(int) == z)
    # ECE, 10 bins
    ece = 0.0
    for b in range(10):
        sel = (p >= b / 10) & (p < (b + 1) / 10)
        if sel.sum() > 0:
            ece += sel.mean() * abs(p[sel].mean() - z[sel].mean())
    return {'logloss': float(ll), 'acc': float(acc), 'ece': float(ece)}


def studyB(df, n_splits=20, seed=20261021):
    rng = np.random.default_rng(seed)
    temps = np.round(np.arange(0.4, 3.05, 0.1), 2)
    rows = {k: [] for k in ['scalar', 'scalar_T', 'context', 'context_T']}
    fam_rows = {'cause': [], 'treat': []}
    for _ in range(n_splits):
        sup_items, qry_items = split_items(df, rng)
        sup = df[df['item'].isin(sup_items)]
        qry = df[df['item'].isin(qry_items)]
        # calibration half of support (by item)
        sitems = sup['item'].unique()
        rng.shuffle(sitems)
        cal_set = set(sitems[:len(sitems) // 2])
        fit = sup[~sup['item'].isin(cal_set)]
        cal = sup[sup['item'].isin(cal_set)]
        zmap_cal = cal.groupby('item')['z'].first().to_dict()
        zmap_q = qry.groupby('item')['z'].first().to_dict()
        gw_f, cw_f = fit_weights(fit)
        lo_cal_s = aggregate_lo(cal, lambda w, a1, a2: gw_f.get(w, 0.0))
        lo_cal_c = aggregate_lo(cal, cw_f)
        def best_T(lo_cal):
            lls = [metrics(lo_cal, zmap_cal, T)['logloss'] for T in temps]
            return float(temps[int(np.argmin(lls))])
        T_s, T_c = best_T(lo_cal_s), best_T(lo_cal_c)
        gw, cw = fit_weights(sup)
        lo_q_s = aggregate_lo(qry, lambda w, a1, a2: gw.get(w, 0.0))
        lo_q_c = aggregate_lo(qry, cw)
        rows['scalar'].append(metrics(lo_q_s, zmap_q))
        rows['scalar_T'].append(metrics(lo_q_s, zmap_q, T_s))
        rows['context'].append(metrics(lo_q_c, zmap_q))
        rows['context_T'].append(metrics(lo_q_c, zmap_q, T_c))
        # per-family accuracy diffs (context - scalar)
        for fam, a1v in [('cause', 0), ('treat', 1)]:
            its = [i for i in zmap_q if qry[qry['item'] == i]['a1'].iloc[0] == a1v]
            if its:
                ps = sigmoid(np.array([lo_q_s[i] for i in its]))
                pc = sigmoid(np.array([lo_q_c[i] for i in its]))
                z = np.array([zmap_q[i] for i in its])
                fam_rows[fam].append(float(
                    np.mean((pc > 0.5).astype(int) == z) -
                    np.mean((ps > 0.5).astype(int) == z)))
    out = {}
    for k, v in rows.items():
        out[k] = {m: float(np.mean([r[m] for r in v])) for m in v[0]}
        out[k]['logloss_se'] = float(np.std([r['logloss'] for r in v]) / np.sqrt(len(v)))
    out['paired_contextT_vs_scalarT_logloss'] = ci95(
        [a['logloss'] - b['logloss'] for a, b in zip(rows['scalar_T'], rows['context_T'])])[0:2]
    out['paired_context_vs_scalar_logloss'] = ci95(
        [a['logloss'] - b['logloss'] for a, b in zip(rows['scalar'], rows['context'])])[0:2]
    out['paired_contextT_vs_scalar_acc'] = ci95(
        [b['acc'] - a['acc'] for a, b in zip(rows['scalar_T'], rows['context_T'])])[0:2]
    out['family_acc_gain'] = {f: ci95(v)[0:2] for f, v in fam_rows.items()}
    out['temps'] = {'scalar_T_mean': None, 'context_T_mean': None}
    return out


def worker_gaps(df):
    gaps = {}
    for w, grp in df.groupby('_worker_id'):
        g0, g1 = grp[grp['a1'] == 0], grp[grp['a1'] == 1]
        if len(g0) >= 15 and len(g1) >= 15:
            gaps[w] = abs((g0['m'] == g0['z']).mean() -
                          (g1['m'] == g1['z']).mean())
    return gaps


def main():
    res = {'studyA': studyA()}
    df, top = load()
    res['studyB_temperature'] = studyB(df)
    # worker influence: drop top-quartile gap workers; control: drop bottom
    gaps = worker_gaps(df)
    gs = pd.Series(gaps).sort_values(ascending=False)
    top_workers = set(gs.index[:len(gs) // 4])
    bot_workers = set(gs.index[-(len(gs) // 4):])
    from studyB_crowdtruth import run_split
    def adv_without(drop):
        d2 = df[~df['_worker_id'].isin(drop)]
        rng = np.random.default_rng(20261021)
        splits = [run_split(d2, rng) for _ in range(20)]
        dl = [s['ScalarLogOdds']['acc'] - s['ContextLogOdds']['acc'] for s in splits]
        m, c, _ = ci95([-x for x in dl])
        return {'context_acc_adv': m, 'ci': c, 'n_workers_left': int(d2['_worker_id'].nunique())}
    res['studyB_influence'] = {'drop_top_gap_quartile': adv_without(top_workers),
                               'drop_bottom_gap_quartile': adv_without(bot_workers)}
    # threshold robustness: rebuild df with different worker minima
    import studyB_crowdtruth as sb
    rob = {}
    orig_load = sb.load
    for thr in [25, 60]:
        src = open(sb.__file__).read()
        ns = {'__file__': sb.__file__}
        exec(src.replace("counts >= 40", f"counts >= {thr}").replace(
            "if __name__ == '__main__':", "if False:"), ns)
        d_thr, _ = ns['load']()
        rng = np.random.default_rng(20261021)
        splits = [ns['run_split'](d_thr, rng) for _ in range(20)]
        dl = [s['ScalarLogOdds']['acc'] - s['ContextLogOdds']['acc'] for s in splits]
        m, c, _ = ci95([-x for x in dl])
        rob[thr] = {'context_acc_adv': m, 'ci': c,
                    'n_workers': int(d_thr['_worker_id'].nunique()),
                    'n_items': int(d_thr['item'].nunique())}
    res['studyB_threshold_robustness'] = rob
    with open(os.path.join(OUT, 'deep_analysis.json'), 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str)[:6000])


if __name__ == '__main__':
    main()
