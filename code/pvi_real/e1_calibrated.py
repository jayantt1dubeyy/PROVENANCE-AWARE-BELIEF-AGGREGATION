"""E1: calibration-integrated aggregators on CrowdTruth RelEx.

The deep analysis showed a post-hoc temperature (fitted on a held-out
half of support) flips Study B's log-loss verdict. E1 builds the
calibration INTO the method, properly: within each split, temperatures
are chosen by 2-fold cross-fitting on the support items (fit weights
on fold A, score fold B, and vice versa; pick T on the pooled
out-of-fold support predictions), then final weights are fit on the
full support and the cross-fitted T is applied to the query. No query
data is touched at any stage of fitting.

Methods: ScalarLogOdds, ScalarCal (integrated), ContextLogOdds,
ContextCal (integrated), DawidSkene, OracleCtx. 20 splits, same
protocol as Study B. Per-family (cause/treat) breakdown for the two
calibrated methods.
Outputs: results/pvi_real/e1_summary.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studyB_crowdtruth import load, logodds_weight, sigmoid, ds_em  # noqa: E402
from deep_analysis import split_items, fit_weights, aggregate_lo, metrics  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'results', 'pvi_real')
TEMPS = np.round(np.arange(0.4, 3.05, 0.1), 2)


def crossfit_T(sup, weight_fn_of_fit):
    """weight_fn_of_fit(fit_df) -> fn(w,a1,a2)->weight. Returns T chosen
    on pooled out-of-fold support predictions."""
    items = np.array(sup['item'].unique(), dtype=object)
    rng = np.random.default_rng(5)
    rng.shuffle(items)
    folds = [set(items[:len(items) // 2]), set(items[len(items) // 2:])]
    lo_oof, z_oof = {}, {}
    for k in range(2):
        fit = sup[sup['item'].isin(folds[1 - k])]
        hold = sup[sup['item'].isin(folds[k])]
        fn = weight_fn_of_fit(fit)
        lo = aggregate_lo(hold, fn)
        zm = hold.groupby('item')['z'].first().to_dict()
        lo_oof.update(lo)
        z_oof.update(zm)
    lls = [metrics(lo_oof, z_oof, T)['logloss'] for T in TEMPS]
    return float(TEMPS[int(np.argmin(lls))])


def run(df, n_splits=20):
    rng = np.random.default_rng(20261021)
    rows = {k: [] for k in ['ScalarLogOdds', 'ScalarCal', 'ContextLogOdds',
                            'ContextCal', 'DawidSkene', 'OracleCtx']}
    fam = {'cause': [], 'treat': []}
    Ts = {'scalar': [], 'context': []}
    for _ in range(n_splits):
        sup_items, qry_items = split_items(df, rng)
        sup = df[df['item'].isin(sup_items)]
        qry = df[df['item'].isin(qry_items)]
        zq = qry.groupby('item')['z'].first().to_dict()

        def scalar_fn(fit):
            gw, _ = fit_weights(fit)
            return lambda w, a1, a2: gw.get(w, 0.0)

        def context_fn(fit):
            _, cw = fit_weights(fit)
            return cw

        T_s = crossfit_T(sup, scalar_fn)
        T_c = crossfit_T(sup, context_fn)
        Ts['scalar'].append(T_s)
        Ts['context'].append(T_c)
        gw, cw = fit_weights(sup)
        lo_s = aggregate_lo(qry, lambda w, a1, a2: gw.get(w, 0.0))
        lo_c = aggregate_lo(qry, cw)
        rows['ScalarLogOdds'].append(metrics(lo_s, zq))
        rows['ScalarCal'].append(metrics(lo_s, zq, T_s))
        rows['ContextLogOdds'].append(metrics(lo_c, zq))
        rows['ContextCal'].append(metrics(lo_c, zq, T_c))
        ds = ds_em(list(zip(qry['item'], qry['_worker_id'], qry['m'])))
        zm = {k: float(v) for k, v in zq.items()}
        p = np.clip(np.array([ds[i] for i in zm]), 1e-9, 1 - 1e-9)
        z = np.array(list(zm.values()))
        rows['DawidSkene'].append({
            'logloss': float(-np.mean(z * np.log(p) + (1 - z) * np.log(1 - p))),
            'acc': float(np.mean((p > 0.5).astype(int) == z)), 'ece': 0.0})
        # oracle context weights from all data
        cells = {}
        for (w, a1, a2), grp in df.groupby(['_worker_id', 'a1', 'a2']):
            cells[(w, a1, a2)] = (float((grp['m'] == grp['z']).sum()), len(grp))
        def ow(w, a1, a2):
            h, n = cells.get((w, a1, a2), (0, 0))
            ph = min(max((h + 1) / (n + 2), 0.02), 0.98)
            return float(np.log(ph / (1 - ph)))
        rows['OracleCtx'].append(metrics(aggregate_lo(qry, ow), zq))
        # per-family: ContextCal vs ScalarCal accuracy
        for famname, a1v in [('cause', 0), ('treat', 1)]:
            its = [i for i in zq if qry[qry['item'] == i]['a1'].iloc[0] == a1v]
            if its:
                ps = sigmoid(np.array([lo_s[i] for i in its]) / T_s)
                pc = sigmoid(np.array([lo_c[i] for i in its]) / T_c)
                zz = np.array([zq[i] for i in its])
                fam[famname].append(float(np.mean((pc > 0.5) == zz) -
                                          np.mean((ps > 0.5) == zz)))
    def agg(v):
        return {m: float(np.mean([r[m] for r in v])) for m in ['logloss', 'acc', 'ece']}
    out = {k: agg(v) for k, v in rows.items()}
    def paired(a, b, key, sign=1):
        d = np.array([sign * (x[key] - y[key]) for x, y in zip(rows[a], rows[b])])
        se = d.std(ddof=1) / np.sqrt(len(d))
        return {'mean': float(d.mean()), 'ci': [float(d.mean() - 1.96 * se),
                                               float(d.mean() + 1.96 * se)]}
    out['paired'] = {
        'ContextCal_vs_ScalarCal_logloss (scalar-ctx)':
            paired('ScalarCal', 'ContextCal', 'logloss'),
        'ContextCal_vs_ScalarCal_acc (ctx-scalar)':
            paired('ScalarCal', 'ContextCal', 'acc', sign=-1),
        'ContextCal_vs_ScalarRaw_logloss (scalar-ctx)':
            paired('ScalarLogOdds', 'ContextCal', 'logloss')}
    out['family_acc_gain_ContextCal'] = {
        f: {'mean': float(np.mean(v)),
            'ci': [float(np.mean(v) - 1.96 * np.std(v, ddof=1) / np.sqrt(len(v))),
                   float(np.mean(v) + 1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))]}
        for f, v in fam.items()}
    out['chosen_T'] = {'scalar_mean': float(np.mean(Ts['scalar'])),
                       'context_mean': float(np.mean(Ts['context']))}
    return out


def main():
    df, top = load()
    res = run(df)
    with open(os.path.join(OUT, 'e1_summary.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
