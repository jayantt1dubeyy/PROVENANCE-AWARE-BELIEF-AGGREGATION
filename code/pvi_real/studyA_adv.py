"""E3: PVI-REAL adversarial / misspecification suite.

Four panel regimes designed to break the paper's assumptions:
  outofset  sources use convention c=8 (parity 1-a2), which is NOT in
            PVI's candidate set {0..7} -- model misspecification.
  relctx    no conventions; reliability depends on context
            (r = 0.92 if a2=0 else 0.58) -- violates PVI's constant-r.
  drift     context conventions whose stability decays between the
            support stream (rho ~ 0.95) and the query stream
            (rho - 0.35) -- violates stationarity.
  mimic     sources behave as identity on support but flip by a2 on
            queries -- adversarial support/query shift.

Methods: Majority, LogOddsScalar, ContextMatchFreq, PerContextTM,
PVI (candidates 0..7), Oracle (true QUERY-phase parameters).
K in {10, 40}, Q = 40, 40 episodes per condition.
Output: results/pvi_real/e3_summary.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studyA_sim import PARITY, sigmoid, pvi_log_odds, oracle_lo  # noqa: E402
from pvi_experiments import bayes_post  # noqa: E402

PARITY[8] = lambda a1, a2: 1 - a2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'results', 'pvi_real')
CANDS = list(range(8))
Q = 40
N_SRC = 6


def draw_items(rng, n):
    return (rng.integers(0, 2, n), rng.integers(0, 2, n),
            rng.integers(0, 2, n))


def emit(rng, c, rho, r, items, r_by_a2=None):
    Z, a1, a2 = items
    rr = r if r_by_a2 is None else np.where(a2 == 1, r_by_a2, r)
    R = np.where(rng.random(len(Z)) < rr, Z, 1 - Z)
    conv = R ^ PARITY[c](a1, a2)
    use = rng.random(len(Z)) < rho
    return np.where(use, conv, R).astype(np.int64)


def sample_srcs(rng, cond):
    srcs = []
    for _ in range(N_SRC):
        r = rng.uniform(0.7, 0.95)
        if cond == 'outofset':
            srcs.append({'c_sup': 8, 'c_qry': 8,
                         'rho_sup': rng.uniform(0.6, 1.0), 'rho_qry': None,
                         'r': r})
        elif cond == 'relctx':
            srcs.append({'c_sup': 0, 'c_qry': 0, 'rho_sup': 1.0,
                         'rho_qry': 1.0, 'r': 0.92, 'r_by_a2': 0.58})
        elif cond == 'drift':
            c = int(rng.integers(2, 8))
            srcs.append({'c_sup': c, 'c_qry': c, 'rho_sup': 0.95,
                         'rho_qry': 0.60, 'r': r})
        elif cond == 'mimic':
            srcs.append({'c_sup': 0, 'c_qry': 3, 'rho_sup': 0.95,
                         'rho_qry': 0.95, 'r': r})
        s = srcs[-1]
        if s['rho_qry'] is None:
            s['rho_qry'] = s['rho_sup']
    return srcs


def run_episode(rng, cond, K):
    srcs = sample_srcs(rng, cond)
    qry_items = draw_items(rng, Q)
    Zq = qry_items[0]
    sup, Mq = [], []
    for s in srcs:
        sup_items = draw_items(rng, K)
        sm = emit(rng, s['c_sup'], s['rho_sup'], s['r'], sup_items,
                  s.get('r_by_a2'))
        qm = emit(rng, s['c_qry'], s['rho_qry'], s['r'], qry_items,
                  s.get('r_by_a2'))
        sup.append((sm, *sup_items))
        Mq.append(qm)
    Mq = np.stack(Mq)
    qa1, qa2 = qry_items[1], qry_items[2]
    p = {}
    p['Majority'] = np.clip(Mq.mean(axis=0), 0.02, 0.98)
    lo_s, lo_c, lo_t, lo_p, lo_o = (np.zeros(Q) for _ in range(5))
    for j, s in enumerate(srcs):
        sm, sz, s1, s2 = sup[j]
        hits = float((sm == sz).mean())
        cons_s = ((sm == sz).sum() + 1) / (K + 2)
        w_lo = np.log(cons_s / (1 - cons_s))
        lo_s += w_lo * (2 * Mq[j] - 1)
        # context freq + TM from support
        for a1 in (0, 1):
            for a2 in (0, 1):
                sel_q = (qa1 == a1) & (qa2 == a2)
                if not sel_q.any():
                    continue
                sel_s = (s1 == a1) & (s2 == a2)
                if sel_s.sum() >= 3:
                    w = 2 * float((sm[sel_s] == sz[sel_s]).mean()) - 1
                else:
                    w = 2 * hits - 1
                lo_c[sel_q] += w * (2 * Mq[j, sel_q] - 1)
                ss = sel_s if sel_s.sum() >= 4 else np.ones_like(sm, bool)
                z1 = sz[ss] == 1
                p11 = (sm[ss][z1].sum() + 1) / (z1.sum() + 2)
                p10 = (sm[ss][~z1].sum() + 1) / ((~z1).sum() + 2)
                mq = Mq[j, sel_q]
                lo_t[sel_q] += np.where(mq == 1, np.log(p11) - np.log(p10),
                                        np.log(1 - p11) - np.log(1 - p10))
        post = bayes_post(sm, s1, s2, sz, CANDS)
        lo_p += pvi_log_odds(post, Mq[j], qa1, qa2)
        if cond == 'relctx':
            # oracle uses per-context reliability, identity convention
            for a2v, rv in [(0, s['r']), (1, s['r_by_a2'])]:
                sel = qa2 == a2v
                lo_o[sel] += oracle_lo(Mq[j, sel], qa1[sel], qa2[sel],
                                       0, 1.0, rv)
        else:
            lo_o += oracle_lo(Mq[j], qa1, qa2, s['c_qry'], s['rho_qry'],
                              s['r'])
    p['LogOddsScalar'] = sigmoid(lo_s)
    p['ContextMatchFreq'] = sigmoid(lo_c)
    p['PerContextTM'] = sigmoid(lo_t)
    p['PVI'] = sigmoid(lo_p)
    p['Oracle'] = sigmoid(lo_o)
    out = {}
    for meth, prob in p.items():
        prob = np.clip(prob, 1e-9, 1 - 1e-9)
        out[meth] = {
            'logloss': float(-np.mean(Zq * np.log(prob) +
                                      (1 - Zq) * np.log(1 - prob))),
            'acc': float(np.mean((prob > 0.5).astype(int) == Zq))}
    return out


def main():
    rng = np.random.default_rng(20261021)
    res = {}
    for cond in ['outofset', 'relctx', 'drift', 'mimic']:
        res[cond] = {}
        for K in [10, 40]:
            acc = {}
            for _ in range(40):
                r = run_episode(rng, cond, K)
                for m, v in r.items():
                    acc.setdefault(m, []).append(v)
            res[cond][K] = {m: {'logloss': float(np.mean([x['logloss'] for x in v])),
                                'acc': float(np.mean([x['acc'] for x in v]))}
                            for m, v in acc.items()}
            print(cond, K, {m: (round(x['logloss'], 3), round(x['acc'], 3))
                            for m, x in res[cond][K].items()}, flush=True)
    with open(os.path.join(OUT, 'e3_summary.json'), 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
