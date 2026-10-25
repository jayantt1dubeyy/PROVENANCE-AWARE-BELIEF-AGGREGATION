"""E2: second real task -- CrowdTruth RelDir (direction judgments),
a different microtask over the same sentences and expert gold labels
as Study B's RelEx task. Tests whether the Study B findings
(context gain + calibration) replicate across tasks.

Binary reduction (pre-declared):
  item = RelDir unit whose candidate relation is 'causes' or 'treats';
  z    = expert verdict from ground_truth_{cause,treat}.csv joined on
         (numeric SID, family); expert 1 -> 1, -1 -> 0.
  m    = 1 iff the worker's constructed direction statement affirms
         the candidate family in the forward direction (statement
         contains the family verb and starts with term1);
         'no_relation' and other-family statements -> 0.
  a1   = family (cause 0 / treat 1); a2 = channel group (largest = 1).
  sources = workers with >= 25 judgments on expert-covered items.
Pipeline: identical to E1 (cross-fitted calibrated scalar/context,
raw variants, DS, OracleCtx), 20 splits.
Output: results/pvi_real/e2_summary.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studyB_crowdtruth import DATA  # noqa: E402
from e1_calibrated import run  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'results', 'pvi_real')
FAM = {'causes': 'cause', 'treats': 'treat'}


def load_reldir():
    gt = {}
    for fam, fn in [('cause', 'ground_truth_cause.csv'),
                    ('treat', 'ground_truth_treat.csv')]:
        df = pd.read_csv(os.path.join(DATA, fn))
        df = df[df['expert'].isin([-1, 1])]
        for sid, exp in zip(df['SID'], df['expert']):
            key = (int(sid), fam)
            val = 1 if exp == 1 else 0
            if key in gt and gt[key] != val:
                gt[key] = None
            else:
                gt[key] = val
    gt = {k: v for k, v in gt.items() if v is not None}
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, 'raw', 'RelDir', '*.csv'))):
        df = pd.read_csv(f, low_memory=False,
                         usecols=['_unit_id', '_worker_id', '_channel',
                                  'relation', 'direction', 'sent_id',
                                  'term1', 'term2'])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw['relation'].isin(FAM) & raw['direction'].notna()].copy()
    raw['family'] = raw['relation'].map(FAM)
    raw['sid'] = pd.to_numeric(
        raw['sent_id'].astype(str).str.extract(r'^(\d+)')[0],
        errors='coerce')
    raw = raw[raw['sid'].notna()].copy()
    raw['sid'] = raw['sid'].astype(int)
    raw['z'] = [gt.get((s, fam)) for s, fam in zip(raw['sid'], raw['family'])]
    raw = raw[raw['z'].notna()].copy()
    raw['z'] = raw['z'].astype(int)

    def affirm(row):
        d = str(row['direction'])
        if d == 'no_relation':
            return 0
        verb = ' causes ' if row['family'] == 'cause' else ' treats '
        if verb not in d:
            return 0
        return 1 if d.upper().startswith(str(row['term1']).strip().upper()) else 0

    raw['m'] = raw.apply(affirm, axis=1).astype(int)
    top_channel = raw['_channel'].value_counts().idxmax()
    raw['a2'] = (raw['_channel'] == top_channel).astype(int)
    raw['a1'] = (raw['family'] == 'treat').astype(int)
    counts = raw.groupby('_worker_id').size()
    raw = raw[raw['_worker_id'].isin(counts[counts >= 25].index)].copy()
    raw['item'] = raw['sid'].astype(str) + '|' + raw['family']
    return raw, str(top_channel)


def main():
    df, top = load_reldir()
    print(f'RelDir: judgments={len(df)} items={df["item"].nunique()} '
          f'workers={df["_worker_id"].nunique()} top_channel={top} '
          f'mean m={df["m"].mean():.3f} base z='
          f'{df.drop_duplicates("item")["z"].mean():.3f}')
    res = run(df)
    res['dataset'] = {'n_judgments': int(len(df)),
                      'n_items': int(df['item'].nunique()),
                      'n_workers': int(df['_worker_id'].nunique())}
    with open(os.path.join(OUT, 'e2_summary.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
