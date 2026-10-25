"""Generate LaTeX table bodies (mean +/- 95% CI over 100 seeds) from the
n100 experiment CSVs, plus calibration tables.

Usage: python3 code/gen_tables_n100.py  (run from the repository root)
Requires all four results/exp*_n100.csv files to be complete.
"""
import pandas as pd
import numpy as np
from scipy import stats

RESULTS = 'results'
METRICS = ['acc', 'nll', 'brier', 'ece']


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h


def cell(x):
    m, h = ci95(x)
    return f'{m:.3f} $\\pm$ {h:.3f}'


def pivot_ci(csv, index, columns='method', values='acc'):
    df = pd.read_csv(f'{RESULTS}/{csv}')
    return df.groupby([index, columns])[values]


def latex_table(g, col_order, index_name=''):
    lines = []
    # g.groups.keys() holds one (index, method) tuple per group; iterate
    # over the distinct index values only. (Audit repair, Oct 2026: the
    # previous loop iterated over every key, emitting one duplicated row
    # per method.)
    for idx in [(v,) for v in sorted({k[0] for k in g.groups.keys()})]:
        row = [str(idx[0])]
        for c in col_order:
            try:
                row.append(cell(g.get_group((idx[0], c))))
            except KeyError:
                row.append('--')
        lines.append(' & '.join(row) + r' \\')
    return '\n'.join(lines)


def calib_table(csv, group_cols, methods):
    df = pd.read_csv(f'{RESULTS}/{csv}')
    lines = []
    for m in methods:
        sub = df[df.method == m]
        if len(sub) == 0:
            continue
        row = [m]
        for met in ['nll', 'brier', 'ece']:
            row.append(cell(sub[met].values))
        lines.append(' & '.join(row) + r' \\')
    return '\n'.join(lines)


if __name__ == '__main__':
    out = []

    out.append('% --- n100 expA rho sweep: accuracy, rows = rho ---')
    g = pivot_ci('expA_n100.csv', 'rho')
    cols = ['NoSupport', 'UnsignedScalar', 'SignedScalar', 'MetaMean',
            'MetaAttn', 'BetaBinomial', 'Bayes']
    out.append(latex_table(g, cols))

    out.append('\n% --- n100 expB ablation: accuracy, one row per method ---')
    df = pd.read_csv(f'{RESULTS}/expB_n100.csv')
    order = ['ContentOnly', 'UnsignedScalar', 'SignedScalar',
             'ContextConditioned', 'AttentionCalibrator', 'BetaBinomial',
             'MetaAttn+Dec', 'MetaAttn+DecCtx', 'DeepAttn',
             'TransformerMeta', 'Bayes', 'Oracle']
    for m in order:
        sub = df[df.method == m]['acc']
        if len(sub):
            out.append(f'{m} & {cell(sub)} \\\\')

    out.append('\n% --- n100 expC K sweep: accuracy, rows = K ---')
    g = pivot_ci('expC_n100.csv', 'K')
    cols = ['BayesUniv', 'BayesFam', 'NoSupport', 'MetaMean', 'MetaAttn']
    out.append(latex_table(g, cols))

    out.append('\n% --- n100 expE consensus: accuracy, rows = n_agents ---')
    g = pivot_ci('expE_n100.csv', 'n_agents')
    cols = ['NaiveMean', 'MajorityVote', 'MetaAttn', 'Bayes', 'Oracle']
    out.append(latex_table(g, cols))

    out.append('\n% --- n100 calibration (mean +/- 95% CI): NLL / Brier / ECE ---')
    for csv, methods in [
        ('expA_n100.csv', ['NoSupport', 'SignedScalar', 'MetaAttn',
                           'BetaBinomial', 'Bayes']),
        ('expB_n100.csv', ['SignedScalar', 'ContextConditioned', 'MetaAttn+Dec',
                           'DeepAttn', 'TransformerMeta', 'BetaBinomial',
                           'Bayes', 'Oracle']),
        ('expE_n100.csv', ['NaiveMean', 'MetaAttn', 'Bayes', 'Oracle']),
    ]:
        out.append(f'% {csv}')
        out.append(calib_table(csv, None, methods))

    text = '\n'.join(out)
    print(text)
    with open(f'{RESULTS}/tables_n100.tex', 'w') as f:
        f.write(text + '\n')
    print(f'\n% wrote {RESULTS}/tables_n100.tex')
