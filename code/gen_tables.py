"""Generate LaTeX table rows (mean +/- std over seeds) from results CSVs.

Usage: python3 code/gen_tables.py  (run from the repository root)
"""
import pandas as pd
import numpy as np

RESULTS = 'results'

pd.set_option('display.width', 200)


def cell(mean, std):
    return f'{mean:.3f} $\\pm$ {std:.3f}'


def pivot(csv, index, columns='method', values='acc'):
    df = pd.read_csv(f'{RESULTS}/{csv}')
    g = df.groupby([index, columns])[values]
    mean = g.mean().unstack(columns)
    std = g.std().unstack(columns)
    return mean, std


def latex_table(mean, std, col_order, index_name=''):
    lines = []
    for idx in mean.index:
        row = [str(idx)]
        for c in col_order:
            m = mean.loc[idx, c] if c in mean.columns else float('nan')
            s = std.loc[idx, c] if c in std.columns else float('nan')
            row.append(cell(m, s) if np.isfinite(m) else '--')
        lines.append(' & '.join(row) + r' \\')
    return '\n'.join(lines)


if __name__ == '__main__':
    print('% --- expA rho sweep (rows: rho) ---')
    mean, std = pivot('expA_rho_sweep.csv', 'rho')
    cols = ['Bayes', 'MetaAttn', 'MetaMean', 'NoSupport',
            'UnsignedScalar', 'SignedScalar']
    print(latex_table(mean, std, cols))

    print('\n% --- expB ablation (single row per method) ---')
    df = pd.read_csv(f'{RESULTS}/expB_ablation.csv')
    g = df.groupby('method')['acc'].agg(['mean', 'std'])
    order = ['ContentOnly', 'UnsignedScalar', 'SignedScalar',
             'ContextConditioned', 'AttentionCalibrator',
             'MetaAttn+Dec', 'MetaAttn+DecCtx', 'Bayes', 'Oracle']
    for m in order:
        if m in g.index:
            print(f'{m} & {cell(g.loc[m, "mean"], g.loc[m, "std"])} \\\\')

    print('\n% --- expC ksweep (rows: K) ---')
    mean, std = pivot('expC_ksweep.csv', 'K')
    cols = ['BayesUniv', 'BayesFam', 'MetaAttn', 'MetaMean', 'NoSupport']
    print(latex_table(mean, std, cols))

    print('\n% --- expD heterogeneity (rows: rel) ---')
    df = pd.read_csv(f'{RESULTS}/expD_heterogeneity.csv')
    g = df.groupby(['rel', 'method'])['acc'].agg(['mean', 'std'])
    for rel in ['homogeneous', 'moderate', 'heterogeneous']:
        row = [rel]
        for m in ['Bayes', 'MetaAttn', 'NoSupport']:
            row.append(cell(g.loc[(rel, m), 'mean'], g.loc[(rel, m), 'std']))
        print(' & '.join(row) + r' \\')

    print('\n% --- expE consensus (rows: n_agents) ---')
    mean, std = pivot('expE_consensus.csv', 'n_agents')
    cols = ['Bayes', 'MetaAttn', 'MajorityVote', 'NaiveMean', 'Oracle']
    print(latex_table(mean, std, cols))

    print('\n% --- expF posterior diagnostic ---')
    df = pd.read_csv(f'{RESULTS}/expF_posterior.csv')
    print(df.to_string(index=False))

    print('\n% --- expG misspecification ---')
    df = pd.read_csv(f'{RESULTS}/expG_misspec.csv')
    g = df.groupby('method')['acc'].agg(['mean', 'std'])
    order = ['BayesFull', 'BayesHalf', 'BayesHalf_in', 'BayesHalf_out',
             'MetaAttn', 'Oracle']
    for m in order:
        if m in g.index:
            print(f'{m} & {cell(g.loc[m, "mean"], g.loc[m, "std"])} \\\\')
