"""
Analyze the 100-seed upgrade results.

For each experiment/method: mean +/- 95% CI (t-interval) for accuracy,
NLL, Brier, ECE. Paired differences versus Bayes (all methods share
seeds, so pairing is valid): mean paired delta, 95% CI of the delta,
and Cohen's d for paired samples.

Usage: python3 code/analyze_n100.py   (run from the repository root)
"""

import numpy as np
import pandas as pd
from scipy import stats

R = 'results'
METRICS = ['acc', 'nll', 'brier', 'ece']


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h


def fmt(m, h):
    return f'{m:.4f} ± {h:.4f}'


def paired_stats(df, group_cols, ref='Bayes'):
    """Per-group paired delta vs ref with CI and Cohen's d."""
    out = []
    groups = df.groupby(group_cols) if group_cols else [('all', df)]
    for key, g in groups:
        piv = g.pivot_table(index='seed', columns='method',
                            values='acc', aggfunc='mean')
        if ref not in piv.columns:
            continue
        for meth in piv.columns:
            if meth == ref:
                continue
            d = (piv[meth] - piv[ref]).dropna().values
            if len(d) < 2:
                continue
            m, h = ci95(d)
            dz = d.mean() / (d.std(ddof=1) + 1e-12)
            out.append({'group': key, 'method': meth,
                        'paired_delta': m, 'delta_ci': h,
                        'cohens_dz': dz, 'n': len(d)})
    return pd.DataFrame(out)


def validate(df, path, expect_seeds=100):
    n = df.seed.nunique()
    nn = int(df.isnull().sum().sum())
    flag = ''
    if n != expect_seeds:
        flag += f'  WARNING: expected {expect_seeds} seeds, got {n}.'
    if nn:
        flag += f'  WARNING: {nn} NaN cells.'
    return f'{path}: {n} unique seeds, {nn} NaNs.{flag}'


def summarize(path, group_cols, title, lines, expect_seeds=100, _df=None):
    df = _df if _df is not None else pd.read_csv(f'{R}/{path}')
    lines.append('')
    lines.append('=' * 70)
    lines.append(title)
    lines.append('=' * 70)
    lines.append(validate(df, path, expect_seeds=expect_seeds))
    for met in METRICS:
        if met not in df.columns:
            continue
        lines.append(f'\n-- {met} (mean ± 95% CI) --')
        s = (df.groupby(group_cols + ['method'])[met]
               .apply(lambda x: fmt(*ci95(x))))
        tab = s.unstack('method') if group_cols else s.to_frame('value').T
        lines.append(tab.to_string())
    lines.append('\n-- paired accuracy delta vs Bayes '
                 '(mean Δ ± 95% CI, Cohen dz) --')
    ps = paired_stats(df, group_cols)
    if len(ps):
        ps['line'] = ps.apply(
            lambda r: f"{r['paired_delta']:+.4f} ± {r['delta_ci']:.4f} "
                      f"(dz={r['cohens_dz']:+.2f})", axis=1)
        lines.append(ps.pivot_table(index='group', columns='method',
                                    values='line',
                                    aggfunc='first').to_string())


if __name__ == '__main__':
    lines = []
    for path, gc, title, expect in [
            ('expA_n100.csv', ['rho'], 'EXPERIMENT A (n=100): rho sweep',
             100),
            ('expB_n100.csv', [], 'EXPERIMENT B (n=30): ablation', 30),
            ('expC_n100.csv', ['K'], 'EXPERIMENT C (n=30): K sweep', 30),
            ('expE_n100.csv', ['n_agents'],
             'EXPERIMENT E (n=30): consensus', 30)]:
        try:
            summarize(path, gc, title, lines, expect_seeds=expect)
        except FileNotFoundError:
            lines.append(f'\n{path}: NOT READY (run still in progress)')
    text = '\n'.join(lines)
    print(text)
    with open(f'{R}/analysis_n100.txt', 'w') as f:
        f.write(text + '\n')
    print(f'\nSaved -> {R}/analysis_n100.txt')
