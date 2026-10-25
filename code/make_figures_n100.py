"""Generate n100 figures from the 100-seed experiment CSVs.

Error bars are 95% t-CIs over seeds (not sample stds).
Run after all four results/exp*_n100.csv files are complete.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RESULTS = 'results'
FIGURES = 'figures'

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

COLORS = {
    'NoSupport': '#888888', 'ContentOnly': '#888888',
    'UnsignedScalar': '#d62728', 'SignedScalar': '#ff7f0e',
    'MetaMean': '#1f77b4', 'MetaAttn': '#2ca02c',
    'MetaAttn+Dec': '#2ca02c', 'MetaAttn+DecCtx': '#2ca02c',
    'AttentionCalibrator': '#2ca02c',
    'ContextConditioned': '#17becf',
    'DeepAttn': '#9467bd', 'TransformerMeta': '#8c564b',
    'BetaBinomial': '#e377c2',
    'Bayes': '#7f7f0f', 'BayesUniv': '#7f7f0f', 'BayesFam': '#8c564b',
    'NaiveMean': '#d62728', 'MajorityVote': '#ff7f0e',
    'Oracle': '#000000',
}


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def mean_ci(df, group, method):
    sub = df[df.method == method].groupby(group)['acc']
    return sub.mean(), sub.apply(ci95)


def fig1_rho_sweep():
    df = pd.read_csv(f'{RESULTS}/expA_n100.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['Bayes', 'BetaBinomial', 'MetaAttn', 'MetaMean',
                   'NoSupport', 'SignedScalar']:
        mean, h = mean_ci(df, 'rho', method)
        ax.errorbar(mean.index, mean.values, yerr=h.values, marker='o',
                    capsize=4, label=method, color=COLORS.get(method, 'gray'),
                    linewidth=2, markersize=6)
    ax.axhline(0.5, ls='--', color='k', alpha=0.3, label='chance')
    ax.set_xlabel(r'Convention stability $\rho$', fontsize=12)
    ax.set_ylabel('Accuracy (mean $\\pm$ 95% CI, 100 seeds)', fontsize=12)
    ax.set_title(r'Accuracy vs. convention stability ($K=10$)', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0.4, 0.9)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig1_rho_sweep_n100.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig1_rho_sweep_n100.pdf', bbox_inches='tight')
    plt.close()
    print('  fig1_rho_sweep_n100 saved')


def fig2_ablation():
    df = pd.read_csv(f'{RESULTS}/expB_n100.csv')
    order = ['ContentOnly', 'UnsignedScalar', 'SignedScalar',
             'ContextConditioned', 'AttentionCalibrator', 'BetaBinomial',
             'MetaAttn+Dec', 'MetaAttn+DecCtx', 'DeepAttn',
             'TransformerMeta', 'Bayes', 'Oracle']
    g = df.groupby('method')['acc']
    mean = g.mean().reindex(order)
    h = g.apply(ci95).reindex(order)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [COLORS.get(m, 'gray') for m in order]
    ax.barh(order, mean.values, xerr=h.values, color=colors, capsize=4)
    ax.axvline(0.5, ls='--', color='k', alpha=0.4)
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel('Accuracy (mean $\\pm$ 95% CI, 100 seeds)', fontsize=12)
    ax.set_title('Representation ablation', fontsize=12)
    for i, (m, v) in enumerate(zip(order, mean.values)):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig2_ablation_n100.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig2_ablation_n100.pdf', bbox_inches='tight')
    plt.close()
    print('  fig2_ablation_n100 saved')


def fig3_ksweep():
    df = pd.read_csv(f'{RESULTS}/expC_n100.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['BayesFam', 'BayesUniv', 'MetaAttn', 'MetaMean',
                   'NoSupport']:
        sub = df[df.method == method]
        if len(sub) == 0:
            continue
        mean, h = mean_ci(df, 'K', method)
        ax.errorbar(mean.index, mean.values, yerr=h.values, marker='o',
                    capsize=4, label=method, color=COLORS.get(method, 'gray'),
                    linewidth=2, markersize=6)
    ax.axhline(0.5, ls='--', color='k', alpha=0.3, label='chance')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '50'])
    ax.set_xlabel(r'Support size $K$', fontsize=12)
    ax.set_ylabel('Accuracy (mean $\\pm$ 95% CI, 100 seeds)', fontsize=12)
    ax.set_title('Convention inference vs. support size', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.4, 0.95)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig3_ksweep_n100.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig3_ksweep_n100.pdf', bbox_inches='tight')
    plt.close()
    print('  fig3_ksweep_n100 saved')


def fig5_consensus():
    df = pd.read_csv(f'{RESULTS}/expE_n100.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['Bayes', 'MetaAttn', 'MajorityVote', 'NaiveMean',
                   'Oracle']:
        mean, h = mean_ci(df, 'n_agents', method)
        ax.errorbar(mean.index, mean.values, yerr=h.values, marker='o',
                    capsize=4, label=method, color=COLORS.get(method, 'gray'),
                    linewidth=2, markersize=6)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks([2, 3, 4, 6, 8, 12])
    ax.set_xticklabels(['2', '3', '4', '6', '8', '12'])
    ax.set_xlabel(r'Number of agents $N$', fontsize=12)
    ax.set_ylabel('Accuracy (mean $\\pm$ 95% CI, 100 seeds)', fontsize=12)
    ax.set_title('Multi-agent consensus', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig5_consensus_n100.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig5_consensus_n100.pdf', bbox_inches='tight')
    plt.close()
    print('  fig5_consensus_n100 saved')


def fig_calibration():
    """ECE by method for expA and expB."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (csv, methods, title) in zip(axes, [
        ('expA_n100.csv',
         ['NoSupport', 'SignedScalar', 'MetaAttn', 'BetaBinomial', 'Bayes'],
         'Exp A: convention stability sweep'),
        ('expB_n100.csv',
         ['SignedScalar', 'ContextConditioned', 'MetaAttn+Dec', 'DeepAttn',
          'TransformerMeta', 'BetaBinomial', 'Bayes', 'Oracle'],
         'Exp B: representation ablation'),
    ]):
        df = pd.read_csv(f'{RESULTS}/{csv}')
        g = df.groupby('method')['ece']
        mean = g.mean().reindex(methods)
        h = g.apply(ci95).reindex(methods)
        colors = [COLORS.get(m, 'gray') for m in methods]
        ax.barh(methods, mean.values, xerr=h.values, color=colors, capsize=4)
        ax.set_xlabel('ECE (mean $\\pm$ 95% CI)', fontsize=11)
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig_calibration_n100.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig_calibration_n100.pdf', bbox_inches='tight')
    plt.close()
    print('  fig_calibration_n100 saved')


if __name__ == '__main__':
    fig1_rho_sweep()
    fig2_ablation()
    fig3_ksweep()
    fig5_consensus()
    fig_calibration()
    print('done')
