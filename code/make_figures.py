"""
Generate all figures from saved experiment CSVs.
Run after pvi_experiments.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

RESULTS = 'results'
FIGURES = 'figures'

COLORS = {
    'ContentOnly': '#888888',
    'NoSupport': '#888888',
    'UnsignedScalar': '#d62728',
    'SignedScalar': '#ff7f0e',
    'MetaMean': '#1f77b4',
    'MetaAttn': '#2ca02c',
    'ContextConditioned': '#17becf',
    'AttentionCalibrator': '#2ca02c',
    'Bayes': '#9467bd',
    'BayesUniv': '#9467bd',
    'BayesFam': '#8c564b',
    'NaiveMean': '#d62728',
    'MajorityVote': '#ff7f0e',
    'Oracle': '#8c564b',
}


def fig1_rho_sweep():
    """Figure 1: Convention stability sweep."""
    df = pd.read_csv(f'{RESULTS}/expA_rho_sweep.csv')

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['Bayes', 'MetaAttn', 'MetaMean', 'NoSupport',
                   'SignedScalar', 'UnsignedScalar']:
        sub = df[df.method == method].groupby('rho')['acc'].agg(['mean', 'std'])
        ax.errorbar(sub.index, sub['mean'], yerr=sub['std'],
                    marker='o', capsize=4, label=method,
                    color=COLORS.get(method, 'gray'),
                    linewidth=2, markersize=6)

    ax.axhline(0.5, ls='--', color='k', alpha=0.3, label='chance')
    ax.set_xlabel(r'Convention stability $\rho$', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(r'Accuracy vs. convention stability ($K=10$)', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0.4, 0.85)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig1_rho_sweep.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig1_rho_sweep.pdf', bbox_inches='tight')
    plt.close()
    print('  fig1_rho_sweep saved')


def fig2_representation_ablation():
    """Figure 2: Representation ablation hierarchy."""
    df = pd.read_csv(f'{RESULTS}/expB_ablation.csv')

    order = ['ContentOnly', 'UnsignedScalar', 'SignedScalar',
             'ContextConditioned', 'AttentionCalibrator',
             'MetaAttn+Dec', 'MetaAttn+DecCtx', 'Bayes', 'Oracle']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary = df.groupby('method')['acc'].agg(['mean', 'std']).reindex(order)
    colors = [COLORS.get(m, 'gray') for m in order]

    bars = ax.barh(order, summary['mean'], xerr=summary['std'],
                   color=colors, capsize=4)
    ax.axvline(0.5, ls='--', color='k', alpha=0.4, label='chance')
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel(r'Accuracy at $\rho=1.0$, $K=10$', fontsize=12)
    ax.set_title('Representation ablation', fontsize=12)

    # Annotate values
    for i, (m, v) in enumerate(zip(order, summary['mean'])):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig2_ablation.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig2_ablation.pdf', bbox_inches='tight')
    plt.close()
    print('  fig2_ablation saved')


def fig3_ksweep():
    """Figure 3: Support-size scaling on held-out conventions."""
    df = pd.read_csv(f'{RESULTS}/expC_ksweep.csv')

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['BayesFam', 'BayesUniv', 'MetaAttn', 'MetaMean', 'NoSupport']:
        sub = df[df.method == method].groupby('K')['acc'].agg(['mean', 'std'])
        ax.errorbar(sub.index, sub['mean'], yerr=sub['std'],
                    marker='o', capsize=4, label=method,
                    color=COLORS.get(method, 'gray'),
                    linewidth=2, markersize=6)

    ax.axhline(0.5, ls='--', color='k', alpha=0.3, label='chance')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '50'])
    ax.set_xlabel(r'Support size $K$', fontsize=12)
    ax.set_ylabel('Accuracy on held-out conventions', fontsize=12)
    ax.set_title('Convention inference vs. support size', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.4, 0.9)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig3_ksweep.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig3_ksweep.pdf', bbox_inches='tight')
    plt.close()
    print('  fig3_ksweep saved')


def fig4_reliability():
    """Figure 4: Reliability heterogeneity."""
    df = pd.read_csv(f'{RESULTS}/expD_heterogeneity.csv')

    rel_order = ['homogeneous', 'moderate', 'heterogeneous']
    methods = ['Bayes', 'MetaAttn', 'NoSupport']
    x = np.arange(len(rel_order))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(methods):
        means = [df[(df.rel == r) & (df.method == m)]['acc'].mean()
                 for r in rel_order]
        stds = [df[(df.rel == r) & (df.method == m)]['acc'].std()
                for r in rel_order]
        ax.bar(x + i * 0.25 - 0.25, means, 0.25, yerr=stds,
               label=m, color=COLORS.get(m, 'gray'), capsize=4)

    ax.axhline(0.5, ls='--', color='k', alpha=0.3, label='chance')
    ax.set_xticks(x)
    ax.set_xticklabels(rel_order)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(r'Reliability heterogeneity ($\rho \approx 1$)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 0.9)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig4_reliability.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig4_reliability.pdf', bbox_inches='tight')
    plt.close()
    print('  fig4_reliability saved')


def fig5_consensus():
    """Figure 5: Multi-agent consensus."""
    df = pd.read_csv(f'{RESULTS}/expE_consensus.csv')

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['Oracle', 'Bayes', 'MetaAttn', 'MajorityVote', 'NaiveMean']:
        sub = df[df.method == method].groupby('n_agents')['acc'].agg(['mean', 'std'])
        ax.errorbar(sub.index, sub['mean'], yerr=sub['std'],
                    marker='o', capsize=4, label=method,
                    color=COLORS.get(method, 'gray'),
                    linewidth=2, markersize=6)

    ax.axhline(0.5, ls='--', color='k', alpha=0.3, label='chance')
    ax.set_xscale('log')
    ax.set_xticks([2, 3, 4, 6, 8, 12])
    ax.set_xticklabels(['2', '3', '4', '6', '8', '12'])
    ax.set_xlabel('Number of agents', fontsize=12)
    ax.set_ylabel('Group accuracy', fontsize=12)
    ax.set_title('Multi-agent consensus under mixed conventions', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig5_consensus.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig5_consensus.pdf', bbox_inches='tight')
    plt.close()
    print('  fig5_consensus saved')


def fig6_misspec():
    """Figure 6: Candidate-set misspecification."""
    df = pd.read_csv(f'{RESULTS}/expG_misspec.csv')

    order = ['BayesFull', 'BayesHalf', 'MetaAttn', 'Oracle']
    labels = ['BayesFull\n(well-specified)', 'BayesHalf\n(candidates 0-3)',
              'MetaAttn', 'Oracle']
    colors = [COLORS.get(m, 'gray') for m in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    summary = df.groupby('method')['acc'].agg(['mean', 'std']).reindex(order)
    bars = ax.bar(labels, summary['mean'], yerr=summary['std'],
                  color=colors, capsize=4)
    ax.axhline(0.5, ls='--', color='k', alpha=0.4, label='chance')
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Candidate-set misspecification ($K=10$)', fontsize=13)
    for i, v in enumerate(summary['mean']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig6_misspec.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig6_misspec.pdf', bbox_inches='tight')
    plt.close()
    print('  fig6_misspec saved')


def fig6_posterior_diagnostic():
    """Figure 6: Posterior concentration across K (optional)."""
    # This figure is illustrative; adjust paths if you store the data.
    # Placeholder for future use.
    pass


if __name__ == '__main__':
    os.makedirs(FIGURES, exist_ok=True)
    print('Generating figures...')
    fig1_rho_sweep()
    fig2_representation_ablation()
    fig3_ksweep()
    fig4_reliability()
    fig5_consensus()
    fig6_misspec()
    print(f'\nAll figures saved in {FIGURES}/')
