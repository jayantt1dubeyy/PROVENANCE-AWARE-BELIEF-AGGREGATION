"""Figures for Experiment I (stability boundary).

Reads results/expI_boundary.csv (rho x K grid, Bayes vs PopMap).
Produces:
  figI_boundary_lines.pdf/.png : accuracy vs rho for K=10 (paper style).
  figI_boundary_heatmap.pdf/.png : (Bayes - PopMap) over the rho x K grid.

Run after results/expI_boundary.csv is complete.
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

COLORS = {'Bayes': '#7f7f0f', 'PopMap': '#888888'}


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def fig_lines():
    df = pd.read_csv(f'{RESULTS}/expI_boundary.csv')
    d10 = df[df.K == 10]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ['Bayes', 'PopMap']:
        sub = d10[d10.method == method].groupby('rho')['acc']
        mean, h = sub.mean(), sub.apply(ci95)
        ax.errorbar(mean.index, mean.values, yerr=h.values, marker='o',
                    capsize=4, label=method,
                    color=COLORS[method], linewidth=2, markersize=6)
    ax.axhline(0.5, ls='--', color='k', alpha=0.3, label='chance')
    ax.set_xlabel(r'Convention stability $\rho$', fontsize=12)
    ax.set_ylabel('Accuracy (mean $\pm$ 95% CI, 30 seeds)', fontsize=12)
    ax.set_title(r'Per-source decoding vs. population mapping ($K=10$)',
                 fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0.4, 0.9)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/figI_boundary_lines.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/figI_boundary_lines.pdf', bbox_inches='tight')
    plt.close()
    print('  figI_boundary_lines saved')


def fig_heatmap():
    df = pd.read_csv(f'{RESULTS}/expI_boundary.csv')
    piv = df.groupby(['K', 'rho', 'method'])['acc'].mean().unstack()
    delta = (piv['Bayes'] - piv['PopMap']).unstack('rho')  # rows K, cols rho
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vmax = float(np.abs(delta.values).max())
    im = ax.imshow(delta.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, origin='lower')
    ax.set_xticks(range(len(delta.columns)))
    ax.set_xticklabels([f'{c:.1f}' for c in delta.columns])
    ax.set_yticks(range(len(delta.index)))
    ax.set_yticklabels([str(c) for c in delta.index])
    ax.set_xlabel(r'Convention stability $\rho$', fontsize=12)
    ax.set_ylabel('Support size $K$', fontsize=12)
    ax.set_title('Bayes minus PopMap accuracy (30 seeds)', fontsize=13)
    for i in range(len(delta.index)):
        for j in range(len(delta.columns)):
            v = delta.values[i, j]
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(v) > vmax * 0.55 else 'black')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Bayes $-$ PopMap', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/figI_boundary_heatmap.png', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/figI_boundary_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print('  figI_boundary_heatmap saved')


if __name__ == '__main__':
    fig_lines()
    fig_heatmap()
