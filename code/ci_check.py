"""Compute mean +/- 95% CI (t-interval, df=n-1) over seeds for every
paper-reported headline number, and flag claims whose CIs undermine
the stated conclusion.

Usage: python3 code/ci_check.py  (run from the repository root)
"""
import pandas as pd
import numpy as np
from scipy import stats

R = 'results'
T = stats.t.ppf(0.975, 9)  # n=10 seeds

def ci(s):
    s = np.asarray(s, dtype=float)
    m = s.mean()
    h = T * s.std(ddof=1) / np.sqrt(len(s))
    return m, h

def fmt(m, h):
    return f'{m:.3f} ± {h:.3f}'

print('t(0.975, df=9) =', round(T, 3))
print()

# ---------- ExpA: Bayes sweep (paper table rows rho x Bayes col) ----------
df = pd.read_csv(f'{R}/expA_rho_sweep.csv')
print('== ExpA Bayes rho sweep ==')
for rho in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    m, h = ci(df[(df.rho == rho) & (df.method == 'Bayes')]['acc'])
    print(f'rho={rho}: {fmt(m,h)}')
# monotonicity check: pairwise CI overlap between adjacent rho
accs = {rho: ci(df[(df.rho == rho) & (df.method == 'Bayes')]['acc']) for rho in [0.5,0.6,0.7,0.8,0.9,1.0]}
print()
print('== ExpA: Bayes vs SignedScalar at rho=1.0 (paper: 0.755 vs 0.621) ==')
for meth in ['Bayes', 'SignedScalar', 'MetaAttn', 'MetaMean']:
    m, h = ci(df[(df.rho == 1.0) & (df.method == meth)]['acc'])
    print(f'{meth}: {fmt(m,h)}')
mB, hB = accs[1.0]; mS, hS = ci(df[(df.rho==1.0)&(df.method=='SignedScalar')]['acc'])
print('Bayes CI above SignedScalar CI:', (mB-hB) > (mS+hS))

# ---------- ExpB ----------
print()
print('== ExpB ablation ==')
df = pd.read_csv(f'{R}/expB_ablation.csv')
for meth in ['ContentOnly','UnsignedScalar','SignedScalar','ContextConditioned',
             'AttentionCalibrator','MetaAttn+Dec','MetaAttn+DecCtx','Bayes','Oracle']:
    m, h = ci(df[df.method == meth]['acc'])
    print(f'{meth}: {fmt(m,h)}')
def get(meth):
    return ci(df[df.method == meth]['acc'])
mD,hD = get('MetaAttn+Dec'); mO,hO = get('Oracle'); mB2,hB2 = get('Bayes')
mC,hC = get('ContextConditioned'); mA,hA = get('AttentionCalibrator')
print('MetaAttn+Dec vs Oracle overlap (paper: "within noise"):', (mD-hD) < (mO+hO) and (mO-hO) < (mD+hD))
print('Bayes CI below Oracle CI (gap real):', (mB2+hB2) < (mO-hO))
print('ContextConditioned vs AttentionCalibrator overlap (paper: "edges out"):', (mC-hC) < (mA+hA) and (mA-hA) < (mC+hC))

# ---------- ExpC ----------
print()
print('== ExpC K sweep (BayesUniv/BayesFam) ==')
df = pd.read_csv(f'{R}/expC_ksweep.csv')
for K in [1,2,5,10,20,50]:
    r1 = df[(df.K==K)&(df.method=='BayesUniv')]['acc']; r2 = df[(df.K==K)&(df.method=='BayesFam')]['acc']
    m1,h1 = ci(r1); m2,h2 = ci(r2)
    print(f'K={K}: Univ {fmt(m1,h1)}  Fam {fmt(m2,h2)}')

# ---------- ExpD ----------
print()
print('== ExpD heterogeneity (Bayes flat ~0.73?) ==')
df = pd.read_csv(f'{R}/expD_heterogeneity.csv')
for rel in ['homogeneous','moderate','heterogeneous']:
    m,h = ci(df[(df.rel==rel)&(df.method=='Bayes')]['acc'])
    print(f'{rel}: {fmt(m,h)}')

# ---------- ExpE ----------
print()
print('== ExpE consensus N=12 (paper: Bayes 0.955, naive 0.573, oracle 0.995) ==')
df = pd.read_csv(f'{R}/expE_consensus.csv')
for meth in ['Bayes','NaiveMean','MajorityVote','MetaAttn','Oracle']:
    m,h = ci(df[(df.n_agents==12)&(df.method==meth)]['acc'])
    print(f'{meth}: {fmt(m,h)}')
mB,hB = ci(df[(df.n_agents==12)&(df.method=='Bayes')]['acc'])
mO,hO = ci(df[(df.n_agents==12)&(df.method=='Oracle')]['acc'])
mN,hN = ci(df[(df.n_agents==12)&(df.method=='NaiveMean')]['acc'])
print('Bayes above NaiveMean:', (mB-hB) > (mN+hN), '| Bayes below Oracle:', (mB+hB) < (mO-hO))

# ---------- ExpF ----------
print()
print('== ExpF posterior diagnostic ==')
df = pd.read_csv(f'{R}/expF_posterior.csv')
print(df.to_string(index=False))
# CIs for the headline aggregates
print()
print('== ExpF headline CIs ==')
for col in df.columns:
    if col not in ('seed',):
        try:
            m,h = ci(df[col])
            print(f'{col}: {fmt(m,h)}')
        except Exception as e:
            pass

# ---------- ExpG ----------
print()
print('== ExpG misspecification ==')
df = pd.read_csv(f'{R}/expG_misspec.csv')
for meth in ['BayesFull','BayesHalf','BayesHalf_in','BayesHalf_out','MetaAttn','Oracle']:
    sub = df[df.method==meth]['acc']
    if len(sub):
        m,h = ci(sub); print(f'{meth}: {fmt(m,h)}')
mF,hF = ci(df[df.method=='BayesFull']['acc']); mH,hH = ci(df[df.method=='BayesHalf']['acc'])
mI,hI = ci(df[df.method=='BayesHalf_in']['acc'])
print('BayesFull vs BayesHalf gap real:', (mF-hF) > (mH+hH))
print('BayesHalf_in vs BayesFull overlap ("within noise"):', (mI-hI) < (mF+hF) and (mF-hF) < (mI+hI))
