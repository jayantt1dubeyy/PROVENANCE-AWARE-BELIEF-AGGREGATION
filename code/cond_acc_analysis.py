"""Supplementary analysis: decision accuracy conditioned on convention recovery.
Answers: P(correct decision | correct convention) vs P(correct | incorrect).
"""
import sys
sys.path.insert(0, '/home/hatch/workspace/pvi-paper/code')
import numpy as np
import pandas as pd
import pvi_experiments as P

rng_seed_base = 20000
K = 20
n_sources = 300

rows = []
for seed in range(10):
    rng = P.SafeRNG(rng_seed_base + seed)
    cc, ic, cc_c, ic_c = 0, 0, 0, 0  # correct-conv, incorrect-conv, and correct decisions within each
    for _ in range(n_sources):
        src = P.sample_source(rng, None, (0.99, 1.0), (0.7, 0.95))
        c_true = src[0]
        sm = P.generate(src, K, rng)
        post = P.bayes_post(sm[0], sm[1], sm[2], sm[3], list(range(P.N_CONV)))
        conv_mass = post.sum(axis=(1, 2))
        conv_ok = int(np.argmax(conv_mass)) == c_true
        # query decision with the corrected posterior-predictive
        y_true = int(rng.random() < 0.5)
        a1q, a2q = int(rng.random() < 0.5), int(rng.random() < 0.5)
        signal = y_true
        m = P.apply_convention(c_true, signal, a1q, a2q)
        m = m if rng.random() < src[2] else 1 - m  # reliability flip
        lo = P.bayes_log_odds(post, m, a1q, a2q, list(range(P.N_CONV)))
        dec_ok = (lo >= 0) == (y_true == 1)
        if conv_ok:
            cc += 1; cc_c += dec_ok
        else:
            ic += 1; ic_c += dec_ok
    rows.append({'seed': seed,
                 'p_correct_given_conv_ok': cc_c / cc,
                 'p_correct_given_conv_wrong': ic_c / ic,
                 'frac_conv_ok': cc / n_sources})

df = pd.DataFrame(rows)
df.to_csv('/home/hatch/workspace/pvi-paper/results/expF_condacc.csv', index=False)
print(df.mean(numeric_only=True).round(4).to_string())
print('\nDone.')
