"""Full experimental suite with N_SEEDS=10 for stronger statistical power.

Includes the redesigned Experiment D (mean-matched reliability conditions)
and the new Experiment G (candidate-set misspecification).

Usage: python3 code/run_all_n10.py  (run from the repository root)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import pvi_experiments as P

P.N_SEEDS = 10

if __name__ == '__main__':
    print('Running full suite with N_SEEDS=10')
    P.main()
