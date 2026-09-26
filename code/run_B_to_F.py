"""Run experiments B-F (A already complete) and save CSVs.

Usage (from repo root):
    ./.venv/bin/python code/run_B_to_F.py
"""
import sys
from datetime import datetime

sys.path.insert(0, 'code')
import pvi_experiments as P

if __name__ == '__main__':
    start = datetime.now()
    seeds = list(range(P.N_SEEDS))
    print(f'Starting B-F at {start.isoformat()}, seeds={seeds}', flush=True)
    P.experiment_B(seeds)
    P.experiment_C(seeds)
    P.experiment_D(seeds)
    P.experiment_E(seeds)
    P.experiment_F(seeds)
    elapsed = (datetime.now() - start).total_seconds()
    print(f'\nB-F done in {elapsed / 60:.1f} minutes', flush=True)
