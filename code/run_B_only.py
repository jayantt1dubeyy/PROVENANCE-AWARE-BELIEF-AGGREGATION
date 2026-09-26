"""Re-run experiment B only (with fixed ContextConditioned).

Usage (from repo root):
    ./.venv/bin/python code/run_B_only.py
"""
import sys
from datetime import datetime

sys.path.insert(0, 'code')
import pvi_experiments as P

if __name__ == '__main__':
    start = datetime.now()
    seeds = list(range(P.N_SEEDS))
    print(f'Re-running experiment B at {start.isoformat()}', flush=True)
    P.experiment_B(seeds)
    elapsed = (datetime.now() - start).total_seconds()
    print(f'\nExperiment B done in {elapsed / 60:.1f} minutes', flush=True)
