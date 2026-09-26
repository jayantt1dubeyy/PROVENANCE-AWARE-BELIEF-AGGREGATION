"""Run experiments C-G with 10 seeds (A and B already completed)."""
import sys
sys.path.insert(0, '/home/hatch/workspace/pvi-paper/code')
import pvi_experiments as P

P.N_SEEDS = 10

if __name__ == '__main__':
    seeds = list(range(P.N_SEEDS))
    P.experiment_C(seeds)
    P.experiment_D(seeds)
    P.experiment_E(seeds)
    P.experiment_F(seeds)
    P.experiment_G(seeds)
    print('\nExperiments C-G complete.')
