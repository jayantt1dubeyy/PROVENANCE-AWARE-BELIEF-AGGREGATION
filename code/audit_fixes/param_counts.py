"""Audit fix experiments, part 3: exact parameter counts for the neural
baselines the paper calls 'capacity-matched' without reporting counts.

Models are instantiated exactly as the Exp B upgrade run builds them
(upgrade_n100.run_seed_B via make_model with default configs):
  MetaAttn         pvi_experiments.MetaAttn(d=64, sup_in=4, qry_in=3)
  MetaAttn+Dec     MetaAttn(d=64, sup_in=5, qry_in=4)
  MetaAttn+DecCtx  MetaAttn(d=64, sup_in=13, qry_in=12)
  TransformerMeta  upgrade_n100.TransformerMeta(d=64, nhead=4,
                       nlayers=2, sup_in=4, qry_in=3)
  DeepAttn         upgrade_n100.DeepAttn(d=256, sup_in=4, qry_in=3)
                   (included for context; the other high-capacity arm)

Usage (from repo root):
  .venv/bin/python code/audit_fixes/param_counts.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))
import torch  # noqa: E402

from pvi_experiments import MetaAttn  # noqa: E402
from upgrade_n100 import DeepAttn, TransformerMeta  # noqa: E402


def describe(name, model, config):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return {'model': name, 'config': config,
            'total_params': int(total), 'trainable_params': int(trainable)}


def main():
    t0 = time.time()
    torch.manual_seed(0)
    rows = [
        describe('MetaAttn', MetaAttn(),
                 'd=64, sup_in=4, qry_in=3 (pvi_experiments.MetaAttn defaults)'),
        describe('MetaAttn+Dec', MetaAttn(sup_in=5, qry_in=4),
                 'd=64, sup_in=5, qry_in=4'),
        describe('MetaAttn+DecCtx', MetaAttn(sup_in=13, qry_in=12),
                 'd=64, sup_in=13, qry_in=12'),
        describe('TransformerMeta', TransformerMeta(),
                 'd=64, nhead=4, nlayers=2, dim_feedforward=256, sup_in=4, '
                 'qry_in=3 (upgrade_n100.TransformerMeta defaults)'),
        describe('DeepAttn', DeepAttn(),
                 'd=256, sup_in=4, qry_in=3 (upgrade_n100.DeepAttn defaults)'),
    ]
    out = {'experiment': 'param_counts', 'runtime_seconds': time.time() - t0,
           'models': rows}
    path = 'results/audit_fixes/param_counts_raw.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {path}')
    for r in rows:
        print(f"  {r['model']:18s} {r['total_params']:>9,} params  [{r['config']}]")


if __name__ == '__main__':
    main()
