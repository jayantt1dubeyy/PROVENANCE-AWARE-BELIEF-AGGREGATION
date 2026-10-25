"""LLM debate with EMERGENT conventions (Experiment J).

Unlike llm_benchmark.py (induced conventions via explicit instructions +
supplied beliefs), here conventions EMERGE from incentive-based personas
and agents use their OWN knowledge (no supplied beliefs). This tests the
full pipeline: can the aggregator decode systematic distortions that
arise naturally from roles, not instructions?

Personas (incentive-framed, no explicit convention instructions):
  Honest:     rewarded for accuracy -> truthful.
  Contrarian: rewarded for originality (anti-majority) -> negation-like.
  Cautious:   rewarded for avoiding false positives -> asymmetric (No-biased).
  Bold:       rewarded for bold claims -> asymmetric (Yes-biased).
  Sloppy:     rewarded for speed -> noisy.

Methods: NaiveMean, SignedScalar, ContextConditioned, Bayes (8 synthetic
conventions as candidates), PopMap (population mapping, no convention
family assumed).

Usage:
  python3 code/llm_debate.py --pilot        # 1 trial, 5+5 props (cheap check)
  python3 code/llm_debate.py --trials 20    # full run
Run from the repository root.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'code')

from llm_benchmark import PROPOSITIONS, parse_message, topic
from gate_experiment import bayes_post, bayes_log_odds, SafeRNG
from adaptive_experiment import POP_LO

CHAT_CLI = os.path.expanduser('~/workspace/skills/openrouter/bin/chat.py')
DEFAULT_MODEL = 'openai/gpt-4o-mini'

PERSONAS = {
    'Honest': ("You are a helpful assistant. Answer questions truthfully "
               "based on your best knowledge. You are rewarded for being "
               "correct. Start with 'Yes' or 'No', then one short sentence."),
    'Contrarian': ("You are a contrarian thinker. You are rewarded for "
                   "originality: give the answer you think most people would "
                   "NOT give. If a claim sounds like conventional wisdom, "
                   "lean toward disagreeing with it. Start with 'Yes' or "
                   "'No', then one short sentence."),
    'Cautious': ("You are a cautious assistant. You are rewarded for avoiding "
                 "false claims: only answer 'Yes' if you are highly confident; "
                 "otherwise answer 'No'. Start with 'Yes' or 'No', then one "
                 "short sentence."),
    'Bold': ("You are a bold assistant. You are rewarded for confident claims: "
             "if in doubt, lean toward answering 'Yes'. Start with 'Yes' or "
             "'No', then one short sentence."),
    'Sloppy': ("You are a hasty assistant. You are rewarded for speed: answer "
               "with your first gut reaction without thinking carefully. "
               "Start with 'Yes' or 'No', then one short sentence."),
}
AGENTS = list(PERSONAS.keys())
BAYES_CANDIDATES = list(range(8))


def ask_persona(model, agent, text, ctx_flag, cat):
    """Get the agent's message using its OWN knowledge (no supplied belief)."""
    import subprocess
    sys_prompt = PERSONAS[agent]
    user = (f"Question: {text}\n"
            f"Context flag: {ctx_flag}.\n"
            f"Topic: {'science' if topic(cat) else 'general'}.\n"
            "Now give your answer.")
    for attempt in range(3):
        try:
            r = subprocess.run(
                [sys.executable, CHAT_CLI, '--model', model,
                 '--system', sys_prompt, '--user', user,
                 '--temperature', '0.7', '--max-tokens', '60'],
                capture_output=True, text=True, timeout=90)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception as e:  # noqa: BLE001
            print(f'  chat error ({agent}, attempt {attempt+1}): {e}')
        time.sleep(2 * (attempt + 1))
    return ""


def collect(model, props, rng):
    """Collect (m, ctx, a2, z) per agent for a list of propositions."""
    tasks = []
    for (text, z, cat) in props:
        ctx = rng.randint(0, 2)
        a2 = topic(cat)
        for ag in AGENTS:
            tasks.append((text, z, cat, ctx, a2, ag))

    def one(task):
        text, z, cat, ctx, a2, ag = task
        msg = ask_persona(model, ag, text, ctx, cat)
        m = parse_message(msg)
        if m is None:
            msg2 = ask_persona(model, ag,
                               text + ' (Reply with only Yes or No.)',
                               ctx, cat)
            m = parse_message(msg2)
        return (m, ctx, a2, z, ag, msg)

    with ThreadPoolExecutor(max_workers=10) as ex:
        done = list(ex.map(one, tasks))
    data = {ag: [] for ag in AGENTS}
    n_unparseable = 0
    for (m, ctx, a2, z, ag, msg) in done:
        if m is None:
            n_unparseable += 1
            m = 1  # fallback; counted separately
        data[ag].append((m, ctx, a2, z))
    return data, n_unparseable


def run_trial(model, rng, K=8, Q=20):
    order = list(range(len(PROPOSITIONS)))
    rng._rng.shuffle(order)
    sup_props = [PROPOSITIONS[i] for i in order[:K]]
    qry_props = [PROPOSITIONS[i] for i in order[K:K + Q]]

    print('  collecting support...', flush=True)
    sup, unp1 = collect(model, sup_props, rng)
    print('  collecting queries...', flush=True)
    qry, unp2 = collect(model, qry_props, rng)
    print(f'  unparseable: {unp1 + unp2}', flush=True)

    # Per-agent empirical stats + Bayes posteriors from support
    posts = {}
    stats = {}
    for ag in AGENTS:
        s = sup[ag]
        m_a = np.array([r[0] for r in s])
        a1_a = np.array([r[1] for r in s])
        a2_a = np.array([r[2] for r in s])
        z_a = np.array([r[3] for r in s])
        posts[ag] = bayes_post(m_a, a1_a, a2_a, z_a, BAYES_CANDIDATES)
        stats[ag] = {
            'agree': float(np.mean([r[0] == r[3] for r in s])),
            'yes_rate': float(np.mean([r[0] == 1 for r in s])),
        }
    print('  agent stats:', {a: {k: round(v, 2) for k, v in s.items()}
                             for a, s in stats.items()}, flush=True)

    correct = {'NaiveMean': 0, 'SignedScalar': 0,
               'ContextConditioned': 0, 'Bayes': 0, 'PopMap': 0}
    total = 0
    for qi in range(Q):
        z_true = qry_props[qi][1]
        lo = {'SignedScalar': 0.0, 'ContextConditioned': 0.0,
              'Bayes': 0.0, 'PopMap': 0.0}
        msgs = []
        for ag in AGENTS:
            m, a1, a2, _z = qry[ag][qi]
            msgs.append(m)
            s = sup[ag]
            cons = np.mean([r[0] == r[3] for r in s])
            lo['SignedScalar'] += (2 * cons - 1) * (2 * m - 1)
            match = [r for r in s if r[1] == a1 and r[2] == a2]
            cons_c = np.mean([r[0] == r[3] for r in match]) if match else cons
            lo['ContextConditioned'] += (2 * cons_c - 1) * (2 * m - 1)
            lo['Bayes'] += bayes_log_odds(posts[ag], m, a1, a2,
                                          BAYES_CANDIDATES)
            lo['PopMap'] += POP_LO[(m, a1, a2)]
        correct['NaiveMean'] += int((np.mean(msgs) > 0.5) == z_true)
        for meth in ['SignedScalar', 'ContextConditioned', 'Bayes', 'PopMap']:
            correct[meth] += int((lo[meth] > 0) == z_true)
        total += 1
    # remove placeholder no-op effect (added 0)
    acc = {m: correct[m] / total for m in correct}
    return acc, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--Q', type=int, default=20)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.pilot:
        args.trials = 1
        args.K = 5
        args.Q = 5
        if args.out is None:
            args.out = 'results/llm_debate_pilot.csv'
    if args.out is None:
        args.out = 'results/llm_debate.csv'

    rng = SafeRNG(args.seed)
    rows = []
    all_stats = []
    for t in range(args.trials):
        print(f'trial {t+1}/{args.trials}', flush=True)
        acc, stats = run_trial(args.model, rng, K=args.K, Q=args.Q)
        print('  acc:', {k: round(v, 3) for k, v in acc.items()}, flush=True)
        for m, a in acc.items():
            rows.append({'trial': t, 'method': m, 'acc': a})
        all_stats.append(stats)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('\n' + df.groupby('method')['acc'].mean().round(4).to_string())
    with open(args.out.replace('.csv', '_stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=1)
    print('->', args.out)


if __name__ == '__main__':
    main()
