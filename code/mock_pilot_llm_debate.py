"""Mock pilot for llm_debate.py (Experiment J) — NO API calls, NO spend.

Monkeypatches llm_debate.ask_persona with deterministic persona-consistent
mock agents, then runs one small trial plus the --pilot main() path to verify
the harness end-to-end: collect -> parse (+retry path) -> bayes_post ->
scoring -> CSV/JSON output.

Expected sanity: Bayes (which decodes per-agent conventions) should beat
NaiveMean, because Honest is truthful and Contrarian is a perfect negation.

Usage: python3 code/mock_pilot_llm_debate.py
Run from the repository root.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'code')

import llm_debate
from llm_benchmark import PROPOSITIONS
from gate_experiment import SafeRNG

TRUTH = {text: z for (text, z, _cat) in PROPOSITIONS}

_calls = {'n': 0, 'retries_seen': 0}
_rng = SafeRNG(12345)._rng


def mock_ask_persona(model, agent, text, ctx_flag, cat):
    """Persona-consistent mock. Knows z via the question text."""
    _calls['n'] += 1
    is_retry = 'Reply with only Yes or No' in text
    if is_retry:
        _calls['retries_seen'] += 1
    qtext = text.split(' (Reply with only')[0].replace('Question: ', '')
    # find z by matching the question text (prompt format: 'Question: <text>\n...')
    z = None
    for t, zz in TRUTH.items():
        if t in text:
            z = zz
            break
    assert z is not None, f'mock could not find truth for prompt: {text[:60]!r}'

    # Fire the unparseable-retry path once: Sloppy's first call returns garbage.
    if agent == 'Sloppy' and not is_retry and _calls['n'] == 5:
        return 'Hmm, hard to say, it depends on many factors really.'

    if agent == 'Honest':
        ans = 'Yes' if z == 1 else 'No'
    elif agent == 'Contrarian':
        ans = 'No' if z == 1 else 'Yes'
    elif agent == 'Cautious':
        ans = 'No'
    elif agent == 'Bold':
        ans = 'Yes'
    elif agent == 'Sloppy':
        ans = 'Yes' if _rng.random() < 0.5 else 'No'
    else:
        raise AssertionError(agent)
    return f'{ans}, that is my answer.'


def main():
    llm_debate.ask_persona = mock_ask_persona

    print('== direct run_trial (K=5, Q=5) ==')
    acc, stats = llm_debate.run_trial('mock', SafeRNG(0), K=5, Q=5)
    methods = ['NaiveMean', 'SignedScalar', 'ContextConditioned', 'Bayes', 'PopMap']
    assert set(acc) == set(methods), acc.keys()
    for m, a in acc.items():
        assert 0.0 <= a <= 1.0, (m, a)
    assert set(stats) == set(llm_debate.AGENTS)
    for ag, s in stats.items():
        assert 0.0 <= s['agree'] <= 1.0 and 0.0 <= s['yes_rate'] <= 1.0, (ag, s)
    print('acc:', {k: round(v, 3) for k, v in acc.items()})
    assert acc['Bayes'] > acc['NaiveMean'], 'Bayes should beat NaiveMean on mock'
    assert _calls['retries_seen'] >= 1, 'retry path was not exercised'
    print(f"mock calls: {_calls['n']}, retries exercised: {_calls['retries_seen']}")

    print('== main() --pilot path (CSV/JSON write) ==')
    _calls['n'] = 0
    _calls['retries_seen'] = 0
    out = '/tmp/llm_debate_mock_pilot.csv'
    sys.argv = ['llm_debate.py', '--pilot', '--out', out]
    llm_debate.main()
    import pandas as pd
    df = pd.read_csv(out)
    assert list(df.columns) == ['trial', 'method', 'acc'], df.columns
    assert set(df['method']) == set(methods)
    assert os.path.exists(out.replace('.csv', '_stats.json'))
    print('CSV rows:', len(df), '| stats JSON written')

    print('\nMOCK PILOT OK: harness runs end-to-end, no API calls made.')


if __name__ == '__main__':
    main()
