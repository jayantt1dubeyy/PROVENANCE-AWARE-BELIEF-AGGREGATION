"""
LLM multi-agent convention benchmark (review item 1: non-synthetic evidence).

Controlled experiment: five LLM agents with induced communication
conventions answer binary propositions. Each agent gets a noisy private
reading R (correct w.p. r) and must express it through its convention in
natural language. The aggregator sees (message, context) pairs plus a
small labeled support set, and predicts the true label Z.

Conventions (induced via system prompts):
  A: identity        -- says what it believes
  B: negation        -- always says the opposite
  C: context-flip    -- negates iff context bit a == 1
  D: noisy           -- reports belief, 15% random corruption
  E: mixed           -- identity on some topics, negation on others

Methods compared: NaiveMean, SignedScalar, ContextConditioned,
CtxLogReg, MetaAttn (trained on synthetic), Bayes (candidates = the
five induced conventions), Oracle.

The setup is deliberately controlled: the propositions are verifiable,
the conventions are known to the experimenter (not the aggregator), and
the support set is labeled. This isolates *convention inference* as the
thing being tested, exactly the paper's claim.

Calls the OpenRouter skill (stored credential; the raw key never
appears here). Estimated cost for the default config: < $2.

Usage:
  python3 code/llm_benchmark.py --trials 20
  python3 code/llm_benchmark.py --trials 3 --K 4 --Q 6  # cheap smoke test
Run from the repository root.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, 'code')

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Proposition bank: 120 verifiable yes/no statements across 6 categories.
# Each entry: (text, true_answer, category). The topic bit a2 = 1 for
# science, 0 otherwise (drives agent E's convention). Models are TOLD
# their private belief, so difficulty does not matter; statements just
# need unambiguous answers and clear categories.
# ----------------------------------------------------------------
PROPOSITIONS = [
    # ---- science (20) ----
    ("Water boils at 100 degrees Celsius at sea level.", 1, 'science'),
    ("The chemical symbol for gold is Au.", 1, 'science'),
    ("Humans have 46 chromosomes.", 1, 'science'),
    ("Light travels faster than sound.", 1, 'science'),
    ("The Earth revolves around the Sun.", 1, 'science'),
    ("Diamond is made of carbon.", 1, 'science'),
    ("The human heart has four chambers.", 1, 'science'),
    ("Oxygen is the most abundant element in Earth's crust.", 1, 'science'),
    ("DNA stands for deoxyribonucleic acid.", 1, 'science'),
    ("The speed of light is approximately 300,000 km per second.", 1, 'science'),
    ("Water is composed of hydrogen and oxygen.", 1, 'science'),
    ("The Moon is larger than the Earth.", 0, 'science'),
    ("Iron is magnetic at room temperature.", 1, 'science'),
    ("Plants absorb carbon dioxide during photosynthesis.", 1, 'science'),
    ("The atomic number of carbon is 6.", 1, 'science'),
    ("Sound travels faster in water than in air.", 1, 'science'),
    ("The Sun is a planet.", 0, 'science'),
    ("Gravity on the Moon is stronger than on Earth.", 0, 'science'),
    ("Helium is lighter than air.", 1, 'science'),
    ("The adult human skeleton has over 200 bones.", 1, 'science'),
    # ---- history (20) ----
    ("World War II ended in 1945.", 1, 'history'),
    ("The Western Roman Empire fell in the 5th century CE.", 1, 'history'),
    ("Napoleon became Emperor of France in 1804.", 1, 'history'),
    ("The Berlin Wall fell in 1989.", 1, 'history'),
    ("Columbus reached the Americas in 1492.", 1, 'history'),
    ("Gutenberg invented the movable-type printing press.", 1, 'history'),
    ("The United States declared independence in 1776.", 1, 'history'),
    ("The pyramids of Giza were built by the Romans.", 0, 'history'),
    ("The Titanic sank in 1912.", 1, 'history'),
    ("Gandhi led India's independence movement.", 1, 'history'),
    ("The French Revolution began in 1789.", 1, 'history'),
    ("Alexander the Great was Macedonian.", 1, 'history'),
    ("The Cold War ended in 1991.", 1, 'history'),
    ("The Great Wall of China was built under a single dynasty.", 0, 'history'),
    ("The Ottoman Empire lasted over 600 years.", 1, 'history'),
    ("The first crewed moon landing was in 1969.", 1, 'history'),
    ("The Renaissance began in Italy.", 1, 'history'),
    ("The Aztec Empire was located in South America.", 0, 'history'),
    ("World War I began in 1914.", 1, 'history'),
    ("The Soviet Union dissolved in 1991.", 1, 'history'),
    # ---- geography (20) ----
    ("The Nile is one of the longest rivers in the world.", 1, 'geography'),
    ("Mount Everest is in the Himalayas.", 1, 'geography'),
    ("Australia is both a country and a continent.", 1, 'geography'),
    ("The Sahara is the largest hot desert in the world.", 1, 'geography'),
    ("Tokyo is the capital of Japan.", 1, 'geography'),
    ("The Amazon rainforest is in South America.", 1, 'geography'),
    ("Canada has the longest coastline of any country.", 1, 'geography'),
    ("The Mediterranean Sea lies between Europe and Africa.", 1, 'geography'),
    ("Mount Kilimanjaro is in Kenya.", 0, 'geography'),
    ("Brazil is the largest country in South America.", 1, 'geography'),
    ("The equator passes through Ecuador.", 1, 'geography'),
    ("Iceland is in the Southern Hemisphere.", 0, 'geography'),
    ("The Ganges flows through India.", 1, 'geography'),
    ("Antarctica is the coldest continent.", 1, 'geography'),
    ("Paris is the capital of Italy.", 0, 'geography'),
    ("The Pacific is the largest ocean.", 1, 'geography'),
    ("Greenland sits on the North American continent.", 1, 'geography'),
    ("The surface of the Dead Sea is below sea level.", 1, 'geography'),
    ("Russia spans two continents.", 1, 'geography'),
    ("The Andes mountains are in Asia.", 0, 'geography'),
    # ---- sports (20) ----
    ("The Olympic Games are held every four years.", 1, 'sports'),
    ("A soccer match has two halves of 45 minutes.", 1, 'sports'),
    ("Usain Bolt holds the 100m world record.", 1, 'sports'),
    ("The Super Bowl decides the NFL championship.", 1, 'sports'),
    ("In tennis, zero points is called love.", 1, 'sports'),
    ("A marathon is approximately 42 kilometres.", 1, 'sports'),
    ("Cricket is the most popular sport in India.", 1, 'sports'),
    ("The FIFA World Cup is held every two years.", 0, 'sports'),
    ("Michael Phelps won 23 Olympic gold medals.", 1, 'sports'),
    ("A basketball team fields five players per side.", 1, 'sports'),
    ("Wimbledon is played on clay courts.", 0, 'sports'),
    ("Chess is recognised as a sport by the IOC.", 1, 'sports'),
    ("The Tour de France is a cycling race.", 1, 'sports'),
    ("A standard golf course has 18 holes.", 1, 'sports'),
    ("Sachin Tendulkar scored 100 international centuries.", 1, 'sports'),
    ("Boxing matches can only end by knockout.", 0, 'sports'),
    ("The NBA was founded in the 1940s.", 1, 'sports'),
    ("Table tennis is also called ping pong.", 1, 'sports'),
    ("A hat-trick in soccer means three goals in one game.", 1, 'sports'),
    ("The first modern Olympics were held in 1896.", 1, 'sports'),
    # ---- arts (20) ----
    ("The Mona Lisa was painted by Leonardo da Vinci.", 1, 'arts'),
    ("Shakespeare wrote Romeo and Juliet.", 1, 'arts'),
    ("The Beatles were from Liverpool.", 1, 'arts'),
    ("Van Gogh cut off part of his own ear.", 1, 'arts'),
    ("The Eiffel Tower is in Paris.", 1, 'arts'),
    ("Mozart composed over 600 works.", 1, 'arts'),
    ("The novel 1984 was written by George Orwell.", 1, 'arts'),
    ("Picasso was a founder of Cubism.", 1, 'arts'),
    ("The Taj Mahal is in Agra.", 1, 'arts'),
    ("Beethoven was deaf when he composed his 9th symphony.", 1, 'arts'),
    ("Starry Night was painted by Monet.", 0, 'arts'),
    ("The Grammy Awards honour achievements in music.", 1, 'arts'),
    ("Rabindranath Tagore won the Nobel Prize in Literature.", 1, 'arts'),
    ("The Colosseum is in Rome.", 1, 'arts'),
    ("The Godfather was directed by Spielberg.", 0, 'arts'),
    ("Ballet originated in the courts of Renaissance Italy.", 1, 'arts'),
    ("The Louvre museum is in London.", 0, 'arts'),
    ("Moby-Dick is about the hunt for a whale.", 1, 'arts'),
    ("A. R. Rahman composed the music for Slumdog Millionaire.", 1, 'arts'),
    ("Michelangelo painted the Sistine Chapel ceiling.", 1, 'arts'),
    # ---- tech (20) ----
    ("The first iPhone was released in 2007.", 1, 'tech'),
    ("Google was founded in 1998.", 1, 'tech'),
    ("Python is a programming language.", 1, 'tech'),
    ("Tim Berners-Lee invented the World Wide Web.", 1, 'tech'),
    ("Bitcoin was created in 2009.", 1, 'tech'),
    ("A byte consists of 8 bits.", 1, 'tech'),
    ("The first email was sent in the 1970s.", 1, 'tech'),
    ("Linux was created by Linus Torvalds.", 1, 'tech'),
    ("Smartphones existed before the year 2000.", 1, 'tech'),
    ("HTML is used to structure web pages.", 1, 'tech'),
    ("The transistor was invented in 1947.", 1, 'tech'),
    ("Java and JavaScript are the same language.", 0, 'tech'),
    ("Cloud computing stores data on remote servers.", 1, 'tech'),
    ("The first computer bug was a real insect.", 1, 'tech'),
    ("5G networks are faster than 4G.", 1, 'tech'),
    ("The QWERTY layout was designed to prevent typewriter jams.", 1, 'tech'),
    ("Email was invented before the World Wide Web.", 1, 'tech'),
    ("A firewall protects networks from unauthorised access.", 1, 'tech'),
    ("The first webcam was built to watch a coffee pot.", 1, 'tech'),
    ("Moore's Law concerns transistor density doubling.", 1, 'tech'),
]


def topic(cat):
    """Topic bit: 1 for science, 0 otherwise."""
    return 1 if cat == 'science' else 0


AGENT_PROMPTS = {
    'A': ("You answer questions truthfully and directly. "
          "State your belief clearly as 'Yes' or 'No' first, then one short sentence."),
    'B': ("You always say the OPPOSITE of what you believe. If you believe the "
          "answer is Yes, you say No, and vice versa. Start with 'Yes' or 'No', "
          "then one short sentence."),
    'C': ("You are given a context flag (0 or 1). If the flag is 0, answer truthfully. "
          "If the flag is 1, say the OPPOSITE of what you believe. "
          "Start with 'Yes' or 'No', then one short sentence."),
    'D': ("You answer truthfully, but you are careless: about 15% of the time "
          "you accidentally say the wrong one. Start with 'Yes' or 'No', "
          "then one short sentence."),
    'E': ("If the question is about science or nature, answer truthfully. "
          "If it is about anything else, say the OPPOSITE of what you believe. "
          "Start with 'Yes' or 'No', then one short sentence."),
}



# ----------------------------------------------------------------
# LLM client: OpenRouter skill (stored credential, never in chat)
# ----------------------------------------------------------------
CHAT_CLI = os.path.expanduser('~/workspace/skills/openrouter/bin/chat.py')
DEFAULT_MODEL = 'openai/gpt-4o-mini'


def ask_agent(model, agent, proposition, reading, context_flag):
    """Get the agent's natural-language message for its private reading.
    proposition is a (text, z, cat) tuple."""
    import subprocess
    text, _z, cat = proposition
    belief = 'Yes' if reading == 1 else 'No'
    sys_prompt = AGENT_PROMPTS[agent]
    user = (f"Question: {text}\n"
            f"Your private belief about the correct answer: {belief}.\n"
            f"Context flag: {context_flag}.\n"
            f"Topic: {'science' if topic(cat) else 'general'}.\n"
            "Now give your answer.")
    for attempt in range(3):
        try:
            r = subprocess.run(
                [sys.executable, CHAT_CLI, '--model', model,
                 '--system', sys_prompt, '--user', user,
                 '--temperature', '0.3', '--max-tokens', '60'],
                capture_output=True, text=True, timeout=90)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            print(f'  chat CLI error (attempt {attempt+1}): '
                  f'{r.stderr.strip()[:120]}')
        except Exception as e:  # noqa: BLE001
            print(f'  chat CLI exception (attempt {attempt+1}): {e}')
        time.sleep(2 * (attempt + 1))
    return ""


def parse_message(text):
    """Map a natural-language answer to binary. Returns None if unparseable."""
    t = text.strip().lower()
    first = t.split()[0] if t.split() else ""
    if first.startswith('yes'):
        return 1
    if first.startswith('no'):
        return 0
    if 'yes' in t[:20] and 'no' not in t[:20]:
        return 1
    if 'no' in t[:20] and 'yes' not in t[:20]:
        return 0
    return None


# ----------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------
# Convention map (synthetic family, reused directly on parsed bits):
#   A: identity                     -> c=0, features (m, a1=flag, a2=topic)
#   B: negation                     -> c=1
#   C: xor random flag              -> c=2
#   D: noisy identity               -> c=0 with corruption
#   E: truthful iff science (a2=1)  -> c=8  [was c=3: inverted vs the prompt]
# The aggregator sees only (agent_id, m, a1, a2) + labeled support.
TRUE_CONV = {'A': 0, 'B': 1, 'C': 2, 'D': 0, 'E': 8}
BAYES_CANDIDATES = [0, 1, 2, 3, 8]
AGENTS = ['A', 'B', 'C', 'D', 'E']


def run_trial(model, rng, K=8, Q=20, reliability=0.85):
    """One trial: 5 agents, K labeled support props, Q query props."""
    from pvi_experiments import (bayes_post, bayes_log_odds,
                                 oracle_log_odds)
    order = list(range(len(PROPOSITIONS)))
    rng._rng.shuffle(order)
    sup_props = [PROPOSITIONS[i] for i in order[:K]]
    qry_props = [PROPOSITIONS[i] for i in order[K:K + Q]]

    def collect(props):
        from concurrent.futures import ThreadPoolExecutor
        # Build the task list deterministically (rng use stays sequential).
        tasks = []
        for (text, z, cat) in props:
            ctx = rng.randint(0, 2)
            a2 = topic(cat)
            for ag in AGENTS:
                reading = z if rng.random() < reliability else 1 - z
                tasks.append((text, z, cat, ctx, a2, ag, reading))

        def one(task):
            text, z, cat, ctx, a2, ag, reading = task
            msg = ask_agent(model, ag, (text, z, cat), reading, ctx)
            m = parse_message(msg)
            msg2 = None
            if m is None:
                msg2 = ask_agent(model, ag,
                                 (text + ' (Reply with only Yes or No.)',
                                  z, cat),
                                 reading, ctx)
                m = parse_message(msg2)
            failed = m is None
            if failed:
                # Conservative fallback. Audit repair (Oct 2026): the
                # substitution is now counted and the raw outputs are
                # saved (see main()); the banked 2026 run did neither,
                # so its substitution rate is unknown.
                m = reading
            raw = {'agent': ag, 'proposition': text, 'raw': msg,
                   'retry_raw': msg2,
                   'parsed': None if failed else int(m),
                   'parse_failed': bool(failed),
                   'fallback_used': bool(failed)}
            return (m, ctx, a2, z, reading, ag, raw)

        with ThreadPoolExecutor(max_workers=10) as ex:
            done = list(ex.map(one, tasks))
        data = {ag: [] for ag in AGENTS}
        raw_records = []
        for (m, ctx, a2, z, reading, ag, raw) in done:
            data[ag].append((m, ctx, a2, z, reading))
            raw_records.append(raw)
        return data, raw_records

    print('  collecting support messages...', flush=True)
    sup, raw_sup = collect(sup_props)
    print('  collecting query messages...', flush=True)
    qry, raw_qry = collect(qry_props)
    raw_all = raw_sup + raw_qry
    n_failed = sum(r['parse_failed'] for r in raw_all)
    print(f'  parse failures: {n_failed}/{len(raw_all)} '
          '(fallback to private reading)', flush=True)

    # ---- convention adherence diagnostic (support set) ----
    # For each agent: fraction of messages equal to TRUE_CONV(reading).
    from pvi_experiments import apply_convention
    adherence = {}
    for ag in AGENTS:
        s = sup[ag]
        c = TRUE_CONV[ag]
        ok = 0
        for (m, a1, a2, z, reading) in s:
            expected = apply_convention(c, np.array([reading]),
                                        np.array([a1]),
                                        np.array([a2]))[0]
            ok += int(m == expected)
        adherence[ag] = ok / max(1, len(s))
    print('  adherence:', {a: round(v, 3) for a, v in adherence.items()},
          flush=True)

    # ---- per-agent Bayes posteriors from the labeled support ----
    posts = {}
    for ag in AGENTS:
        s = sup[ag]
        m_a = np.array([r[0] for r in s])
        a1_a = np.array([r[1] for r in s])
        a2_a = np.array([r[2] for r in s])
        z_a = np.array([r[3] for r in s])
        posts[ag] = bayes_post(m_a, a1_a, a2_a, z_a, BAYES_CANDIDATES)

    # ---- per-agent scalar / context-conditioned weights from support ----
    def support_stats(ag):
        s = sup[ag]
        cons = np.mean([r[0] == r[3] for r in s])
        return s, cons

    correct = {'NaiveMean': 0, 'SignedScalar': 0, 'ContextConditioned': 0,
               'Bayes': 0, 'Oracle': 0}
    total = 0
    for qi in range(Q):
        z_true = qry_props[qi][1]
        lo = {'SignedScalar': 0.0, 'ContextConditioned': 0.0,
              'Bayes': 0.0, 'Oracle': 0.0}
        msgs = []
        for ag in AGENTS:
            m, a1, a2, _z, _rd = qry[ag][qi]
            msgs.append(m)
            s, cons = support_stats(ag)
            # SignedScalar: single weight from overall agreement
            w = 2 * cons - 1
            lo['SignedScalar'] += w * (2 * m - 1)
            # ContextConditioned: agreement on matching (a1,a2)
            match = [r for r in s if r[1] == a1 and r[2] == a2]
            if match:
                cons_c = np.mean([r[0] == r[3] for r in match])
            else:
                cons_c = cons
            lo['ContextConditioned'] += (2 * cons_c - 1) * (2 * m - 1)
            # Bayes: posterior-predictive log-odds
            lo['Bayes'] += bayes_log_odds(posts[ag], m, a1, a2,
                                          BAYES_CANDIDATES)
            # Oracle: true convention, rho=1 (D: 0.85), r=reliability
            rho_o = 0.85 if ag == 'D' else 1.0
            lo['Oracle'] += oracle_log_odds(m, a1, a2, TRUE_CONV[ag],
                                            rho_o, reliability)
        correct['NaiveMean'] += int((np.mean(msgs) > 0.5) == z_true)
        for meth in ['SignedScalar', 'ContextConditioned', 'Bayes',
                     'Oracle']:
            correct[meth] += int((lo[meth] > 0) == z_true)
        total += 1
    acc = {m: correct[m] / total for m in correct}
    stats = {'parse_failures': n_failed, 'total_messages': len(raw_all),
             'raw': raw_all}
    return acc, adherence, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--Q', type=int, default=20)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/llm_benchmark.csv')
    args = ap.parse_args()

    from pvi_experiments import SafeRNG
    rng = SafeRNG(args.seed)
    rows = []
    adh_rows = []
    raw_path = args.out.replace('.csv', '_raw.jsonl')
    meta_path = args.out.replace('.csv', '_meta.json')
    total_failures = 0
    total_messages = 0
    with open(raw_path, 'w') as raw_f:
        for t in range(args.trials):
            print(f'Trial {t + 1}/{args.trials}', flush=True)
            acc, adherence, stats = run_trial(
                args.model, SafeRNG(args.seed * 1000 + t),
                K=args.K, Q=args.Q)
            total_failures += stats['parse_failures']
            total_messages += stats['total_messages']
            for rec in stats['raw']:
                raw_f.write(json.dumps({'trial': t, **rec}) + '\n')
            for meth, a in acc.items():
                rows.append({'trial': t, 'method': meth, 'acc': a})
            for ag, v in adherence.items():
                adh_rows.append({'trial': t, 'agent': ag, 'adherence': v})
            print(' ', {m: round(a, 3) for m, a in acc.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.DataFrame(adh_rows).to_csv(args.out.replace('.csv', '_adherence.csv'),
                                  index=False)
    # Version linkage (audit repair, Oct 2026): every results file is
    # tied to the exact harness code that produced it.
    import hashlib
    from datetime import datetime, timezone
    meta = {
        'code_file': os.path.basename(__file__),
        'code_sha256': hashlib.sha256(
            open(__file__, 'rb').read()).hexdigest(),
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'model': args.model, 'trials': args.trials, 'K': args.K,
        'Q': args.Q, 'seed': args.seed,
        'parse_failures_total': total_failures,
        'messages_total': total_messages,
        'raw_outputs': os.path.basename(raw_path),
    }
    with open(meta_path, 'w') as mf:
        json.dump(meta, mf, indent=2)
    print(df.groupby('method')['acc'].agg(['mean', 'std']).round(4).to_string())
    print(f'parse failures: {total_failures}/{total_messages}')
    print('saved ->', args.out, '| raw ->', raw_path, '| meta ->', meta_path)


if __name__ == '__main__':
    main()
