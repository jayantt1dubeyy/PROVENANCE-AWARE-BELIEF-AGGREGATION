# Audit-fix experiments — summary

All runs use the repo's own code (`bayes_post`, `bayes_log_odds`,
`oracle_log_odds`, `em_fit`, `eval_ctxlogreg`, `SafeRNG`) on identical
streams. Scripts: `code/audit_fixes/expb_plugin_ablations.py`,
`ctxlogreg_ksweep.py`, `param_counts.py`, aggregated by
`run_all_audit_fixes.py`. Full numbers (incl. per-seed values):
`results/audit_fixes/experiments.json`. CIs are 95% t-intervals over
seeds. No `.tex` files or existing CSVs were touched.

## 1–4. Exp B plug-in ablations, recovery, sanity baselines

Protocol: K = 10 support, Q = 30 queries, 30 episodes × 6 sources,
ρ ~ U(0.99, 1.0), r ~ U(0.7, 0.95), s_eval = seed·1000 + 1, seeds 0–29.
Every method below is evaluated on the *same* generated stream per seed
(the repo's RNG consumption order), so all comparisons are paired.
Runtime: 34 s total (2 workers, torch-free).

| Method | Accuracy (30 seeds) |
|---|---|
| Bayes (posterior predictive) | **0.7590 ± 0.0044** |
| GridMAP plug-in | 0.7530 ± 0.0048 |
| PostMean plug-in | 0.7494 ± 0.0052 |
| EMDecoder_plugin | 0.7343 ± 0.0038 |
| EMDecoder (original rule) | 0.6870 ± 0.0054 |
| LogOddsScalar (new, §5) | 0.6207 ± 0.0041 |

Definitions. **GridMAP**: per source, take the MAP cell (c, ρ, r) of the
paper's grid posterior and decide queries with `oracle_log_odds` at that
cell. **PostMean**: take the MAP convention c* (the convention of the
joint MAP cell); compute the posterior means ρ̄ = E[ρ | c*, support] and
r̄ = E[r | c*, support] under the grid posterior (conditional on c*);
decide with `oracle_log_odds` at (c*, ρ̄, r̄).

Paired deltas vs Bayes (30 seeds):

- Bayes − GridMAP = **+0.0060** [+0.0031, +0.0090], dz = +0.76.
  The cost of replacing posterior averaging by the posterior's *own*
  MAP point estimate is **0.6 points**, not the 2.6 points the paper's
  ablation narrative implies. (Consistent with the audit's 3-seed
  check: 0.747/0.747/0.769 vs Bayes 0.768/0.752/0.771 — my seeds 0–2
  reproduce those exactly.)
- Bayes − PostMean = +0.0096 [+0.0058, +0.0133], dz = +0.96.
  PostMean is *worse* than GridMAP (paired GridMAP − PostMean =
  +0.0036 [+0.0021, +0.0051]): plugging in conditional posterior means
  of (ρ, r) under the MAP convention does not beat the MAP cell.
- EMDecoder_plugin − EMDecoder = **+0.0473** [+0.0438, +0.0508],
  dz = +5.02. At 30 seeds the cost of the original rule's dropping ρ̂
  is **4.7 points** (banked 10-seed figure: 4.9). This confirms the
  audit's §C2: the 4.9-point component measures the wrong decision
  formula, *not* "committing to a single convention" — both rules use
  the same MAP convention.
- Bayes − EMDecoder_plugin = +0.0247 [+0.0216, +0.0278]: what remains
  of the Bayes–EM gap after the formula is fixed is EM's weaker fit
  (see recovery below), not a clean averaging effect either.

Convention recovery (5,400 sources, pooled = per-seed mean):

- Grid posterior MAP: **0.6398 ± 0.0138**
- EM MAP (`em_fit`): **0.3930 ± 0.0141**

The paper's "MAP recovery ≈ 0.39" was asserted about EM with no data
in the repo. It is **correct for EM** (0.393) — now backed by data —
but the grid posterior's own MAP recovers the true convention 64% of
the time, which is why the GridMAP plug-in (0.7530) sits far above
every EM variant. Any text that lets "0.39" stand next to the
posterior's performance conflates the two estimators.

Sanity checks against banked CSVs — all bit-exact unless flagged:

- My Bayes per-seed accuracies = `expB_n100.csv` Bayes, seeds 0–29:
  max abs diff **0.0**. Mean 0.7590 ✓ (paper's number).
- My EMDecoder / EMDecoder_plugin per-seed accuracies =
  `sens_emdecoder.csv`, seeds 0–9: max abs diff **0.0** (banked
  10-seed means 0.6829 / 0.7316 ✓). The 30-seed means (0.6870 /
  0.7343) differ from the banked ones only because the banked run
  stopped at 10 seeds.

## 5. LogOddsScalar (new baseline)

Definition (parameter-free, support-only): mirror SignedScalar's
decision form — per-source logit = w_s·(2m_q − 1), predict 1 iff
logit > 0 — but set the weight analytically from the source's support
agreement rate cons = mean(m_sup = z_sup) (the same `cons` quantity
the repo's scalar baselines compute): p = clip(cons, 0.05, 0.95),
w_s = log(p / (1 − p)). No learned global scale, no training.

Result: **0.6207 ± 0.0041** (30 seeds) — and it is *decision-identical*
to the banked trained SignedScalar on all 30 seeds (per-seed
accuracies equal, max diff 0.0): `sample_batch` consumes the RNG in
the same order as the per-source loop (identical streams), and with
the trained tanh(raw_w) > 0, SignedScalar's decision reduces to the
same sign rule (the clip never changes a sign). So the strongest
member of the signed scalar class the audit asked for — per-source
log-odds weighting — scores 0.6207, 13.8 points below Bayes
(paired +0.1383 [+0.1323, +0.1443], dz = +8.60). The scalar-class gap
is not an artefact of the weak learned parametrisation.

## 6. CtxLogReg K-sweep (λ = 0.1, 10 seeds, repo `eval_ctxlogreg`)

Runtime: 2.6 s. Per-seed values in `experiments.json`.

| K | Accuracy |
|---|---|
| 2 | 0.5342 ± 0.0061 |
| 5 | 0.5902 ± 0.0131 |
| 10 | 0.6367 ± 0.0077 |
| 20 | 0.6771 ± 0.0091 |
| 50 | 0.7128 ± 0.0071 |

**⚠ Flag — the paper's quoted "0.63 → 0.71" does not match its stated
K range.** At K = 50 the value is 0.7128 ✓ (~0.71), but at K = 2 the
accuracy is 0.5342, not 0.63. The 0.63 figure is the K = 10 value
(0.6367). The internal note's prose sweep (5 seeds, K ∈ {10, 30, 50}:
0.635 / 0.704 / 0.712) suggests the paper's sentence mislabels the
range: the data support "0.64 → 0.71 for K = 10 → 50", not
"0.63 → 0.71 for K = 2 → 50".

**⚠ Flag — one banked row does not reproduce.** At K = 10, 9 of 10
seeds bit-match `expB_ctxlogreg.csv` (λ = 0.1). Seed 5 differs by
9/5400 queries (mine 0.629259 vs banked 0.627593); a serial rerun on
this machine confirms the current code produces 0.629259, so the
banked seed-5 row was generated under a numerically different
environment (one source's IRLS fit flips a few borderline queries).
Effect on the 10-seed mean: +0.0002 (0.6367 vs banked 0.6365). No
headline number changes, but it is a second instance of the audit's
version-linkage problem: nothing ties the banked row to the code +
environment that produced it.

## 7. Parameter counts (instantiated in the repo venv, torch 2.14 CPU)

Configs exactly as `upgrade_n100.run_seed_B` builds them
(`make_model` defaults). Runtime < 1 s.

| Model | Config | Parameters |
|---|---|---|
| MetaAttn | d = 64, sup_in = 4, qry_in = 3 | 25,667 |
| MetaAttn+Dec | d = 64, sup_in = 5, qry_in = 4 | 25,795 |
| MetaAttn+DecCtx | d = 64, sup_in = 13, qry_in = 12 | 26,819 |
| TransformerMeta | d = 64, 4 heads, 2 layers, ff = 256, sup_in = 4, qry_in = 3 | **104,834** |
| DeepAttn (context) | d = 256, sup_in = 4, qry_in = 3 | 594,947 |

**⚠ Flag — "capacity-matched" is not true by parameter count.**
TransformerMeta has 104,834 parameters vs MetaAttn's 25,667 — a
4.1× difference (same width d = 64, but the transformer's attention +
feed-forward blocks dominate). If the paper's claim means "same
width", it should say so and print these counts; as a parameter-count
statement it is false. (For reference, the high-capacity arm DeepAttn
is 594,947 — 23× MetaAttn.)

## Bottom line for the fixes

- The "2.6 points from posterior averaging" claim should be replaced:
  the honest decomposition at 30 seeds is 4.7 points from dropping ρ̂
  (formula error) and **0.6 points** from MAP-plug-in vs averaging
  (Bayes − GridMAP), with a further ~1.9 points of the EM gap coming
  from EM's fit quality (recovery 0.39 vs 0.64), not from averaging
  per se.
- "MAP recovery ≈ 0.39" is true of EM only; the grid posterior's MAP
  recovery is 0.64. Both numbers now have data backing.
- The per-source log-odds scalar (strongest signed-scalar member)
  scores 0.6207 and is decision-identical to the trained
  SignedScalar — the scalar-class conclusion survives the stronger
  baseline.
- CtxLogReg's rise is real but starts at 0.534 (K = 2), not 0.63;
  the quoted range in the paper is mislabelled.
- TransformerMeta is not capacity-matched to MetaAttn by count
  (104,834 vs 25,667); print the counts.
