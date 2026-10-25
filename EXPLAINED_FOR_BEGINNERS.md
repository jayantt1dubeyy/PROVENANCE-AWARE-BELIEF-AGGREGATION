# PVI — Provenance-Aware Belief Aggregation, Explained for Beginners

This is a beginner explainer for this project, in plain language with one simple analogy. It is not a replacement for the technical paper or report.

**How to read the honesty labels:**

- **Verified** = the number or result is in a file Jayant actually produced, and it was checked here.
- **Reported, not verified** = the number comes from Jayant's own summary or slides. No seeds, no confidence intervals, and no test code were given with it. Treat it as "he says so", not "proven".
- **Plan only** = an idea or a target. It has not happened yet.

---


**In one line:** When many agents talk to each other, do not just ask "who do I trust?" Ask "what did they actually mean?"

**Analogy:** Imagine friends who all use the word "yes", but one friend always means the opposite. If you only count how reliable each friend is, you will still get the answer wrong. You first need their codebook.

**What Jayant actually did (verified):**

- He wrote a 17-page manuscript. It is a research paper draft, not a finished peer-reviewed publication.
- He proved two limits, inside the rules of his own model. The proofs say simple trust scores cannot fix a misread message.
- He built a Bayesian aggregator. That is a fancy way of saying: the system guesses each speaker's codebook from examples, then decodes the message.
- He tested it in simulations, over 190 seed runs in the headline tables.

**The real result (verified, with a limit):**

- In his tests, the Bayesian method beat simple trust-score methods. It matched or beat the learned calibrators he tested, in most cases.
- But it was not always the best. In one experiment, a learned calibrator beat it. That is stated in his own README, and it matters.
- He also tested real human-judgment data in a later section, called PVI-REAL. That is a step toward the real world. It is not the whole real world.

**What is NOT proved / still open:**

- The theorems are proved inside his model assumptions only. They do not prove the method works everywhere in real life.
- Real-world universality is NOT proved.
- The preprint upload to Zenodo is still an open step. The paper is not presented here as accepted or published.
