"""Real-world validation: aggregating genuine human judgments (CIFAR-10H).

AUDIT-FIX VARIANT (code/audit_fixes/real_world_cifar10h_ds.py): identical to
code/real_world_cifar10h.py except that predict_dawid_skene below is a
GENUINE Dawid-Skene EM. The original computed the label posterior once from
vote fractions and never re-ran the E-step, so its "DawidSkene" column was a
one-shot plug-in estimate, not EM. Outputs go to results/audit_fixes/.

CIFAR-10H (Peterson, Battleday, Griffiths & Russakovsky, ICCV 2019) contains
~50 human category judgments for each of the 10,000 CIFAR-10 test images,
with anonymised annotator IDs linking workers across images.

Design: split images 50/50 (seeded) into an estimation set and an
evaluation set. Per-worker confusion behaviour is estimated ONLY on the
estimation set (with the CIFAR-10 gold labels), then each evaluation image's
annotations are aggregated to predict its gold label. This tests the
reliability-heterogeneity half of the thesis -- model the source, don't
average messages -- on genuine human judgments.

Methods
  MajorityVote    : argmax of raw vote counts (the real-world default).
  AccuracyWeighted: votes weighted by each worker's estimation-set accuracy
                    (scalar reliability weighting).
  BayesDirichlet  : per-image posterior over classes from per-worker
                    confusion matrices (Dirichlet/Laplace smoothed) fixed
                    on the estimation set; argmax of the posterior.
  DawidSkene      : classic Dawid-Skene EM over the evaluation images,
                    confusion matrices warm-started from the estimation
                    set (estimation-set counts act as pseudocounts in
                    every M-step). FIXED: the posterior is recomputed
                    from the current confusion matrices and the current
                    class-prevalence estimate at every iteration (true
                    E-step), iterating to convergence.
  Oracle          : BayesDirichlet with confusion matrices estimated from ALL
                    data including the evaluation image (upper bound).

Outputs: results/audit_fixes/real_world_cifar10h.csv (per-image, per-method
correctness) and results/audit_fixes/real_world_cifar10h_5judg.csv (same
for the 5-judgment regime, subsample seed 1), plus a printed summary table
with 95% CIs and paired deltas vs MajorityVote.

The raw file is data/cifar10h-raw.zip (extracted to cifar10h-raw.csv) from
https://github.com/jcpeterson/cifar-10h. Point CIFAR10H_CSV at your copy,
or place it at data/cifar10h-raw.csv in this repo.
"""

import csv
import os
import numpy as np

SEED = 0
N_CLASSES = 10
SMOOTH = 1.0  # Laplace / Dirichlet prior pseudocount
DS_ITERS = 100  # audit fix: genuine EM gets the full iteration budget
DS_TOL = 1e-6   # convergence: max abs change in the confusion matrices

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # code/audit_fixes -> repo root
OUT = os.path.join(REPO, "results", "audit_fixes", "real_world_cifar10h.csv")
OUT5 = os.path.join(REPO, "results", "audit_fixes",
                    "real_world_cifar10h_5judg.csv")


def resolve_raw():
    """Locate cifar10h-raw.csv: CIFAR10H_CSV env var, then repo data/, then
    the legacy /tmp location. The file is data/cifar10h-raw.zip extracted
    from https://github.com/jcpeterson/cifar-10h."""
    candidates = [c for c in [
        os.environ.get("CIFAR10H_CSV"),
        os.path.join(REPO, "data", "cifar10h-raw.csv"),
        "/tmp/cifar10h/cifar10h-raw.csv",
    ] if c]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise SystemExit(
        "cifar10h-raw.csv not found. Download data/cifar10h-raw.zip from "
        "https://github.com/jcpeterson/cifar-10h, extract cifar10h-raw.csv, "
        "then set CIFAR10H_CSV=/path/to/cifar10h-raw.csv or place it at "
        "data/cifar10h-raw.csv in the repo. Checked: "
        + ", ".join(candidates))


RAW = resolve_raw()


def load():
    img_ids, workers, gold, chosen = [], [], [], []
    with open(RAW, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["is_attn_check"]) == 1:
                continue
            img_ids.append(int(row["cifar10_test_test_idx"]))
            workers.append(int(row["annotator_id"]))
            gold.append(int(row["true_label"]))
            chosen.append(int(row["chosen_label"]))
    img_ids = np.array(img_ids)
    workers = np.array(workers)
    gold = np.array(gold)
    chosen = np.array(chosen)
    assert not (np.isnan(img_ids).any() or np.isnan(workers).any()
                or np.isnan(gold).any() or np.isnan(chosen).any())
    assert (img_ids >= 0).all()
    return img_ids, workers, gold, chosen


def main():
    rng = np.random.default_rng(SEED)
    img_ids, workers, gold, chosen = load()

    images = np.unique(img_ids)
    n_workers = int(workers.max()) + 1
    print(f"images={len(images)} annotators~{n_workers} "
          f"judgments={len(img_ids)}")

    # --- sanity checks ---
    per_worker = np.bincount(workers)
    per_image = np.bincount(img_ids)
    print(f"annotations/worker: median={np.median(per_worker):.0f} "
          f"min={per_worker.min()} max={per_worker.max()}")
    print(f"annotations/image: median={np.median(per_image):.0f} "
          f"min={per_image.min()} max={per_image.max()}")
    gold_per_image = np.array([gold[img_ids == i][0] for i in images])
    hist = np.bincount(gold_per_image, minlength=N_CLASSES)
    print(f"gold label distribution: {hist}")
    print(f"mean worker accuracy (raw): "
          f"{(chosen == gold).mean():.4f}")

    # --- split images into estimation / evaluation ---
    perm = rng.permutation(images)
    est_images = set(perm[: len(perm) // 2].tolist())
    eval_images = perm[len(perm) // 2:]
    est_mask = np.array([i in est_images for i in img_ids])
    print(f"estimation images={est_mask.sum()} "
          f"evaluation images={len(eval_images)}")

    # --- per-worker confusion matrices on the estimation set ---
    # conf[w, true, chosen]; conf_counts keeps the raw (smoothed) counts so
    # Dawid-Skene can use them as pseudocounts in every M-step
    conf_counts = np.full((n_workers, N_CLASSES, N_CLASSES), SMOOTH)
    for w in range(n_workers):
        m = est_mask & (workers == w)
        if m.any():
            np.add.at(conf_counts[w], (gold[m], chosen[m]), 1)
    conf = conf_counts / conf_counts.sum(axis=2, keepdims=True)
    worker_acc = np.array([
        (chosen[est_mask & (workers == w)] == gold[est_mask & (workers == w)]).mean()
        if (est_mask & (workers == w)).any() else np.nan
        for w in range(n_workers)
    ])
    n_fallback = int(np.isnan(worker_acc).sum())
    print(f"workers with no estimation data (fallback to uniform): "
          f"{n_fallback}")
    # fallback: uniform confusion, mean accuracy
    mean_acc = np.nanmean(worker_acc)
    worker_acc = np.where(np.isnan(worker_acc), mean_acc, worker_acc)
    conf[np.isnan(conf).any(axis=(1, 2))] = 1.0 / N_CLASSES

    # --- oracle confusion from ALL data (upper bound) ---
    conf_all = np.full((n_workers, N_CLASSES, N_CLASSES), SMOOTH)
    for w in range(n_workers):
        m = workers == w
        np.add.at(conf_all[w], (gold[m], chosen[m]), 1)
    conf_all = conf_all / conf_all.sum(axis=2, keepdims=True)

    # --- per-evaluation-image annotation lists ---
    eval_ann = []  # list of (gold_label, [(worker, chosen), ...])
    for i in eval_images:
        m = img_ids == i
        eval_ann.append((int(gold[m][0]),
                         list(zip(workers[m].tolist(),
                                  chosen[m].tolist()))))
    tie_count = 0
    for g, ann in eval_ann:
        counts = np.bincount([c for _, c in ann], minlength=N_CLASSES)
        if (counts == counts.max()).sum() > 1:
            tie_count += 1
    print(f"majority-vote tie rate on eval images: "
          f"{tie_count}/{len(eval_ann)} = {tie_count/len(eval_ann):.4f}")

    def predict_majority(ann):
        counts = np.bincount([c for _, c in ann], minlength=N_CLASSES)
        return int(np.argmax(counts))  # ties -> smallest label, deterministic

    def predict_weighted(ann):
        s = np.zeros(N_CLASSES)
        for w, c in ann:
            s[c] += worker_acc[w]
        return int(np.argmax(s))

    def predict_bayes(ann, cmat, prior=None):
        if prior is None:
            prior = np.full(N_CLASSES, 1.0 / N_CLASSES)
        logp = np.log(prior + 1e-300)
        for w, c in ann:
            logp = logp + np.log(cmat[w, :, c] + 1e-300)
        return int(np.argmax(logp))

    def predict_dawid_skene(all_ann, counts_init):
        # GENUINE Dawid-Skene EM over all evaluation images (audit fix).
        # The original version computed the posterior ONCE from vote
        # fractions and never updated it, so its loop only re-derived the
        # same confusion matrices from a frozen posterior -- that is a
        # one-shot plug-in estimator, not EM. Here every iteration runs
        # both steps:
        #   M-step: confusion matrices from the current posterior, with
        #           the estimation-set counts kept as pseudocounts
        #           (identical warm start to the original).
        #   E-step: each image's label posterior recomputed from the
        #           current confusion matrices and the class prevalence
        #           estimated from the current posterior.
        # Iterate until the max abs change in the confusion matrices is
        # < DS_TOL (max DS_ITERS iterations). Predictions are the argmax
        # of the final posterior.
        n = len(all_ann)
        # flatten the annotation lists once for the vectorised steps
        ann_img, ann_w, ann_c = [], [], []
        for i, (g, ann) in enumerate(all_ann):
            for w, c in ann:
                ann_img.append(i)
                ann_w.append(w)
                ann_c.append(c)
        ann_img = np.array(ann_img)
        ann_w = np.array(ann_w)
        ann_c = np.array(ann_c)
        # initialise the posterior from vote fractions (as the original)
        post = np.zeros((n, N_CLASSES))
        for i, (g, ann) in enumerate(all_ann):
            counts = np.bincount([c for _, c in ann], minlength=N_CLASSES)
            post[i] = counts / max(counts.sum(), 1)
        prevalence = post.mean(axis=0)
        prevalence = prevalence / prevalence.sum()
        cmat = None
        for it in range(DS_ITERS):
            # --- M-step (same pseudocount warm start as the original) ---
            new_cmat = counts_init.copy()
            # new_cmat[w, :, c] += post[i] for each annotation (i, w, c):
            # add posterior rows into the (worker, chosen) slices
            np.add.at(new_cmat.transpose(0, 2, 1), (ann_w, ann_c),
                      post[ann_img])
            new_cmat = new_cmat / new_cmat.sum(axis=2, keepdims=True)
            # --- E-step: posterior from current cmat + prevalence ---
            log_cmat = np.log(new_cmat)
            logpost = np.tile(np.log(prevalence + 1e-300), (n, 1))
            np.add.at(logpost, ann_img, log_cmat[ann_w, :, ann_c])
            logpost = logpost - logpost.max(axis=1, keepdims=True)
            post = np.exp(logpost)
            post = post / post.sum(axis=1, keepdims=True)
            prevalence = post.mean(axis=0)
            prevalence = prevalence / prevalence.sum()
            # --- convergence on the confusion matrices ---
            delta = (np.abs(new_cmat - cmat).max() if cmat is not None
                     else np.inf)
            cmat = new_cmat
            if delta < DS_TOL:
                print(f"DawidSkene EM: converged after {it + 1} iterations "
                      f"(max |dcmat| = {delta:.2e})")
                break
        else:
            print(f"DawidSkene EM: hit iteration cap {DS_ITERS} "
                  f"(max |dcmat| = {delta:.2e})")
        return [int(np.argmax(post[i])) for i in range(n)]

    methods = ["MajorityVote", "AccuracyWeighted", "BayesDirichlet",
               "DawidSkene", "Oracle"]
    correct = {m: np.zeros(len(eval_ann), dtype=int) for m in methods}
    ds_preds = predict_dawid_skene(eval_ann, conf_counts)
    for i, (g, ann) in enumerate(eval_ann):
        correct["MajorityVote"][i] = predict_majority(ann) == g
        correct["AccuracyWeighted"][i] = predict_weighted(ann) == g
        correct["BayesDirichlet"][i] = predict_bayes(ann, conf) == g
        correct["DawidSkene"][i] = ds_preds[i] == g
        correct["Oracle"][i] = predict_bayes(ann, conf_all) == g

    # --- save per-image correctness ---
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_idx", "gold"] + methods)
        for i, (g, ann) in enumerate(eval_ann):
            w.writerow([int(eval_images[i]), g] +
                       [int(correct[m][i]) for m in methods])
    print(f"wrote {OUT}")

    # --- summary with 95% CIs and paired deltas vs MajorityVote ---
    def report(correct, n, title):
        tcrit = 1.960  # n = 5000, normal approx.
        base = correct["MajorityVote"].astype(float)
        print(f"\n{title} (n={n})")
        print(f"method            acc ± 95% CI        Δ vs MV ± 95% CI   dz")
        for m in methods:
            x = correct[m].astype(float)
            acc, half = x.mean(), tcrit * x.std(ddof=1) / np.sqrt(n)
            d = x - base
            dh = tcrit * d.std(ddof=1) / np.sqrt(n)
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
            print(f"{m:16s} {acc:.4f} ± {half:.4f}    "
                  f"{d.mean():+.4f} ± {dh:.4f}  {dz:+.2f}")

    n = len(eval_ann)
    report(correct, n, "all ~51 judgments/image")

    # --- low-annotation regime: 5 random judgments per image ---
    # (typical crowdsourcing budgets are 3-5 labels/item, not 51)
    rng5 = np.random.default_rng(1)
    eval_ann5 = []
    for g, ann in eval_ann:
        idx = rng5.choice(len(ann), size=min(5, len(ann)), replace=False)
        eval_ann5.append((g, [ann[j] for j in idx]))
    correct5 = {m: np.zeros(len(eval_ann5), dtype=int) for m in methods}
    ds5 = predict_dawid_skene(eval_ann5, conf_counts)
    for i, (g, ann) in enumerate(eval_ann5):
        correct5["MajorityVote"][i] = predict_majority(ann) == g
        correct5["AccuracyWeighted"][i] = predict_weighted(ann) == g
        correct5["BayesDirichlet"][i] = predict_bayes(ann, conf) == g
        correct5["DawidSkene"][i] = ds5[i] == g
        correct5["Oracle"][i] = predict_bayes(ann, conf_all) == g
    report(correct5, len(eval_ann5), "5 judgments/image")

    # --- save the 5-judgment per-image correctness (deterministic, seed 1) ---
    with open(OUT5, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_idx", "gold"] + methods)
        for i, (g, ann) in enumerate(eval_ann5):
            w.writerow([int(eval_images[i]), g] +
                       [int(correct5[m][i]) for m in methods])
    print(f"wrote {OUT5}")


if __name__ == "__main__":
    main()
