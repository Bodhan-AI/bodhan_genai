# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Turn ``lid_eval`` score vectors into the decisions we actually have to make.

Two things get decided here, and both are decided from evidence rather than
taste:

1. **What should ``allowed_langs`` default to?** Restricting the candidate set is
   a hard filter — audio outside the set is silently reassigned, never flagged —
   so a narrower default has to earn its place by raising accuracy on languages
   we do serve without collapsing ones we do. Each candidate policy is an argmax
   over a subset of the same saved vector, so all of them are compared here at
   zero extra compute.

2. **Is there a confidence below which we should refuse to guess?** The sweep
   reports, for each threshold, how much of the corpus is still answered and how
   accurate those answers are — an abstention curve, not a single number.

Also prints the confusion structure, with the near-neighbour pairs
(hi/ur, hi/bgc, hi/hne, gu/bhb, mr/bhb) called out, since those are where a
wrong label costs the most: the wrong language yields the wrong *script*.

Usage
-----
    python lid_report.py --scores 'lid_voi.shard*.jsonl' --title VOI
"""

from __future__ import annotations

import argparse
import collections
import glob
import json

TRAINED = (
    "as",
    "bgc",
    "bhb",
    "bho",
    "bn",
    "brx",
    "doi",
    "en",
    "gu",
    "hi",
    "hne",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
)
NEAR_PAIRS = [
    ("hi", "ur"),
    ("hi", "bgc"),
    ("hi", "hne"),
    ("gu", "bhb"),
    ("mr", "bhb"),
    ("bho", "mai"),
    ("hi", "bho"),
    ("sd", "ur"),
]


def load(patterns):
    rows, n_err = [], 0
    for pat in patterns:
        for path in sorted(glob.glob(pat)) or [pat]:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("error"):
                        n_err += 1
                    elif r.get("scores"):
                        rows.append(r)
    return rows, n_err


def predict(scores: dict, candidates: set[str] | None):
    """argmax over the candidate subset, plus its probability."""
    items = (
        scores.items()
        if candidates is None
        else [(k, v) for k, v in scores.items() if k in candidates]
    )
    if not items:
        return None, 0.0
    return max(items, key=lambda kv: kv[1])


def evaluate(rows, candidates, labelled_only=True):
    """(accuracy, n_scored, per-language {lang: (correct, total)})."""
    per = collections.defaultdict(lambda: [0, 0])
    correct = total = 0
    for r in rows:
        truth = r.get("true_lang")
        if not truth:
            continue
        if labelled_only and candidates is not None and truth not in candidates:
            # truth is outside the policy's reach: it CANNOT be got right. Counted
            # separately by the caller rather than silently inflating/deflating.
            per[truth][1] += 1
            total += 1
            continue
        pred, _ = predict(r["scores"], candidates)
        hit = pred == truth
        per[truth][0] += hit
        per[truth][1] += 1
        correct += hit
        total += 1
    return (correct / total if total else 0.0), total, per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="lid_eval JSONL (globs ok).")
    ap.add_argument("--title", default="")
    ap.add_argument("--top-confusions", type=int, default=25)
    ap.add_argument(
        "--per-lang-policy",
        default="trained-27",
        help="Which policy the per-language table compares against the unrestricted "
        "baseline. Set this to the policy the comparison table favoured.",
    )
    args = ap.parse_args()

    rows, n_err = load(args.scores)
    if not rows:
        raise SystemExit("no scored rows found")
    labelled = {r["true_lang"] for r in rows if r.get("true_lang")}
    n_lab = sum(1 for r in rows if r.get("true_lang"))
    vocab_n = rows[0].get("n_lang_tokens")

    print(f"=== LID report {args.title} ===")
    print(f"rows scored {len(rows)}   errors {n_err}   with ground truth {n_lab}")
    print(f"vocabulary language tokens: {vocab_n}   trained: {len(TRAINED)}")
    print(f"labels present in this corpus ({len(labelled)}): {' '.join(sorted(labelled))}")
    unserved = labelled - set(TRAINED)
    if unserved:
        print(f"  !! labels OUTSIDE the trained set: {sorted(unserved)}")

    # ---- 1. policy comparison -------------------------------------------------
    policies = {
        "vocab (no restriction)": None,
        "trained-27": set(TRAINED),
        "labelled-only": set(labelled),
        "trained-27 minus bgc,bhb": set(TRAINED) - {"bgc", "bhb"},
        "trained-27 minus bgc,bhb,hne": set(TRAINED) - {"bgc", "bhb", "hne"},
    }
    print()
    print("--- policy comparison (top-1 accuracy on rows with ground truth) ---")
    print(f"{'policy':32s} {'acc':>8s} {'n':>8s} {'excl.langs':>11s} {'excl.rows':>10s}")
    results = {}
    n_by_lang = collections.Counter(r["true_lang"] for r in rows if r.get("true_lang"))
    for name, cand in policies.items():
        acc, n, per = evaluate(rows, cand)
        gone = set() if cand is None else labelled - cand
        results[name] = (acc, n, per)
        print(f"{name:32s} {acc:8.4f} {n:8d} {len(gone):11d} {sum(n_by_lang[g] for g in gone):10d}")
    print()
    print("  excl.langs / excl.rows = languages the policy excludes and how many rows")
    print("  carry those labels. Those rows can never be correct, which is the cost a")
    print("  narrow policy has to beat. Read acc together with excl.rows: a policy that")
    print("  wins on acc while excluding real traffic has not actually won.")

    # ---- 2. per-language, best two policies ----------------------------------
    base = results["vocab (no restriction)"][2]
    if args.per_lang_policy not in results:
        raise SystemExit(f"--per-lang-policy must be one of: {sorted(results)}")
    narrow = results[args.per_lang_policy][2]
    print()
    print(f"--- per-language accuracy: unrestricted vs {args.per_lang_policy} ---")
    print(f"{'lang':6s} {'n':>7s} {'vocab':>8s} {'narrowed':>10s} {'delta':>8s}")
    for lang in sorted(set(base) | set(narrow)):
        cb, tb = base.get(lang, (0, 0))
        cn, tn = narrow.get(lang, (0, 0))
        ab = cb / tb if tb else 0.0
        an = cn / tn if tn else 0.0
        flag = "  <-- worse" if an < ab - 0.005 else ("  <-- better" if an > ab + 0.005 else "")
        print(f"{lang:6s} {tb:7d} {ab:8.4f} {an:10.4f} {an - ab:+8.4f}{flag}")

    # ---- 3. confusion --------------------------------------------------------
    conf = collections.Counter()
    for r in rows:
        truth = r.get("true_lang")
        if not truth:
            continue
        pred, _ = predict(r["scores"], None)
        if pred != truth:
            conf[(truth, pred)] += 1
    print()
    print(f"--- top {args.top_confusions} confusions (unrestricted) ---")
    for (t, p), n in conf.most_common(args.top_confusions):
        print(f"  {t:>5s} -> {p:<5s} {n:6d}")

    print()
    print("--- near-neighbour pairs (both directions) ---")
    for a, b in NEAR_PAIRS:
        ab, ba = conf.get((a, b), 0), conf.get((b, a), 0)
        na = base.get(a, (0, 0))[1]
        nb = base.get(b, (0, 0))[1]
        if not (na or nb):
            continue
        ra = f"{ab / na:.3f}" if na else "  -  "
        rb = f"{ba / nb:.3f}" if nb else "  -  "
        print(f"  {a}->{b}: {ab:5d}/{na:<6d} ({ra})    {b}->{a}: {ba:5d}/{nb:<6d} ({rb})")

    # ---- 4. abstention curve -------------------------------------------------
    print()
    print("--- confidence threshold sweep (unrestricted argmax) ---")
    print(
        f"{'thresh':>8s} {'answered':>10s} {'coverage':>10s} {'acc|answered':>13s} {'acc overall':>12s}"
    )
    scored = []
    for r in rows:
        truth = r.get("true_lang")
        if not truth:
            continue
        pred, prob = predict(r["scores"], None)
        scored.append((prob, pred == truth))
    scored.sort(reverse=True)
    total = len(scored)
    for th in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999):
        kept = [hit for prob, hit in scored if prob >= th]
        if not kept:
            print(f"{th:8.3f} {0:10d} {0.0:10.4f} {0.0:13.4f} {0.0:12.4f}")
            continue
        acc_ans = sum(kept) / len(kept)
        print(
            f"{th:8.3f} {len(kept):10d} {len(kept) / total:10.4f} "
            f"{acc_ans:13.4f} {sum(kept) / total:12.4f}"
        )
    print()
    print("  'acc overall' counts an abstention as a miss — the honest number if the")
    print("  caller has no fallback. Pick a threshold only if the two columns diverge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
