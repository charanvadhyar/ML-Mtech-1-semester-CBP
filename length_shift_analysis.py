"""
Does the length artifact predict real-world false accusations?

The benchmark audit found that in several languages the label is partly
readable from document length alone. This script tests the consequence:
if a model learned "longer means fake", then deploying it on text longer
than its training data should make it flag real news as fake, and the
size of that length shift should rank how badly.

For each language it computes
    shift_ratio = mean words in live feed text / mean words in training
and correlates that against the share of real live news the model flagged
as fake.

Reporting note: with n=5 languages, scipy's asymptotic Spearman p-value is
invalid (it prints 0.0000 for a perfect rank correlation). An exact
permutation p over all 120 orderings is computed instead.
"""
import json
import os
from itertools import permutations

import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))


def find(*names):
    for n in names:
        for p in (os.path.join(HERE, n), n,
                  os.path.join("/mnt/user-data/uploads/MLSHAP", n)):
            if os.path.exists(p):
                return p
    raise SystemExit(f"could not find any of {names}")


def exact_spearman_p(x, y):
    """Two-tailed exact permutation p-value. Valid at tiny n, unlike the
    asymptotic approximation scipy reports."""
    obs = abs(spearmanr(x, y).statistic)
    stats = [abs(spearmanr(x, list(p)).statistic) for p in permutations(y)]
    hits = sum(1 for s in stats if s >= obs - 1e-12)
    return hits / len(stats), hits, len(stats)


def main():
    train = pd.read_csv(find("multilingual_fakenews.csv"))
    train["text"] = train["text"].astype(str)
    feed = pd.read_csv(find("feed_articles.csv"))
    feed["text"] = feed["text"].astype(str)
    summ = pd.read_csv(find("feed_leakage_summary.csv"))
    audit = pd.read_csv(find("benchmark_audit.csv"))

    rows = []
    for lang in sorted(summ["lang"]):
        tw = train[train["lang"] == lang]["text"].str.split().str.len().mean()
        fw = feed[feed["lang"] == lang]["text"].str.split().str.len().mean()
        a = audit[audit["language"] == lang].iloc[0]
        rows.append({
            "lang": lang,
            "train_words": round(float(tw), 1),
            "feed_words": round(float(fw), 1),
            "shift_ratio": round(float(fw / tw), 2),
            "length_only_acc": float(a["length_only_accuracy"]),
            "longer_means_fake": bool(a["mean_chars_fake"] > a["mean_chars_real"]),
            "pct_real_news_flagged_fake": float(
                summ[summ["lang"] == lang]["overall_pct_flagged_fake"].iloc[0]),
        })

    df = pd.DataFrame(rows).sort_values("shift_ratio", ascending=False)
    print(df.to_string(index=False))

    x = df["shift_ratio"].tolist()
    y = df["pct_real_news_flagged_fake"].tolist()
    pr, pr_p = pearsonr(x, y)
    sr = spearmanr(x, y).statistic
    p_exact, hits, total = exact_spearman_p(x, y)

    print(f"\nPearson  r   = {pr:.3f}  (p = {pr_p:.4f})")
    print(f"Spearman rho = {sr:.3f}")
    print(f"  exact two-tailed permutation p = {p_exact:.4f} "
          f"({hits} of {total} orderings)")
    print("  scipy's asymptotic p is not valid at n=5 -- quote the exact value.")
    print(f"\nn = {len(df)} languages. Suggestive, not established: the languages"
          "\ndiffer in many ways besides length shift. What supports it is the"
          "\nmechanism, and that the static length-only probe in the audit"
          "\npredicted this before any live data was collected.")

    df.to_csv(os.path.join(HERE, "length_shift_vs_falseflag.csv"), index=False)
    with open(os.path.join(HERE, "length_shift_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "n_languages": int(len(df)),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pr_p), 4),
            "spearman_rho": round(float(sr), 4),
            "spearman_p_exact_two_tailed": round(float(p_exact), 4),
            "permutation_orderings": total,
            "note": ("scipy's asymptotic Spearman p is invalid at n=5; the exact "
                     "permutation value is reported instead."),
            "per_language": rows,
        }, f, indent=2)
    print("\nSaved length_shift_vs_falseflag.csv, length_shift_stats.json")


if __name__ == "__main__":
    main()
