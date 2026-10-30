"""
Does benchmark leakage cause real-world misclassification?

The audit showed the Hindi and Telugu 'fake' classes are fact-CHECK
articles about false claims, so those models are fact-check-genre
detectors. If that is true, there is a concrete consequence: ordinary,
perfectly truthful news should get flagged as fake simply for *mentioning*
a viral claim, because that is the vocabulary of the genre the model
actually learned.

This tests that on live news. For each language, every pulled article is
scored by that language's own model, then split by whether it contains
fact-check vocabulary, and the predicted-fake rates of the two groups are
compared.

Stated in advance so it is falsifiable:
  - Hindi and Telugu should show a LARGE gap (marker articles flagged fake
    far more often), because their marker gap in training was +71 and +83
    points.
  - Malayalam should show LITTLE or NO gap -- its training marker gap was
    +0.2 points, so it is the control.

A caveat that belongs with the numbers: RSS gives headline-length text,
while the Hindi and Telugu models were trained on 1200-character article
leads. Those two therefore run out of domain here, so their absolute
fake-rates are not meaningful. The comparison still holds because both
groups within a language are equally out of domain.
"""
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

def find_feed_csv():
    """Resolve feed_articles.csv whether this runs on the user's machine
    (beside the script) or in the Claude sandbox (staged upload path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "feed_articles.csv"),
              "feed_articles.csv",
              "/mnt/user-data/uploads/MLSHAP/feed_articles.csv"):
        if os.path.exists(p):
            return p
    raise SystemExit("feed_articles.csv not found -- run pull_feeds.py first.")


FEED_CSV = find_feed_csv()

# Same marker sets as benchmark_audit.py, so the two are directly comparable.
MARKERS = {
    "hindi": ["बूम", "वायरल", "दावा", "दावे", "दावों", "फर्जी", "फैक्ट", "चेक",
              "कैप्शन", "स्रोत", "गलत", "भ्रामक", "शेयर", "सोशल", "मीडिया"],
    "telugu": ["బూమ్", "వైరల్", "ఫ్యాక్ట్", "చెక్", "సోషల్", "మీడియా",
               "వీడియో", "ఫోటో", "క్లెయిమ్", "షేర్", "తప్పు", "నిజం"],
    "malayalam": ["വൈറൽ", "ഫാക്ട്", "വ്യാജ", "പ്രചരി", "സോഷ്യൽ", "മീഡിയ"],
    "english": ["fact check", "fact-check", "viral", "claim", "misleading",
                "debunk", "rumour", "rumor", "false"],
    "tamil": ["வைரல்", "உண்மை", "போலி", "தவறான", "சமூக"],
}


def load_model(lang):
    mp, vp = f"fakenews_{lang}_model.json", f"fakenews_{lang}_vectorizer.joblib"
    if not (os.path.exists(mp) and os.path.exists(vp)):
        return None, None
    m = XGBClassifier()
    m.load_model(mp)
    return m, joblib.load(vp)


def main():
    feed = pd.read_csv(FEED_CSV)
    feed["text"] = feed["text"].astype(str)
    rows, per_article, examples = [], [], []

    for lang in sorted(feed["lang"].unique()):
        sub = feed[feed["lang"] == lang].copy()
        model, vec = load_model(lang)
        if model is None:
            print(f"=== {lang}: no trained model, skipping")
            continue

        X = vec.transform(sub["text"]).toarray()
        prob = model.predict_proba(X)[:, 1]
        sub["p_fake"] = prob
        sub["pred_fake"] = prob >= 0.5

        markers = MARKERS.get(lang, [])
        pat = "|".join(re.escape(m) for m in markers)
        sub["has_marker"] = sub["text"].str.lower().str.contains(pat.lower(), regex=True)

        g1 = sub[sub["has_marker"]]
        g0 = sub[~sub["has_marker"]]
        r1 = float(g1["pred_fake"].mean() * 100) if len(g1) else float("nan")
        r0 = float(g0["pred_fake"].mean() * 100) if len(g0) else float("nan")

        print(f"=== {lang}  ({len(sub)} live articles)")
        print(f"    mentions fact-check vocab : {len(g1):>4} articles -> "
              f"{r1:5.1f}% flagged fake" if len(g1) else
              f"    mentions fact-check vocab : none found")
        print(f"    does not mention it       : {len(g0):>4} articles -> "
              f"{r0:5.1f}% flagged fake")
        if len(g1) and len(g0):
            print(f"    GAP: {r1 - r0:+.1f} points")
        print(f"    overall flagged fake      : {sub['pred_fake'].mean() * 100:.1f}%"
              f"   mean P(fake) {prob.mean():.3f}")
        print()

        rows.append({
            "lang": lang,
            "n_articles": int(len(sub)),
            "n_with_marker": int(len(g1)),
            "pct_flagged_fake_with_marker": round(r1, 1) if len(g1) else None,
            "pct_flagged_fake_without": round(r0, 1) if len(g0) else None,
            "gap_points": round(r1 - r0, 1) if len(g1) and len(g0) else None,
            "overall_pct_flagged_fake": round(float(sub["pred_fake"].mean() * 100), 1),
            "mean_p_fake": round(float(prob.mean()), 3),
        })

        # Keep the most confidently-flagged real articles as concrete examples,
        # with the tokens that drove each decision.
        names = list(vec.get_feature_names_out())
        sv = shap.TreeExplainer(model).shap_values(X)
        order = np.argsort(prob)[::-1][:5]
        for i in order:
            contrib = sv[i]
            top = np.argsort(np.abs(contrib))[::-1][:6]
            examples.append({
                "lang": lang,
                "p_fake": round(float(prob[i]), 3),
                "has_marker": bool(sub.iloc[i]["has_marker"]),
                "title": str(sub.iloc[i]["title"])[:160],
                "source": sub.iloc[i]["source"],
                "top_tokens": ", ".join(
                    f"{names[j]}({contrib[j]:+.3f})" for j in top if X[i, j] != 0),
            })

        keep = sub[["lang", "source", "title", "url", "p_fake", "pred_fake",
                    "has_marker"]].copy()
        per_article.append(keep)

    if not rows:
        print("No languages could be scored.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv("feed_leakage_summary.csv", index=False)
    pd.concat(per_article, ignore_index=True).to_csv("feed_scored_articles.csv", index=False)
    ex = pd.DataFrame(examples)
    ex.to_csv("feed_flagged_examples.csv", index=False)
    with open("feed_leakage_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("================ SUMMARY ================")
    print(summary.to_string(index=False))
    print("\nMost confidently flagged live articles (all are real news):")
    print(ex[["lang", "p_fake", "has_marker", "title"]].to_string(index=False))
    print("\nSaved feed_leakage_summary.csv/.json, feed_scored_articles.csv,"
          " feed_flagged_examples.csv")


if __name__ == "__main__":
    main()
