"""
Leakage ablation for the Hindi and Telugu fake news data.

Those two models score ~99%, which is not credible for fake news
detection. SHAP said why: the top features are debunking vocabulary
(viral / claim / fake / caption / source) and `BOOM`, the brand name of
an Indian fact-checking outlet. The "fake" class appears to be
fact-CHECK articles *about* false claims, not false articles -- so the
classifier is a fact-check-genre detector.

Marker prevalence confirms it:
    Hindi   97.0% of 'fake' vs 16.0% of 'real' contain a marker
    Telugu  93.1% of 'fake' vs  5.6% of 'real'

This script tests causation rather than asserting it, by retraining with
the markers stripped from the text and their tokens barred from the
vocabulary. If accuracy collapses, the score was the artifact. If it only
dips, the leakage is a whole document genre rather than a few words --
which is a stronger indictment of the dataset, not a weaker one.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TOKEN_PATTERN = r"(?u)[\wऀ-ॿ஀-௿ఀ-౿ഀ-ൿ]{2,}"

MARKERS = {
    "hindi": ["बूम", "वायरल", "दावा", "दावे", "दावों", "फर्जी", "फैक्ट", "चेक",
              "कैप्शन", "स्रोत", "गलत", "भ्रामक", "वीडियो", "तस्वीर", "शेयर",
              "सोशल", "मीडिया", "वायरस"],
    "telugu": ["బూమ్", "వైరల్", "ఫ్యాక్ట్", "చెక్", "సోషల్", "మీడియా",
               "వీడియో", "ఫోటో", "క్లెయిమ్", "షేర్", "తప్పు", "నిజం"],
}


def run(lang, df, markers):
    strip = re.compile("|".join(re.escape(m) for m in markers))
    out = {}
    for setting in ["original", "markers_stripped"]:
        d = df.copy()
        if setting == "markers_stripped":
            d["text"] = d["text"].str.replace(strip, " ", regex=True)
        tr, te = train_test_split(d, test_size=0.2, random_state=42, stratify=d["y"])
        vec = TfidfVectorizer(max_features=1500, min_df=2, sublinear_tf=True,
                              token_pattern=TOKEN_PATTERN)
        X = vec.fit_transform(tr["text"]).toarray()
        m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                          subsample=0.9, colsample_bytree=0.9,
                          eval_metric="logloss", random_state=42)
        m.fit(X, tr["y"].values)
        acc = accuracy_score(te["y"], m.predict(vec.transform(te["text"]).toarray()))
        out[setting] = round(float(acc), 4)
        print(f"    {setting:18s} accuracy {acc:.4f}")
    out["drop"] = round(out["original"] - out["markers_stripped"], 4)
    print(f"    drop from stripping markers: {out['drop']:+.4f}")
    return out


def main():
    data = pd.read_csv("multilingual_fakenews.csv")
    data["text"] = data["text"].astype(str)
    results = {}
    for lang, markers in MARKERS.items():
        d = data[data["lang"] == lang]
        if d.empty:
            continue
        print(f"=== {lang}  ({len(d)} rows, {len(markers)} markers stripped)")
        results[lang] = run(lang, d, markers)
        print()

    with open("leakage_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).T.to_csv("leakage_ablation.csv")
    print("Saved leakage_ablation.json / .csv")


if __name__ == "__main__":
    main()
