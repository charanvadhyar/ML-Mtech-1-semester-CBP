"""
Apply the DravidianCodeMix-trained classical model (TF-IDF+XGBoost) to our
OWN real scraped YouTube comments (comments.csv), and run the Language-
Segment SHAP Attribution using the REAL, entity-aware word tags our own
rulebasedextractor.py + scrapeextractclassifypipeline.py pipeline already
produced (tanglish_tagged.csv's english_words/tamil_words/tanglish_words
columns) -- not a re-derived approximation.

This is the actual "train on DravidianCodeMix, explain on real-world
Tanglish YouTube comments" interpretability demo for the review.
"""
import csv
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

import os as _os
def _find(*names):
    """Resolve a data file whether this runs on the user's machine (beside
    the script) or in the Claude sandbox (staged upload paths)."""
    _here = _os.path.dirname(_os.path.abspath(__file__))
    for n in names:
        for p in (_os.path.join(_here, n), n,
                  _os.path.join("/mnt/user-data/uploads", n),
                  _os.path.join("/mnt/user-data/uploads/MLSHAP", n)):
            if _os.path.exists(p):
                return p
    raise SystemExit("could not find any of %s" % (names,))


csv.field_size_limit(sys.maxsize)

TAMIL_RANGE = re.compile(r"[஀-௿]")

model = XGBClassifier()
model.load_model("classical_xgb_model.json")
vectorizer = joblib.load("classical_tfidf_vectorizer.joblib")
with open("classical_feature_names.json") as f:
    feature_names = json.load(f)

comments = pd.read_csv(_find("comments.csv"))
comments["text"] = comments["text"].fillna("")

tagged = pd.read_csv(_find("output/tanglish_tagged.csv", "tanglish_tagged.csv"), engine="python")
# tagged is comments.csv + extra columns, same row order/length
assert len(tagged) == len(comments) + 1 or len(tagged) == len(comments), \
    f"Row count mismatch: comments={len(comments)} tagged={len(tagged)}"
tagged = tagged.iloc[: len(comments)].reset_index(drop=True)

def build_word_tag_map(row):
    """From this comment's own pipeline output: which words were tagged EN / TA-script / TANGLISH?"""
    m = {}
    for col, tag in [("english_words", "EN"), ("tamil_words", "TA-script"), ("tanglish_words", "TANGLISH?")]:
        val = row.get(col, "")
        if pd.isna(val) or not str(val).strip():
            continue
        for w in str(val).split(", "):
            w = w.strip().lower()
            if w:
                m[w] = tag
    return m

print("Vectorizing + predicting sentiment on real scraped comments ...")
X_all = vectorizer.transform(comments["text"]).toarray()
preds = model.predict(X_all)
probs = model.predict_proba(X_all)[:, 1]
comments["predicted_sentiment"] = np.where(preds == 1, "Positive", "Negative")
comments["positive_prob"] = probs.round(3)

print(comments["predicted_sentiment"].value_counts())
print()

N_SHAP = 300
# comments.csv is grouped by video_id in blocks, so a stratified sample
# across all four videos is needed (not the first N rows, which would
# all come from a single movie).
per_video = max(1, N_SHAP // comments["video_id"].nunique())
sample_idx = (
    comments.groupby("video_id", group_keys=False)
    .apply(lambda g: g.sample(n=min(per_video, len(g)), random_state=42))
    .index
)
sample_idx = sorted(sample_idx)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_all[sample_idx])

rows = []
for pos, i in enumerate(sample_idx):
    present_idx = np.nonzero(X_all[i])[0]
    if len(present_idx) == 0:
        continue
    word_tag_map = build_word_tag_map(tagged.iloc[i])
    abs_shap = np.abs(shap_values[pos])
    total = abs_shap[present_idx].sum()
    if total == 0:
        continue

    tag_mass = {"EN": 0.0, "TA-script": 0.0, "TANGLISH?": 0.0, "UNTAGGED": 0.0}
    for idx in present_idx:
        term = feature_names[idx]
        if term in word_tag_map:
            # Preferred: the real per-comment tag from our own entity-aware
            # pipeline (distinguishes EN vs TANGLISH? correctly).
            tag = word_tag_map[term]
        elif TAMIL_RANGE.search(term):
            # sklearn's default tokenizer sometimes splits Tamil-script
            # words into different fragments than our own tokenizer did,
            # so an exact-string match to the pipeline's word list can
            # miss a term that is still unambiguously Tamil script.
            tag = "TA-script"
        else:
            tag = "UNTAGGED"
        tag_mass[tag] += abs_shap[idx]

    row = {
        "video_id": comments.iloc[i]["video_id"],
        "text": comments.iloc[i]["text"],
        "predicted_sentiment": comments.iloc[i]["predicted_sentiment"],
    }
    for tag, mass in tag_mass.items():
        row[f"{tag}_shap_pct"] = round(100 * mass / total, 2)
    rows.append(row)

per_instance = pd.DataFrame(rows)
per_instance.to_csv("scraped_comments_language_segment_shap.csv", index=False)
comments.to_csv("scraped_comments_with_predictions.csv", index=False)

pct_cols = [c for c in per_instance.columns if c.endswith("_shap_pct")]
print(f"Dataset-level trend on {len(per_instance)} REAL scraped YouTube comments")
print("(classical TF-IDF+XGBoost model, trained on DravidianCodeMix, explained with TreeSHAP,")
print("language tags from our own rule-based entity-aware pipeline):\n")
print(per_instance[pct_cols].mean().round(2).sort_values(ascending=False))

print("\nSplit by predicted sentiment:")
print(per_instance.groupby("predicted_sentiment")[pct_cols].mean().round(2))

print("\nSplit by video (does the model's language reliance differ by movie/fandom?):")
print(per_instance.groupby("video_id")[pct_cols].mean().round(2))

with open("scraped_language_segment_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_instances": len(per_instance),
        "overall_mean_pct": per_instance[pct_cols].mean().round(2).to_dict(),
        "by_predicted_sentiment": per_instance.groupby("predicted_sentiment")[pct_cols].mean().round(2).to_dict(),
        "by_video": per_instance.groupby("video_id")[pct_cols].mean().round(2).to_dict(),
    }, f, indent=2)

print("\nSaved: scraped_comments_language_segment_shap.csv, "
      "scraped_comments_with_predictions.csv, scraped_language_segment_summary.json")
