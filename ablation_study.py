"""
Causal ablation study (not just attribution): mask one token category at a
time in each real scraped comment, re-run the DravidianCodeMix-trained
classical model, and measure how much the prediction actually moves.

Two categories of interest, chosen over plain SHAP attribution because
they turn a correlational claim ("the model attends to X") into a causal
one ("removing X changes the output by Y"):

  1. Language-ablation: mask EN / TA-script / TANGLISH? tokens in turn,
     using the real per-comment tags from our own entity-aware pipeline.
  2. Entity-leakage ablation: mask the real extracted named entities
     (actor/movie/crew names from entity_exclusion_list.json) and see
     whether the model -- which has never seen these specific movies --
     is nonetheless leaning on those proper nouns.

Metric per category: mean |delta P(Positive)| and the %% of comments whose
*predicted class* actually flips, over only the comments that contain at
least one word in that category (masking a category a comment doesn't
have is a no-op, not evidence of robustness).
"""
import csv
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
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

model = XGBClassifier()
model.load_model("classical_xgb_model.json")
vectorizer = joblib.load("classical_tfidf_vectorizer.joblib")

comments = pd.read_csv(_find("comments.csv"))
comments["text"] = comments["text"].fillna("")

tagged = pd.read_csv(_find("output/tanglish_tagged.csv", "tanglish_tagged.csv"), engine="python")
tagged = tagged.iloc[: len(comments)].reset_index(drop=True)

with open(_find("entity_exclusion_list.json")) as f:
    entity_data = json.load(f)
ENTITY_WORDS = set(entity_data["combined_exclusion_list"])
# also split multi-word entities into their component words for whole-word masking
for e in list(ENTITY_WORDS):
    for w in e.split():
        if len(w) > 2:
            ENTITY_WORDS.add(w.lower())


def mask_words(text, words_to_mask):
    """Remove whole-word (case-insensitive) occurrences of any word in
    words_to_mask from text. Returns (masked_text, n_masked)."""
    if not words_to_mask:
        return text, 0
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in sorted(words_to_mask, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    n_masked = len(pattern.findall(text))
    masked = pattern.sub(" ", text)
    return masked, n_masked


def get_word_list(row, col):
    val = row.get(col, "")
    if pd.isna(val) or not str(val).strip():
        return set()
    return {w.strip().lower() for w in str(val).split(", ") if w.strip()}


N = 500
sample = comments.sample(n=min(N, len(comments)), random_state=42)

categories = {
    "mask_EN": lambda i: get_word_list(tagged.iloc[i], "english_words"),
    "mask_TA-script": lambda i: get_word_list(tagged.iloc[i], "tamil_words"),
    "mask_TANGLISH": lambda i: get_word_list(tagged.iloc[i], "tanglish_words"),
    "mask_ENTITY": lambda i: ENTITY_WORDS,  # checked against this comment's own text below
}

orig_probs = model.predict_proba(vectorizer.transform(sample["text"]).toarray())[:, 1]
orig_preds = orig_probs >= 0.5

results = {cat: {"deltas": [], "flips": 0, "n_affected": 0} for cat in categories}

for cat_name, get_words in categories.items():
    texts_masked = []
    affected_mask = []
    for pos, (i, row) in enumerate(sample.iterrows()):
        words = get_words(i) if cat_name != "mask_ENTITY" else ENTITY_WORDS
        masked_text, n_masked = mask_words(row["text"], words)
        texts_masked.append(masked_text)
        affected_mask.append(n_masked > 0)
    affected_mask = np.array(affected_mask)

    masked_probs = model.predict_proba(vectorizer.transform(texts_masked).toarray())[:, 1]
    masked_preds = masked_probs >= 0.5

    deltas = np.abs(masked_probs[affected_mask] - orig_probs[affected_mask])
    flips = (masked_preds[affected_mask] != orig_preds[affected_mask]).sum()

    results[cat_name]["deltas"] = deltas.tolist()
    results[cat_name]["flips"] = int(flips)
    results[cat_name]["n_affected"] = int(affected_mask.sum())

print(f"Ablation study on {len(sample)} real scraped comments "
      f"(classical TF-IDF+XGBoost model, trained on DravidianCodeMix):\n")
summary_rows = []
for cat_name, r in results.items():
    deltas = np.array(r["deltas"])
    n_aff = r["n_affected"]
    mean_delta = deltas.mean() if n_aff > 0 else 0.0
    flip_rate = 100 * r["flips"] / n_aff if n_aff > 0 else 0.0
    summary_rows.append({
        "category": cat_name,
        "n_comments_affected": n_aff,
        "mean_abs_delta_P": round(float(mean_delta), 4),
        "pct_predictions_flipped": round(flip_rate, 2),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("mean_abs_delta_P", ascending=False)
print(summary_df.to_string(index=False))

summary_df.to_csv("ablation_study_summary.csv", index=False)
with open("ablation_study_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_rows, f, indent=2)

print("\nSaved: ablation_study_summary.csv, ablation_study_summary.json")
print(
    "\nInterpretation guide: a HIGH mean_abs_delta_P / flip-rate for a "
    "category means the model's decision causally depends on those "
    "tokens being present -- not just correlates with them. A category "
    "with near-zero shift despite many comments containing it (e.g. "
    "entities, if the DravidianCodeMix vocabulary barely overlaps with "
    "these specific movies' names) is itself a finding: no leakage."
)
