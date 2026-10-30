"""
Fake news detection on Dravidian code-mixed text, explained with SHAP.

Data: DravidianLangTech @ EACL-2024 shared task, Task 1 (Malayalam
code-mixed social media comments, binary Fake / original). Official
train / dev / test splits are used as published -- the test split is
only ever touched for final scoring.

This is the credibility-task counterpart of the sentiment work:
  - same model family (TF-IDF + XGBoost) so TreeSHAP stays exact
  - same Language-Segment SHAP Attribution novelty, with the script
    range switched from Tamil to Malayalam
  - same three-way tag scheme: native script / English / romanized
    ("Manglish"), the last being the bucket that makes code-mixed
    text hard

Outputs feed the dashboard and the later real-news-feed application.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from spellchecker import SpellChecker
from xgboost import XGBClassifier

MALAYALAM_RANGE = re.compile(r"[ഀ-ൿ]")
spell = SpellChecker()


def tag_term(term: str) -> str:
    """Language tag for one vocabulary term. Mirrors the Tamil tagger,
    with the native-script range swapped for Malayalam."""
    if MALAYALAM_RANGE.search(term):
        return "ML-script"
    if term.lower() in ENGLISH_STOP_WORDS:
        return "EN"
    if not term.isalpha():
        return "NOISE"
    if spell.known([term.lower()]):
        return "EN"
    return "MANGLISH?"  # romanized, in neither dictionary


def load_split(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    # Label hygiene: the published files mix 'Fake' / 'original' casing.
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["fake", "original"])]
    df = df[df["text"].str.len() > 2]
    df["y"] = (df["label"] == "fake").astype(int)
    return df.reset_index(drop=True)


def main():
    train = load_split("Task-1_trainset.csv")
    dev = load_split("Task-1_devset.csv")
    test = load_split("Task-1-testset.csv")
    print(f"Official splits -- train {len(train)}, dev {len(dev)}, test {len(test)}")
    print("Train label balance:", train["label"].value_counts().to_dict())
    print()

    # sklearn's default token_pattern is r"(?u)\b\w\w+\b", and Python's \w
    # does NOT match Malayalam combining vowel signs (they are Unicode Mn,
    # and 'ാ'.isalnum() is False). That shatters every Malayalam word at its
    # vowel marks -- "നല്ല" becomes "നല". Explicitly including the Malayalam
    # block in the character class keeps whole words together, which the
    # language-segment attribution depends on being correct.
    INDIC_TOKEN_PATTERN = r"(?u)[\wഀ-ൿ஀-௿]{2,}"
    vectorizer = TfidfVectorizer(
        max_features=1500, min_df=2, sublinear_tf=True, ngram_range=(1, 1),
        token_pattern=INDIC_TOKEN_PATTERN,
    )
    X_train = vectorizer.fit_transform(train["text"]).toarray()
    X_dev = vectorizer.transform(dev["text"]).toarray()
    X_test = vectorizer.transform(test["text"]).toarray()
    feature_names = list(vectorizer.get_feature_names_out())

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, train["y"].values)

    dev_pred = model.predict(X_dev)
    test_pred = model.predict(X_test)
    dev_acc = accuracy_score(dev["y"], dev_pred)
    test_acc = accuracy_score(test["y"], test_pred)
    test_f1 = f1_score(test["y"], test_pred, average="macro")

    print(f"Dev  accuracy: {dev_acc:.4f}")
    print(f"TEST accuracy: {test_acc:.4f}   macro-F1: {test_f1:.4f}")
    print()
    print(classification_report(test["y"], test_pred, target_names=["original", "fake"]))

    # ---------------------------------------------------------------
    # TreeSHAP -- exact for tree ensembles
    # ---------------------------------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    recon = explainer.expected_value + shap_values[0].sum()
    raw = float(model.predict(X_test[:1], output_margin=True)[0])
    print(f"TreeSHAP additivity check: base+sum(phi) = {recon:.4f}  raw margin = {raw:.4f}")

    # ---------------------------------------------------------------
    # Language-Segment SHAP Attribution (the novelty, ported)
    # ---------------------------------------------------------------
    tags = np.array([tag_term(t) for t in feature_names])
    tag_df = pd.DataFrame({"term": feature_names, "tag": tags})
    tag_df.to_csv("fakenews_vocab_language_tags.csv", index=False)
    print("\nVocabulary tag distribution:")
    print(tag_df["tag"].value_counts().to_string())

    unique_tags = ["EN", "ML-script", "MANGLISH?", "NOISE"]
    rows = []
    for i in range(len(test)):
        present = X_test[i] != 0
        if not present.any():
            continue
        abs_shap = np.abs(shap_values[i])
        total = abs_shap[present].sum()
        if total == 0:
            continue
        row = {
            "text": test.iloc[i]["text"],
            "true_label": test.iloc[i]["label"],
            "predicted": "fake" if test_pred[i] == 1 else "original",
        }
        for tag in unique_tags:
            mask = present & (tags == tag)
            row[f"{tag}_shap_pct"] = round(100 * abs_shap[mask].sum() / total, 2)
        rows.append(row)

    per_instance = pd.DataFrame(rows)
    per_instance.to_csv("fakenews_language_segment_shap.csv", index=False)
    pct_cols = [c for c in per_instance.columns if c.endswith("_shap_pct")]

    print(f"\nLanguage-Segment attribution on {len(per_instance)} test comments")
    print("(mean % of |TreeSHAP| mass per language segment):")
    print(per_instance[pct_cols].mean().round(2).sort_values(ascending=False).to_string())
    print("\nSplit by predicted class:")
    print(per_instance.groupby("predicted")[pct_cols].mean().round(2).to_string())

    # Most informative tokens, with their language tag -- this is what the
    # later sentiment-leakage test consumes.
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:40]
    top_df = pd.DataFrame({
        "token": [feature_names[i] for i in top_idx],
        "mean_abs_shap": mean_abs[top_idx].round(5),
        "language_tag": tags[top_idx],
        # signed mean: positive pushes toward "fake"
        "mean_signed_shap": shap_values[:, top_idx].mean(axis=0).round(5),
    })
    top_df.to_csv("fakenews_top_tokens.csv", index=False)
    print("\nTop 15 tokens driving the fake/original decision:")
    print(top_df.head(15).to_string(index=False))

    np.save("fakenews_shap_values.npy", shap_values)
    np.save("fakenews_X_test.npy", X_test)
    model.save_model("fakenews_xgb_model.json")
    joblib.dump(vectorizer, "fakenews_tfidf_vectorizer.joblib")
    with open("fakenews_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    test.to_csv("fakenews_test_with_text.csv", index=False)
    with open("fakenews_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "task": "DravidianLangTech EACL-2024 Task 1, Malayalam code-mixed fake news",
            "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
            "dev_accuracy": round(float(dev_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_f1), 4),
            "vocab_tag_counts": tag_df["tag"].value_counts().to_dict(),
            "mean_language_segment_shap_pct": per_instance[pct_cols].mean().round(2).to_dict(),
        }, f, indent=2)

    print("\nSaved model, vectorizer, SHAP values and summaries.")


if __name__ == "__main__":
    main()
