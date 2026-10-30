"""
Part B - Classical ML half: TF-IDF + XGBoost on real DravidianCodeMix
Tamil-English binary sentiment (Positive/Negative), then TreeSHAP (exact,
fast, tree-specific) and KernelSHAP (model-agnostic baseline) on the same
model, on the same test instances, so the two can be compared directly.
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

train = pd.read_csv("dravidian_sentiment_train.csv")
test = pd.read_csv("dravidian_sentiment_test.csv")

# sklearn's default token_pattern r"(?u)\b\w\w+\b" breaks Tamil words at
# their combining vowel signs, because Python's \w does not match Unicode
# Mn marks ('ா'.isalnum() is False) -- so "படம்" was being indexed as the
# fragment "படம". Including the Tamil block in the character class keeps
# whole words together, which the language-segment attribution needs.
INDIC_TOKEN_PATTERN = r"(?u)[\w஀-௿ഀ-ൿ]{2,}"
vectorizer = TfidfVectorizer(
    max_features=400, min_df=3, sublinear_tf=True,
    token_pattern=INDIC_TOKEN_PATTERN,
)
X_train = vectorizer.fit_transform(train["text"]).toarray()
X_test = vectorizer.transform(test["text"]).toarray()
y_train, y_test = train["y"].values, test["y"].values
feature_names = vectorizer.get_feature_names_out()

model = XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.1,
    eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.4f}\n")
print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

# ---------------------------------------------------------------------------
# TreeSHAP: exact, fast, specific to tree ensembles
# ---------------------------------------------------------------------------
tree_explainer = shap.TreeExplainer(model)
tree_shap_values = tree_explainer.shap_values(X_test[:200])
print("\nTreeSHAP computed on 200 test instances.")
print("TreeSHAP additivity check (instance 0):",
      "base + sum(shap) =",
      round(tree_explainer.expected_value + tree_shap_values[0].sum(), 4),
      " model raw margin =",
      round(float(model.predict(X_test[:1], output_margin=True)[0]), 4))

# ---------------------------------------------------------------------------
# KernelSHAP: model-agnostic baseline, computed on a small sample
# (KernelSHAP is slow -- this is the whole reason TreeSHAP exists for trees)
# ---------------------------------------------------------------------------
background = shap.sample(X_train, 50, random_state=42)
kernel_explainer = shap.KernelExplainer(
    lambda x: model.predict_proba(x)[:, 1], background
)
N_KERNEL = 15
kernel_shap_values = kernel_explainer.shap_values(X_test[:N_KERNEL], nsamples=200)
print(f"\nKernelSHAP computed on {N_KERNEL} test instances (nsamples=200, model-agnostic).")

# ---------------------------------------------------------------------------
# Save everything needed for the comparison + language-segment steps
# ---------------------------------------------------------------------------
np.save("classical_X_test.npy", X_test)
np.save("classical_tree_shap_values.npy", tree_shap_values)
np.save("classical_kernel_shap_values.npy", kernel_shap_values)
with open("classical_feature_names.json", "w") as f:
    json.dump(list(feature_names), f)
with open("classical_expected_value.json", "w") as f:
    json.dump(float(tree_explainer.expected_value), f)
test.to_csv("classical_test_with_text.csv", index=False)
model.save_model("classical_xgb_model.json")
joblib.dump(vectorizer, "classical_tfidf_vectorizer.joblib")

print("\nSaved: classical_X_test.npy, classical_tree_shap_values.npy, "
      "classical_kernel_shap_values.npy, classical_feature_names.json, "
      "classical_xgb_model.json")
