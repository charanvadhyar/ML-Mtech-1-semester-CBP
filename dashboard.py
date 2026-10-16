"""
Streamlit dashboard for the full SHAP explainability project:
  Part A - base SHAP paper recreation (facts from the notebook)
  Part B - SHAP variants comparison (TreeSHAP / KernelSHAP / PartitionSHAP)
  Part C - Language-Segment SHAP Attribution (the novelty), both on
           DravidianCodeMix and on our own real scraped YouTube comments
  Live demo - type any Tamil-English sentence, get a live explanation

Run with:  streamlit run dashboard.py
(from inside the MLSHAP folder, after train_classical.py / apply_classical_to_scraped.py
and, optionally, run_bert_shap.bat have been run so their output files exist here.)
"""
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spellchecker import SpellChecker
from xgboost import XGBClassifier

st.set_page_config(page_title="Code-Mixed SHAP Explainability", layout="wide")

TAMIL_RANGE = re.compile(r"[஀-௿]")
TAG_COLORS = {
    "EN": "#B8D9F5", "TA-script": "#FFD9A0", "TANGLISH?": "#FFB3A7",
    "NOISE": "#E0E0E0", "UNTAGGED": "#E6E6E6",
}


@st.cache_resource
def load_spellchecker():
    return SpellChecker()


def tag_term(term, spell):
    if TAMIL_RANGE.search(term):
        return "TA-script"
    if term.lower() in ENGLISH_STOP_WORDS:
        return "EN"
    if not term.isalpha():
        return "NOISE"
    if spell.known([term.lower()]):
        return "EN"
    return "TANGLISH?"


@st.cache_resource
def load_classical():
    model = XGBClassifier()
    model.load_model("classical_xgb_model.json")
    vectorizer = joblib.load("classical_tfidf_vectorizer.joblib")
    with open("classical_feature_names.json") as f:
        feature_names = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, vectorizer, feature_names, explainer


def file_exists(path):
    return os.path.exists(path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
st.sidebar.title("Explaining Model Predictions\nUsing SHAP for Text Classifiers")
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Part A -- Base SHAP Recreation",
        "Part B -- SHAP Variants Comparison",
        "Part C -- Language-Segment Attribution (Novelty)",
        "Ablation Study (Causal Test)",
        "Live Demo",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Tamil-English code-mixed sentiment classification, trained on "
    "DravidianCodeMix, explained with SHAP, and applied to real scraped "
    "YouTube comments."
)

# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Explaining Model Predictions Using SHAP for Text Classifiers")
    st.markdown(
        """
This project has three parts:

1. **Part A** -- recreate the base SHAP paper (Lundberg & Lee, 2017) from the
   literal Shapley-value formula, and verify its additivity/missingness
   properties against the `shap` library and against LIME.
2. **Part B** -- train two models (classical TF-IDF+XGBoost, and
   transformer mBERT/IndicBERT) on the **real DravidianCodeMix Tamil-English
   sentiment dataset**, and compare SHAP variants (TreeSHAP, KernelSHAP,
   PartitionSHAP) on the same predictions.
3. **Part C (the novelty)** -- *Language-Segment SHAP Attribution*: tag
   every token as English / Tamil-script / Tanglish (romanized, not in
   either dictionary) and aggregate each model's SHAP attribution mass by
   language segment, to answer: **does the model lean on English or Tamil
   tokens to make its decision?** We run this both on DravidianCodeMix
   itself and on our **own real scraped YouTube comments** (via a
   from-scratch, rule-based, no-paid-API data pipeline: scrape -> extract
   named entities from video credits -> statistically validate -> tag every
   word -> classify).
        """
    )

    col1, col2, col3 = st.columns(3)
    if file_exists("dravidian_sentiment_train.csv"):
        n = len(pd.read_csv("dravidian_sentiment_train.csv")) + len(pd.read_csv("dravidian_sentiment_test.csv"))
        col1.metric("DravidianCodeMix comments used", f"{n:,}")
    if file_exists("comments.csv"):
        sc = pd.read_csv("comments.csv")
        col2.metric("Real scraped YouTube comments", f"{len(sc):,}", f"{sc['video_id'].nunique()} videos")
    if file_exists("entity_exclusion_list.json"):
        ex = load_json("entity_exclusion_list.json")
        col3.metric("Named entities auto-extracted", len(ex.get("combined_exclusion_list", [])))

# ---------------------------------------------------------------------------
elif page == "Part A -- Base SHAP Recreation":
    st.title("Part A -- Recreating the Base SHAP Paper")
    st.markdown(
        """
From `shap_base_paper_recreation.ipynb`: a brute-force Shapley value
implementation, built directly from the formula in Lundberg & Lee (2017),
and checked against three properties.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Additivity: reconstruction error", "0.0000000000", "exact")
    c2.metric("Missingness: masked feature phi", "0.0", "exact")
    c3.metric("Cross-check vs shap.KernelExplainer", "match to 4 d.p.", "0.0000 mean |diff|")

    st.subheader("SHAP vs. LIME -- why \"unified approach\" is not just a slogan")
    st.markdown(
        """
Both SHAP and LIME fit a weighted linear surrogate model on perturbed
samples -- the only difference is the weighting kernel. SHAP's kernel is
derived so the surrogate's coefficients are *exactly* the Shapley values;
LIME's kernel is a heuristic choice with no such guarantee.
        """
    )
    c1, c2 = st.columns(2)
    c1.metric("SHAP reconstruction error (same instance)", "0.0000", "exact")
    c2.metric("LIME reconstruction error (same instance)", "~0.598", "approximate")
    st.caption(
        "Numbers are from the executed notebook (shap_base_paper_recreation.ipynb). "
        "Re-open that notebook for the full derivation and code."
    )

# ---------------------------------------------------------------------------
elif page == "Part B -- SHAP Variants Comparison":
    st.title("Part B -- Multiple SHAP Implementations")
    st.markdown(
        "One dataset (DravidianCodeMix Tamil-English sentiment), two model families, "
        "several SHAP variants -- do they agree on what matters?"
    )

    if not file_exists("classical_tree_shap_values.npy"):
        st.warning("Run `train_classical.py` first to generate these results.")
    else:
        model, vectorizer, feature_names, explainer = load_classical()
        tree_shap = np.load("classical_tree_shap_values.npy")
        kernel_shap = np.load("classical_kernel_shap_values.npy")
        test_df = pd.read_csv("classical_test_with_text.csv")

        st.subheader("Classical model: TF-IDF + XGBoost")
        X_test = np.load("classical_X_test.npy")
        from sklearn.metrics import accuracy_score
        preds = model.predict(X_test)
        acc = accuracy_score(test_df["y"].values[: len(preds)], preds)
        st.metric("Test accuracy", f"{acc:.1%}")

        st.subheader("TreeSHAP vs. KernelSHAP -- same model, same instances")
        n_overlap = kernel_shap.shape[0]
        tree_subset = tree_shap[:n_overlap]
        # Compare per-instance total |SHAP| ranking agreement (Spearman-like, via correlation of magnitudes)
        corrs = [
            float(np.corrcoef(tree_subset[i], kernel_shap[i])[0, 1])
            for i in range(n_overlap)
        ]
        st.metric(
            f"Mean correlation between TreeSHAP and KernelSHAP attributions ({n_overlap} instances)",
            f"{np.mean(corrs):.3f}",
        )
        st.caption(
            "TreeSHAP is exact and fast for tree ensembles; KernelSHAP is the slower, "
            "model-agnostic method SHAP generalizes from. High agreement here is the "
            "expected sanity check -- they're approximating the same quantity."
        )

        st.subheader("Top tokens driving this model's predictions (mean |TreeSHAP|)")
        mean_abs = np.abs(tree_shap).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]
        top_df = pd.DataFrame({
            "token": [feature_names[i] for i in top_idx],
            "mean_abs_shap": [mean_abs[i] for i in top_idx],
        })
        st.bar_chart(top_df.set_index("token"))

        st.markdown("---")
        st.subheader("Transformer model: mBERT / IndicBERT + PartitionSHAP")
        for name in ["mbert", "indicbert"]:
            summary_path = f"bert_{name}_summary.json"
            if file_exists(summary_path):
                s = load_json(summary_path)
                st.markdown(f"**{name}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Test accuracy", f"{s['test_accuracy']:.1%}")
                c2.metric("Train size", s["n_train"])
                c3.metric("PartitionSHAP instances", s["n_shap_instances"])
            else:
                st.info(
                    f"**{name}**: not run yet. On the machine with the local model "
                    f"weights, run `run_bert_shap.bat`, then re-open this dashboard."
                )

# ---------------------------------------------------------------------------
elif page == "Part C -- Language-Segment Attribution (Novelty)":
    st.title("Part C -- Language-Segment SHAP Attribution")
    st.markdown(
        """
**The question:** when the model predicts sentiment on a Tamil-English
code-mixed comment, does it lean on the English tokens, the Tamil-script
tokens, or the romanized "Tanglish" tokens that are in neither dictionary?

**The method:** tag every token the model actually used (by script/dictionary,
or -- on our own scraped data -- our real entity-aware pipeline's own tags),
then sum each token's \\|SHAP\\| contribution by language segment.
        """
    )

    tab1, tab2 = st.tabs(["On DravidianCodeMix", "On our own real scraped YouTube comments"])

    with tab1:
        if not file_exists("language_segment_shap_per_instance.csv"):
            st.warning("Run `language_segment_shap.py` first.")
        else:
            per_instance = pd.read_csv("language_segment_shap_per_instance.csv")
            pct_cols = [c for c in per_instance.columns if c.endswith("_shap_pct")]
            st.subheader("Dataset-level trend (classical model, TreeSHAP)")
            st.bar_chart(per_instance[pct_cols].mean().sort_values(ascending=False))
            st.subheader("Split by predicted label")
            st.dataframe(per_instance.groupby("label")[pct_cols].mean().round(2))

    with tab2:
        if not file_exists("scraped_language_segment_summary.json"):
            st.warning("Run `apply_classical_to_scraped.py` first.")
        else:
            summary = load_json("scraped_language_segment_summary.json")
            st.subheader(f"Dataset-level trend ({summary['n_instances']} real scraped comments)")
            overall = pd.Series(summary["overall_mean_pct"]).sort_values(ascending=False)
            st.bar_chart(overall)

            st.subheader("By video (does language reliance differ by movie/fandom?)")
            by_video = pd.DataFrame(summary["by_video"])
            st.dataframe(by_video.round(2))
            st.bar_chart(by_video)

            st.subheader("Worked examples")
            per_instance = pd.read_csv("scraped_comments_language_segment_shap.csv")
            for _, row in per_instance.sample(min(5, len(per_instance)), random_state=1).iterrows():
                with st.container(border=True):
                    st.write(f"**{row['text']}**")
                    st.caption(f"Predicted: {row['predicted_sentiment']}  |  video: {row['video_id']}")
                    tag_pcts = {
                        k.replace("_shap_pct", ""): v
                        for k, v in row.items()
                        if k.endswith("_shap_pct")
                    }
                    st.bar_chart(pd.Series(tag_pcts))

        for name in ["mbert", "indicbert"]:
            scraped_summary_path = f"bert_{name}_scraped_summary.json"
            if file_exists(scraped_summary_path):
                s = load_json(scraped_summary_path)
                st.subheader(f"{name} -- same analysis, transformer model")
                st.bar_chart(pd.Series(s["overall_mean_pct"]))

# ---------------------------------------------------------------------------
elif page == "Ablation Study (Causal Test)":
    st.title("Ablation Study -- a causal test, not just attribution")
    st.markdown(
        """
SHAP attribution tells you *how much a token contributed* to a given
prediction, but correlation isn't causation. This test is simpler and
stronger: **mask out every token in one category and literally re-run the
model** on real scraped comments, then measure how much the prediction
actually moves.

Two categories, chosen to test two different questions:
- **Language categories (EN / TA-script / Tanglish)** -- does the model's
  decision *causally depend* on one language's tokens being present?
- **Named entities** (actor/movie/crew names, from our own rule-based
  extraction) -- is the model secretly leaning on proper nouns specific to
  these movies, rather than genuine sentiment language?
        """
    )

    if not file_exists("ablation_study_summary.csv"):
        st.warning("Run `ablation_study.py` first to generate these results.")
    else:
        summary = pd.read_csv("ablation_study_summary.csv")
        st.subheader("Mean |change in P(Positive)| when a category is masked")
        st.bar_chart(summary.set_index("category")["mean_abs_delta_P"])

        st.subheader("% of affected comments whose predicted class actually flipped")
        st.bar_chart(summary.set_index("category")["pct_predictions_flipped"])

        st.dataframe(summary, use_container_width=True)

        st.info(
            "Reading this table: a high mean-delta / flip-rate means the model's "
            "decision causally depends on that category -- not just correlates "
            "with it. A category with many affected comments but near-zero shift "
            "(e.g. named entities) is itself a finding: the model isn't leaking "
            "on movie-specific proper nouns, which is reassuring for the "
            "classifier's validity, not a null result to bury."
        )

# ---------------------------------------------------------------------------
elif page == "Live Demo":
    st.title("Live Demo -- try your own sentence")
    st.markdown(
        "Type any Tamil-English code-mixed sentence. The classical model "
        "(trained on DravidianCodeMix) predicts sentiment, and TreeSHAP "
        "explains which words drove it, tagged by language."
    )

    example = "Semma mass padam da, thalaiva vera level acting!"
    text = st.text_input("Your sentence", value=example)

    if text and file_exists("classical_xgb_model.json"):
        spell = load_spellchecker()
        model, vectorizer, feature_names, explainer = load_classical()

        x = vectorizer.transform([text]).toarray()
        prob = model.predict_proba(x)[0, 1]
        pred = "Positive" if prob >= 0.5 else "Negative"

        c1, c2 = st.columns(2)
        c1.metric("Predicted sentiment", pred)
        c2.metric("P(Positive)", f"{prob:.2f}")

        shap_vals = explainer.shap_values(x)[0]
        present_idx = np.nonzero(x[0])[0]
        if len(present_idx) == 0:
            st.info("None of this sentence's words are in the model's TF-IDF vocabulary (it's a small, 400-term vocab from DravidianCodeMix).")
        else:
            rows = []
            for idx in present_idx:
                term = feature_names[idx]
                rows.append({
                    "token": term,
                    "shap_value": round(float(shap_vals[idx]), 4),
                    "language_tag": tag_term(term, spell),
                })
            contrib_df = pd.DataFrame(rows).sort_values("shap_value", key=abs, ascending=False)
            st.subheader("Word-level contributions (toward Positive)")
            st.dataframe(contrib_df, use_container_width=True)

            tag_mass = contrib_df.groupby("language_tag")["shap_value"].apply(lambda s: s.abs().sum())
            st.subheader("This sentence's language-segment breakdown")
            st.bar_chart(tag_mass)
