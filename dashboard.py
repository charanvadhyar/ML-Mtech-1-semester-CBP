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
        "Fake News Detection",
        "Multilingual + Transfer",
        "Benchmark Audit (key finding)",
        "Live News: Leakage Consequences",
        "Score a Live URL",
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

    st.markdown("---")
    st.subheader("Headline results")

    if file_exists("multilingual_summary.json"):
        ms = load_json("multilingual_summary.json")
        rows = []
        au = pd.read_csv("benchmark_audit.csv") if file_exists("benchmark_audit.csv") else None
        fl = pd.read_csv("feed_leakage_summary.csv") if file_exists("feed_leakage_summary.csv") else None
        for l in ms["languages"]:
            r = {"language": l, "accuracy": ms["per_language"][l]["accuracy"]}
            if au is not None and (au["language"] == l).any():
                a = au[au["language"] == l].iloc[0]
                r["length-only acc"] = a["length_only_accuracy"]
                r["fact-check markers, fake"] = a["marker_pct_fake"]
                r["fact-check markers, real"] = a["marker_pct_real"]
            if fl is not None and (fl["lang"] == l).any():
                r["real news flagged fake"] = fl[fl["lang"] == l]["overall_pct_flagged_fake"].iloc[0]
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "Read left to right: the accuracy column is what a paper would report. "
            "The next three are what SHAP found underneath it. The last is what "
            "happens when the model meets real news."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Additivity verified", "0.0000", "exact, from the formula")
    if file_exists("multilingual_summary.json"):
        ms = load_json("multilingual_summary.json")
        off = [(a, b, v) for a, d in ms["transfer_matrix"].items()
               for b, v in d.items() if a != b]
        best = max(off, key=lambda t: t[2])
        c2.metric("Best cross-lingual transfer", f"{best[2]:.1%}",
                  f"{best[0]} to {best[1]}")
    if file_exists("length_shift_stats.json"):
        ls = load_json("length_shift_stats.json")
        c3.metric("Length shift vs false flags",
                  f"rho = {ls['spearman_rho']:.2f}",
                  f"exact p = {ls['spearman_p_exact_two_tailed']}")

    st.info(
        "**The through-line.** SHAP was recreated from the Shapley formula and "
        "verified exact (Part A). Applied to code-mixed sentiment, it showed which "
        "language a classifier leans on (Part C). Extended to five-language fake "
        "news, it revealed that every benchmark leaks -- two of them are "
        "fact-check-genre detectors scoring ~99%. Tested on live news, that leakage "
        "produces measurable false accusations against real journalism. Accuracy "
        "alone would have called all of this a success."
    )

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
elif page == "Fake News Detection":
    st.title("Fake News Detection on Dravidian Code-Mixed Text")
    st.markdown(
        """
The same SHAP machinery, pointed at a credibility task instead of a
sentiment one. Data is the **DravidianLangTech @ EACL-2024 Task 1**
shared task: Malayalam code-mixed social media comments labelled
*Fake* / *original*, using the official train/dev/test splits.

Language-Segment SHAP Attribution ports over directly -- only the
native-script Unicode range changes (Tamil to Malayalam), giving a
three-way tag of **ML-script / EN / MANGLISH?** (romanized, in neither
dictionary).
        """
    )

    if not file_exists("fakenews_summary.json"):
        st.warning("Run `train_fakenews.py` first to generate these results.")
    else:
        s = load_json("fakenews_summary.json")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{s['test_accuracy']:.1%}")
        c2.metric("Test macro-F1", f"{s['test_macro_f1']:.3f}")
        c3.metric("Train / dev / test", f"{s['n_train']} / {s['n_dev']} / {s['n_test']}")
        st.caption(
            "Held-out official test split, never used for fitting or tuning. "
            "Shared-task winners reached roughly 0.86-0.90 F1 with fine-tuned "
            "transformers, so this is a classical baseline, not state of the art."
        )

        st.subheader("Language-Segment attribution")
        st.bar_chart(pd.Series(s["mean_language_segment_shap_pct"]).sort_values(ascending=False))

        if file_exists("fakenews_language_segment_shap.csv"):
            per_instance = pd.read_csv("fakenews_language_segment_shap.csv")
            pct_cols = [c for c in per_instance.columns if c.endswith("_shap_pct")]
            st.subheader("Split by predicted class -- the key asymmetry")
            split = per_instance.groupby("predicted")[pct_cols].mean().round(2)
            st.dataframe(split, use_container_width=True)
            st.info(
                "English tokens carry far more of the decision when the model "
                "predicts **fake** than when it predicts **original**. The model "
                "has learned that English loanwords are a fake-news signal in this "
                "corpus -- which the top-token table below explains."
            )

        if file_exists("fakenews_top_tokens.csv"):
            st.subheader("What the model actually keys on")
            top = pd.read_csv("fakenews_top_tokens.csv")
            st.dataframe(top.head(20), use_container_width=True)
            st.warning(
                "**This is a topic shortcut, not credibility detection.** The "
                "strongest fake-pushing tokens are `china` / `ചԐന` and "
                "`corona` / `കൊറേണ`, while `party`, `cpm` and "
                "`പിണറായി` (Kerala politics) push toward original. "
                "The model is largely a COVID-era topic detector. Note that the same "
                "concept in both scripts carries the same directional signal, which "
                "is good evidence the language split is capturing real bilingual "
                "topic signal rather than an artifact."
            )
            st.caption(
                "A further caveat worth stating in the review: this dataset labels "
                "*comments*, not claims. Items labelled Fake include strings with no "
                "truth value at all, so the label is inherited from the parent post. "
                "The task is closer to 'was this posted under a fake news item' than "
                "'is this claim false'."
            )

# ---------------------------------------------------------------------------
elif page == "Multilingual + Transfer":
    st.title("Multilingual Fake News Detection and Cross-Lingual Transfer")

    if not file_exists("multilingual_summary.json"):
        st.warning("Run `prepare_multilingual.py` then `train_multilingual.py` first.")
    else:
        s = load_json("multilingual_summary.json")
        langs = s["languages"]
        st.markdown(
            f"One TF-IDF + XGBoost model per language ({', '.join(langs)}), each "
            "scored on its own held-out split. Every language is balanced "
            "fake/real, so **chance is 50% everywhere** and the numbers are "
            "directly comparable."
        )

        st.subheader("Per-language performance")
        rows = [{"language": l,
                 "accuracy": s["per_language"][l]["accuracy"],
                 "macro F1": s["per_language"][l]["macro_f1"],
                 "train": s["per_language"][l]["n_train"],
                 "test": s["per_language"][l]["n_test"],
                 "split": s["per_language"][l]["split"]} for l in langs]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Language-Segment attribution, per language")
        seg = pd.DataFrame({l: s["per_language"][l]["segment_attribution_pct"]
                            for l in langs}).T
        st.dataframe(seg.round(2), use_container_width=True)
        st.bar_chart(seg)
        st.caption(
            "NATIVE = the language's own script, EN = English dictionary words, "
            "ROMANIZED = Latin letters that are not English words (transliterated "
            "native text), NOISE = non-alphabetic. Only Malayalam is substantially "
            "code-mixed; the Tamil corpus is near-monolingual news headlines."
        )

        if file_exists("multilingual_transfer_matrix.csv"):
            st.subheader("Cross-lingual transfer -- the headline result")
            mat = pd.read_csv("multilingual_transfer_matrix.csv", index_col=0)
            st.dataframe((mat * 100).round(1), use_container_width=True)
            st.caption("Rows = trained on, columns = tested on. 50% is chance.")
            st.success(
                "**Transfer happens only where the language segment is shared, and "
                "the attribution predicts it.** Malayalam puts 27% of its attribution "
                "on English tokens, and a Malayalam-trained model scores 62.2% on "
                "English news. Tamil puts 0.2% on English tokens, and transfers at "
                "48.9% -- chance. Verified directly: when the Malayalam model judges "
                "English text, 97.3% of its SHAP mass sits on EN-tagged tokens and "
                "0.0% on native-script ones."
            )
            st.warning(
                "**Honest caveat.** The tokens actually carrying that transfer are "
                "mostly English function words (`of`, `the`, `to`, `in`) plus a few "
                "topic words (`china`, `india`, `news`). So the transferred signal is "
                "register and topic, not credibility knowledge. The predictive claim "
                "about segment share holds; calling it genuine credibility transfer "
                "would overstate it."
            )

        if file_exists("multilingual_pooled_vs_dedicated.csv"):
            st.subheader("Pooling all languages into one model")
            comp = pd.read_csv("multilingual_pooled_vs_dedicated.csv")
            st.dataframe(comp, use_container_width=True)
            st.metric("Pooled overall accuracy", f"{s['pooled_overall_accuracy']:.1%}")
            st.info(
                "Pooling **hurts the low-resource languages**: Malayalam loses about "
                "9.5 points and Tamil 6.7, while English barely moves. English is "
                "roughly 77% of the pooled training data, so it captures the shared "
                "vocabulary budget. The high pooled headline number is mostly English "
                "performance in disguise -- a good argument for per-language models."
            )

        if file_exists("multilingual_top_tokens.csv"):
            st.subheader("What each language's model keys on")
            top = pd.read_csv("multilingual_top_tokens.csv")
            pick = st.selectbox("Language", langs)
            st.dataframe(top[top["lang"] == pick].drop(columns=["lang"]),
                         use_container_width=True)

# ---------------------------------------------------------------------------
elif page == "Benchmark Audit (key finding)":
    st.title("Auditing Five Fake News Benchmarks with SHAP")
    st.markdown(
        """
Every per-language model here scores well -- 78.7% to 99.7%. This page
asks whether those scores measure credibility detection or
label-correlated artifacts. Three probes per language:

1. **Length-only baseline** -- a depth-3 tree on a single feature, the
   document's character count. Well above 50% means the label is partly
   readable before any word is examined.
2. **Fact-check marker prevalence** -- how often debunking vocabulary
   (*viral*, *claim*, *fake*, *caption*, *source*, and `BOOM`, an Indian
   fact-checking outlet) appears in each class.
3. **Marker ablation** -- retrain with those markers stripped. If accuracy
   holds, the leakage is a whole document genre rather than a few
   removable words, which is worse.
        """
    )

    if not file_exists("benchmark_audit.csv"):
        st.warning("Run `benchmark_audit.py` first.")
    else:
        audit = pd.read_csv("benchmark_audit.csv")
        st.subheader("The audit")
        st.dataframe(audit, use_container_width=True)

        st.subheader("Reported accuracy vs. what a single length feature already gets")
        st.bar_chart(audit.set_index("language")[["full_accuracy", "length_only_accuracy"]])

        st.subheader("Fact-check marker prevalence by class")
        st.bar_chart(audit.set_index("language")[["marker_pct_fake", "marker_pct_real"]])

        st.error(
            "**The accuracy ranking is almost exactly inverted by trustworthiness.** "
            "Telugu (99.7%) and Hindi (98.8%) score highest and are the most "
            "contaminated: 97.9% and 98.5% of their *fake* articles contain debunking "
            "vocabulary, against 14.6% and 27.4% of *real* ones. Their 'fake' class is "
            "fact-CHECK articles **about** false claims, not false articles -- so these "
            "models are fact-check-genre detectors. Stripping the markers changes "
            "accuracy by 0.0000 and 0.0026, meaning the leakage pervades the whole "
            "genre and cannot be removed feature by feature."
        )
        st.warning(
            "**Tamil leaks through length instead.** A single character-count feature "
            "reaches 75.2%, which is **66.8% of the model's entire above-chance "
            "performance** -- fake headlines average 110 characters against 86 for real. "
            "English is contaminated both ways: 56.1% of its fake statements carry "
            "fact-check markers against 1.2% of real ones, and length alone covers a "
            "third of its gain."
        )
        st.success(
            "**Malayalam is the cleanest of the five, and it scores lowest.** Its "
            "fact-check marker gap is +0.2 points, essentially zero, so its 78.7% is "
            "the most honest number in the table. It is also the only genuinely "
            "code-mixed corpus, which is why the Language-Segment novelty has real "
            "material to work with there and little elsewhere."
        )
        st.caption(
            "Takeaway for the write-up: this is the strongest argument in the project "
            "for why explainability is needed rather than nice to have. Accuracy alone "
            "would have reported a 99.7% Telugu fake news detector as a success."
        )

# ---------------------------------------------------------------------------
elif page == "Live News: Leakage Consequences":
    st.title("Does the Leakage Cause Real Harm? Live News Test")
    st.markdown(
        """
The audit showed these benchmarks leak. This page tests whether that
leakage causes **actual misclassification on real, current news** --
pulled live from Google News and publisher RSS in all five languages,
then scored by each language's own model.

Two predictions were registered in advance, which matters because one
held and one did not.
        """
    )

    if not file_exists("feed_leakage_summary.csv"):
        st.warning("Run `pull_feeds.bat`, then `feed_leakage_test.py`.")
    else:
        summ = pd.read_csv("feed_leakage_summary.csv")
        st.subheader("Prediction 1: fact-check vocabulary triggers false 'fake' flags")
        st.dataframe(summ, use_container_width=True)
        st.success(
            "**Held, where it could be tested.** In English, live articles that merely "
            "*mention* a viral claim are flagged fake **45.0%** of the time against "
            "**13.4%** for those that don't -- a +31.6 point gap. Hindi shows the same "
            "effect, 11.8% vs 1.3%, a roughly ninefold relative jump."
        )
        st.warning(
            "**Untestable in two languages, and I am not counting those as support.** "
            "Malayalam's live articles contained *zero* fact-check markers, so no gap "
            "can be computed. Telugu's model assigns mean P(fake) of 0.022 and predicts "
            "'real' for every single article -- it is saturated by domain shift, so its "
            "0.0 gap is uninformative rather than evidence."
        )

        if file_exists("length_shift_vs_falseflag.csv"):
            st.subheader("Prediction 2 (unanticipated, and the stronger finding)")
            shift = pd.read_csv("length_shift_vs_falseflag.csv")
            st.dataframe(shift, use_container_width=True)
            st.bar_chart(shift.set_index("lang")[["shift_ratio",
                                                   "pct_real_news_flagged_fake"]])
            st.error(
                "**The length artifact predicts real-world false accusations, exactly "
                "in rank order.** Models that learned 'longer means fake' flag ordinary "
                "journalism as fake whenever live text is longer than their training "
                "text. Malayalam, trained on 11-word comments and shown 63-word "
                "headlines (5.6x), flags **77.3% of real Kerala news as fake**. Tamil "
                "(4.6x) flags 34.0%, English (2.3x) 15.2%, Hindi (1.4x) 4.5%. Telugu, "
                "the one language where live text is *shorter* than training (0.28x), "
                "flags 0.0%."
            )
            st.caption(
                "Spearman rho = 1.000 across all five languages -- a perfect monotonic "
                "ranking. Exact two-tailed permutation p = 0.0167 (2 of 120 orderings); "
                "scipy's asymptotic p is not valid at n=5 and should not be quoted here. "
                "With only five points, and languages differing in many ways besides "
                "length shift, treat this as strongly suggestive rather than "
                "established. It does carry a clear mechanism, and it was predicted by "
                "the static length-only probe in the audit before any live data existed."
            )

        if file_exists("feed_flagged_examples.csv"):
            st.subheader("Real articles the models called fake")
            ex = pd.read_csv("feed_flagged_examples.csv")
            st.dataframe(ex, use_container_width=True)
            st.info(
                "Every row is legitimate journalism. The clearest case: *'SEBI cautions "
                "investors against fraudulent stock-tip WhatsApp groups'* scored "
                "**P(fake) = 0.996** -- a financial regulator warning the public about "
                "fraud, called fake news at 99.6% confidence, because warning people "
                "about scams uses the vocabulary of describing scams."
            )

# ---------------------------------------------------------------------------
elif page == "Score a Live URL":
    st.title("Score a Live News Article")
    st.markdown(
        "Paste any news URL, or pick one from today's pulled feed. The app "
        "fetches the article, detects its language by script, routes it to "
        "that language's model, and explains the verdict token by token."
    )

    try:
        import article_extract as AE
    except Exception as e:
        st.error(f"`article_extract.py` must sit next to dashboard.py ({e}).")
        AE = None

    MODEL_FOR = {
        "english": ("fakenews_english_model.json", "fakenews_english_vectorizer.joblib", None),
        "hindi": ("fakenews_hindi_model.json", "fakenews_hindi_vectorizer.joblib", "\u0900-\u097F"),
        "tamil": ("fakenews_tamil_model.json", "fakenews_tamil_vectorizer.joblib", "\u0B80-\u0BFF"),
        "telugu": ("fakenews_telugu_model.json", "fakenews_telugu_vectorizer.joblib", "\u0C00-\u0C7F"),
        "malayalam": ("fakenews_malayalam_model.json", "fakenews_malayalam_vectorizer.joblib", "\u0D00-\u0D7F"),
    }

    @st.cache_resource
    def _load(mp, vp):
        from xgboost import XGBClassifier
        m = XGBClassifier()
        m.load_model(mp)
        return m, joblib.load(vp), None

    @st.cache_data(show_spinner=False)
    def _fetch(u):
        return AE.load_url(u)

    if AE is not None:
        mode = st.radio("Input", ["Paste a URL", "Pick from today's feed",
                                  "Score the whole feed"], horizontal=True)

        # ----------------------------------------------------------
        # Batch mode: score every article pulled today, ranked, so you
        # can see at a glance what the models make of real news.
        # ----------------------------------------------------------
        if mode == "Score the whole feed":
            if not file_exists("feed_articles.csv"):
                st.warning("No feed_articles.csv yet -- run `pull_feeds.bat`.")
            else:
                fd = pd.read_csv("feed_articles.csv")
                fd["text"] = fd["text"].astype(str)

                @st.cache_data(show_spinner=False)
                def score_feed(sig):
                    out = []
                    for lg, grp in fd.groupby("lang"):
                        mp, vp, native = MODEL_FOR.get(lg, (None, None, None))
                        if not (mp and file_exists(mp) and file_exists(vp)):
                            continue
                        mdl, vz, _ = _load(mp, vp)
                        nm = list(vz.get_feature_names_out())
                        X = vz.transform(grp["text"]).toarray()
                        pr = mdl.predict_proba(X)[:, 1]
                        sv = shap.TreeExplainer(mdl).shap_values(X)
                        for k in range(len(grp)):
                            r = grp.iloc[k]
                            pres = np.nonzero(X[k])[0]
                            top = (", ".join(
                                nm[j] for j in pres[np.argsort(-sv[k][pres])][:4])
                                if len(pres) else "(no vocabulary match)")
                            ratio, nw = AE.shift_ratio(lg, str(r["text"]))
                            out.append({
                                "lang": lg,
                                "P(fake)": round(float(pr[k]), 3),
                                "verdict": "FAKE" if pr[k] >= 0.5 else "real",
                                "words": nw,
                                "shift": ratio,
                                "top tokens pushing FAKE": top,
                                "title": str(r["title"])[:110],
                                "source": r["source"],
                                "url": r["url"],
                            })
                    return pd.DataFrame(out)

                with st.spinner("Scoring every article pulled today..."):
                    scored = score_feed(len(fd))

                if scored.empty:
                    st.warning("No models available for the pulled languages.")
                else:
                    st.markdown(
                        "**Every one of these is real journalism from a mainstream "
                        "outlet.** Anything marked FAKE is a false accusation."
                    )
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Articles scored", len(scored))
                    k2.metric("Falsely flagged FAKE",
                              f"{(scored['verdict'] == 'FAKE').mean():.1%}")
                    k3.metric("Feed pulled", str(fd["published"].dropna().iloc[0])[:16]
                              if fd["published"].notna().any() else "today")

                    st.subheader("False-flag rate by language")
                    byl = scored.groupby("lang").agg(
                        articles=("verdict", "size"),
                        pct_flagged_fake=("verdict",
                                          lambda s: round(100 * (s == "FAKE").mean(), 1)),
                        mean_p_fake=("P(fake)", lambda s: round(s.mean(), 3)),
                    )
                    st.dataframe(byl, use_container_width=True)
                    st.bar_chart(byl["pct_flagged_fake"])

                    only = st.multiselect("Filter languages",
                                          sorted(scored["lang"].unique()),
                                          default=sorted(scored["lang"].unique()))
                    only_fake = st.checkbox("Show only articles flagged FAKE")
                    view = scored[scored["lang"].isin(only)]
                    if only_fake:
                        view = view[view["verdict"] == "FAKE"]
                    st.subheader(f"Today's news, ranked by P(fake) -- {len(view)} rows")
                    st.dataframe(
                        view.sort_values("P(fake)", ascending=False),
                        use_container_width=True,
                        column_config={"url": st.column_config.LinkColumn("url")},
                    )
                    st.download_button(
                        "Download these scores as CSV",
                        view.sort_values("P(fake)", ascending=False).to_csv(index=False),
                        file_name="todays_feed_scored.csv", mime="text/csv")
                    st.caption(
                        "The 'top tokens pushing FAKE' column is the short version "
                        "of the SHAP explanation. Switch to 'Pick from today's feed' "
                        "for the full token-level breakdown of any single article."
                    )

        art = None
        if mode == "Paste a URL":
            url = st.text_input("News article URL", placeholder="https://...")
            if url.strip():
                with st.spinner("Fetching and extracting..."):
                    try:
                        art = _fetch(url.strip())
                    except Exception as e:
                        st.error(
                            f"Could not fetch that page: {type(e).__name__}. "
                            "Some publishers block automated requests, or the "
                            "page may be paywalled. Try another, or use the "
                            "feed option."
                        )
        elif mode == "Pick from today's feed":
            if not file_exists("feed_articles.csv"):
                st.warning("No feed_articles.csv yet -- run `pull_feeds.bat`.")
            else:
                fd = pd.read_csv("feed_articles.csv")
                fd["text"] = fd["text"].astype(str)
                lang_pick = st.selectbox("Language", sorted(fd["lang"].unique()))
                sub = fd[fd["lang"] == lang_pick].reset_index(drop=True)
                idx = st.selectbox(
                    "Article", range(len(sub)),
                    format_func=lambda i: str(sub.iloc[i]["title"])[:110])
                row = sub.iloc[idx]
                ratio, n_words = AE.shift_ratio(lang_pick, str(row["text"]))
                art = {"url": row["url"], "title": row["title"],
                       "text": str(row["text"]), "lang": lang_pick,
                       "script_confidence": 1.0, "n_words": n_words,
                       "shift_ratio": ratio}
                st.caption(
                    "Feed items are headline plus summary, so they are short. "
                    "Pasting the full article URL gives the models more to work "
                    "with."
                )

        if art and art["text"]:
            lang = art["lang"]
            st.markdown("---")
            st.subheader(art["title"] or "(no headline found)")

            c1, c2, c3 = st.columns(3)
            c1.metric("Detected language", lang,
                      f"{art['script_confidence']:.0%} of letters in script")
            c2.metric("Article length", f"{art['n_words']} words")
            if art["shift_ratio"]:
                c3.metric("vs. training length", f"{art['shift_ratio']}x")

            with st.expander("Extracted text"):
                st.write(art["text"][:3000] + ("..." if len(art["text"]) > 3000 else ""))

            mp, vp, native = MODEL_FOR[lang]
            if not (file_exists(mp) and file_exists(vp)):
                st.warning(f"No trained model for {lang} in this folder yet.")
            else:
                model, vec, _ = _load(mp, vp)
                names = list(vec.get_feature_names_out())
                x = vec.transform([art["text"]]).toarray()
                prob = float(model.predict_proba(x)[0, 1])
                base = float(model.predict_proba(np.zeros_like(x))[0, 1])

                verdict = "FAKE" if prob >= 0.5 else "real"
                a, b = st.columns(2)
                a.metric("Model verdict", verdict)
                b.metric("P(fake)", f"{prob:.3f}", f"base rate {base:.3f}")

                if art["shift_ratio"] and art["shift_ratio"] > 2:
                    st.error(
                        f"**Treat this verdict as unreliable.** This article is "
                        f"{art['shift_ratio']}x longer than the documents this "
                        f"{lang} model was trained on ({AE.TRAIN_WORDS[lang]:.0f} "
                        "words on average). The audit found these models partly "
                        "encode 'longer means fake', so length alone pushes the "
                        "score up. This is the failure mode measured on the "
                        "live-news page, visible on a single article."
                    )
                elif art["shift_ratio"] and art["shift_ratio"] < 0.5:
                    st.warning(
                        f"This article is much *shorter* than this model's "
                        f"training documents ({AE.TRAIN_WORDS[lang]:.0f} words), "
                        "which biases it toward 'real'. Also out of domain."
                    )

                spell = load_spellchecker()
                nre = re.compile(f"[{native}]") if native else None
                sv = shap.TreeExplainer(model).shap_values(x)[0]
                present = np.nonzero(x[0])[0]

                if len(present) == 0:
                    st.info(
                        "None of this article's words are in the model's "
                        "vocabulary, so it returned its base rate. The verdict "
                        "carries no information about this article."
                    )
                else:
                    rows = [{
                        "token": names[i],
                        "shap_value": round(float(sv[i]), 4),
                        "pushes_toward": "FAKE" if sv[i] > 0 else "real",
                        "language_tag": ("NATIVE" if (nre and nre.search(names[i]))
                                         else tag_term(names[i], spell)),
                    } for i in present]
                    contrib = pd.DataFrame(rows).sort_values(
                        "shap_value", key=abs, ascending=False)
                    st.subheader("Why the model said that")
                    st.dataframe(contrib.head(25), use_container_width=True)

                    mass = contrib.groupby("language_tag")["shap_value"].apply(
                        lambda s: s.abs().sum())
                    st.subheader("Language-segment breakdown")
                    st.bar_chart(100 * mass / mass.sum())

                    toward_fake = contrib[contrib["shap_value"] > 0]["token"].head(8).tolist()
                    marker_like = [t for t in toward_fake if t.lower() in {
                        "viral", "claim", "fake", "false", "misleading", "rumour",
                        "rumor", "video", "social", "media", "photo", "share"}]
                    if marker_like:
                        st.warning(
                            "**Leakage, caught live.** Among the tokens pushing this "
                            f"article toward FAKE are {', '.join(marker_like)} -- "
                            "fact-check genre vocabulary, not evidence about whether "
                            "this particular story is true. The benchmark's 'fake' "
                            "class was largely fact-check articles, so the model "
                            "penalises any story that discusses a claim at all."
                        )

# ---------------------------------------------------------------------------
elif page == "Live Demo":
    st.title("Live Demo -- try any sentence on any model")
    st.markdown(
        "Pick a model, type a sentence, and get the prediction plus a "
        "word-by-word TreeSHAP explanation tagged by language segment. "
        "The six models are the Tamil-English **sentiment** classifier and "
        "the five **fake news** classifiers."
    )

    MODELS = {
        "Sentiment (Tamil-English)": {
            "model": "classical_xgb_model.json",
            "vec": "classical_tfidf_vectorizer.joblib",
            "names": "classical_feature_names.json",
            "pos": "Positive", "neg": "Negative", "native": None,
            "example": "Semma mass padam da, thalaiva vera level acting!",
        },
        "Fake news -- English": {
            "model": "fakenews_english_model.json",
            "vec": "fakenews_english_vectorizer.joblib", "names": None,
            "pos": "FAKE", "neg": "real", "native": None,
            "example": "Viral WhatsApp claim about new currency note is misleading",
        },
        "Fake news -- Malayalam": {
            "model": "fakenews_malayalam_model.json",
            "vec": "fakenews_malayalam_vectorizer.joblib", "names": None,
            "pos": "FAKE", "neg": "real", "native": "\u0D00-\u0D7F",
            "example": "\u0d08 \u0d35\u0d3e\u0d7c\u0d24\u0d4d\u0d24 \u0d35\u0d48\u0d31\u0d7d \u0d06\u0d23\u0d4d \u0d2a\u0d15\u0d4d\u0d37\u0d47 \u0d35\u0d4d\u0d2f\u0d3e\u0d1c\u0d2e\u0d3e\u0d23\u0d4d",
        },
        "Fake news -- Tamil": {
            "model": "fakenews_tamil_model.json",
            "vec": "fakenews_tamil_vectorizer.joblib", "names": None,
            "pos": "FAKE", "neg": "real", "native": "\u0B80-\u0BFF",
            "example": "\u0b87\u0ba8\u0bcd\u0ba4 \u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bbf \u0bb5\u0bc8\u0bb0\u0bb2\u0bbe\u0b95 \u0baa\u0bb0\u0bb5\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1",
        },
        "Fake news -- Hindi": {
            "model": "fakenews_hindi_model.json",
            "vec": "fakenews_hindi_vectorizer.joblib", "names": None,
            "pos": "FAKE", "neg": "real", "native": "\u0900-\u097F",
            "example": "\u0938\u094b\u0936\u0932 \u092e\u0940\u0921\u093f\u092f\u093e \u092a\u0930 \u0935\u093e\u092f\u0930\u0932 \u0926\u093e\u0935\u093e \u092a\u0942\u0930\u0940 \u0924\u0930\u0939 \u092b\u0930\u094d\u091c\u0940 \u0939\u0948",
        },
        "Fake news -- Telugu": {
            "model": "fakenews_telugu_model.json",
            "vec": "fakenews_telugu_vectorizer.joblib", "names": None,
            "pos": "FAKE", "neg": "real", "native": "\u0C00-\u0C7F",
            "example": "\u0c38\u0c4b\u0c36\u0c32\u0c4d \u0c2e\u0c40\u0c21\u0c3f\u0c2f\u0c3e\u0c32\u0c4b \u0c35\u0c48\u0c30\u0c32\u0c4d \u0c05\u0c35\u0c41\u0c24\u0c41\u0c28\u0c4d\u0c28 \u0c35\u0c3e\u0c30\u0c4d\u0c24 \u0c24\u0c2a\u0c4d\u0c2a\u0c41",
        },
    }

    @st.cache_resource
    def load_any(mp, vp):
        from xgboost import XGBClassifier
        m = XGBClassifier()
        m.load_model(mp)
        v = joblib.load(vp)
        return m, v, shap.TreeExplainer(m)

    choice = st.selectbox("Model", list(MODELS))
    cfg = MODELS[choice]

    if not (file_exists(cfg["model"]) and file_exists(cfg["vec"])):
        st.warning(f"`{cfg['model']}` is not in this folder yet. "
                   "Run the training step for it first.")
    else:
        text = st.text_area("Your sentence", value=cfg["example"], height=90)
        if text.strip():
            spell = load_spellchecker()
            model, vectorizer, explainer = load_any(cfg["model"], cfg["vec"])
            names = list(vectorizer.get_feature_names_out())

            x = vectorizer.transform([text]).toarray()
            prob = float(model.predict_proba(x)[0, 1])
            label = cfg["pos"] if prob >= 0.5 else cfg["neg"]

            c1, c2 = st.columns(2)
            c1.metric("Prediction", label)
            c2.metric(f"P({cfg['pos']})", f"{prob:.3f}")

            native = re.compile(f"[{cfg['native']}]") if cfg["native"] else None

            def tag_for(term):
                if native is not None and native.search(term):
                    return "NATIVE"
                return tag_term(term, spell)

            shap_vals = explainer.shap_values(x)[0]
            present = np.nonzero(x[0])[0]
            if len(present) == 0:
                st.info(
                    "None of these words are in this model's vocabulary, so it "
                    "falls back to its base rate. That is itself worth seeing: "
                    "these are small TF-IDF vocabularies, and out-of-vocabulary "
                    "input is exactly where the domain-shift failures on the "
                    "live-news page come from."
                )
            else:
                rows = [{
                    "token": names[i],
                    "shap_value": round(float(shap_vals[i]), 4),
                    "pushes_toward": cfg["pos"] if shap_vals[i] > 0 else cfg["neg"],
                    "language_tag": tag_for(names[i]),
                } for i in present]
                contrib = pd.DataFrame(rows).sort_values(
                    "shap_value", key=abs, ascending=False)
                st.subheader("Word-level contributions")
                st.dataframe(contrib, use_container_width=True)

                st.subheader("Language-segment breakdown for this sentence")
                mass = contrib.groupby("language_tag")["shap_value"].apply(
                    lambda s: s.abs().sum())
                st.bar_chart(100 * mass / mass.sum())
                st.caption(
                    f"Base rate (no features firing): "
                    f"P({cfg['pos']}) = "
                    f"{float(model.predict_proba(np.zeros_like(x))[0, 1]):.3f}. "
                    "Each token's SHAP value is its additive contribution away "
                    "from that baseline, which is the property verified from the "
                    "formula in Part A."
                )

            if "Fake news" in choice:
                st.info(
                    "Try adding a word like *viral*, *claim* or *fake* to an "
                    "otherwise ordinary sentence. On the English, Hindi and "
                    "Telugu models the score tends to jump, because their "
                    "training 'fake' class was largely fact-CHECK articles "
                    "about false claims. That is the leakage, hands on."
                )
