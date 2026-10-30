"""
Multilingual fake news detection with SHAP, across however many languages
prepare_multilingual.py managed to assemble.

Three things are computed:

1. PER-LANGUAGE MODELS. One TF-IDF + XGBoost per language (TreeSHAP stays
   exact on trees), scored on its own held-out split, plus Language-Segment
   SHAP Attribution using generic tags -- NATIVE / EN / ROMANIZED / NOISE --
   so the numbers are comparable across languages.

2. CROSS-LINGUAL TRANSFER MATRIX. Train on language A, test on language B.
   Every language is label-balanced, so chance is 50% and anything above it
   is real transfer. This is the interesting one: if a Tamil-trained model
   beats chance on Malayalam, something is crossing the language boundary,
   and the language-segment tags say what.

3. POOLED MODEL. All languages at once, then its per-language accuracy is
   compared against the dedicated models to see whether pooling helps
   (transfer) or hurts (interference). Because each language was balanced
   before pooling, a model cannot profit from merely identifying the
   language, so this comparison is about the text.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
from xgboost import XGBClassifier

spell = SpellChecker()

# Python's \w does not match Indic combining vowel signs (they are Unicode
# Mn, and 'ா'.isalnum() is False), so sklearn's default token_pattern
# shatters every Indic word at its vowel marks. Naming the script blocks
# explicitly keeps whole words together.
SCRIPT_BLOCKS = {
    "tamil": "஀-௿",
    "malayalam": "ഀ-ൿ",
    "hindi": "ऀ-ॿ",
    "telugu": "ఀ-౿",
    "english": "",
}
ALL_BLOCKS = "".join(SCRIPT_BLOCKS.values())
TOKEN_PATTERN = rf"(?u)[\w{ALL_BLOCKS}]{{2,}}"
TAGS = ["EN", "NATIVE", "ROMANIZED", "NOISE"]


def make_tagger(lang):
    """Language tag for a vocabulary term, relative to one language's own
    native script. 'ROMANIZED' is the hard bucket: Latin letters that are
    not English words -- transliterated native-language text."""
    import re
    block = SCRIPT_BLOCKS.get(lang, "")
    native = re.compile(f"[{block}]") if block else None

    def tag(term):
        if native is not None and native.search(term):
            return "NATIVE"
        if term.lower() in ENGLISH_STOP_WORDS:
            return "EN"
        if not term.isalpha():
            return "NOISE"
        if spell.known([term.lower()]):
            return "EN"
        return "ROMANIZED"

    return tag


def split_language(df, seed=42):
    """Use the published splits when the source has them, otherwise a
    stratified 80/20. Keeps the Malayalam shared-task test set intact."""
    if set(df["split"]) >= {"train", "test"}:
        tr = df[df["split"].isin(["train", "dev"])]
        te = df[df["split"] == "test"]
        return tr, te, "official"
    tr, te = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["y"])
    return tr, te, "stratified 80/20"


def fit_one(train_df, max_features=1500):
    vec = TfidfVectorizer(max_features=max_features, min_df=2, sublinear_tf=True,
                          token_pattern=TOKEN_PATTERN)
    X = vec.fit_transform(train_df["text"]).toarray()
    model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                          subsample=0.9, colsample_bytree=0.9,
                          eval_metric="logloss", random_state=42)
    model.fit(X, train_df["y"].values)
    return vec, model


def main():
    data = pd.read_csv("multilingual_fakenews.csv")
    data["text"] = data["text"].astype(str)
    langs = sorted(data["lang"].unique())
    print(f"Languages: {langs}\n")

    models, vecs, tests, per_lang = {}, {}, {}, {}
    seg_rows, top_rows = [], []

    # -------------------------------------------------------------
    # 1. Per-language models
    # -------------------------------------------------------------
    for lang in langs:
        d = data[data["lang"] == lang]
        tr, te, how = split_language(d)
        vec, model = fit_one(tr)
        Xte = vec.transform(te["text"]).toarray()
        pred = model.predict(Xte)
        acc = accuracy_score(te["y"], pred)
        f1 = f1_score(te["y"], pred, average="macro")
        print(f"=== {lang:10s} train {len(tr):>6}  test {len(te):>5} ({how})"
              f"   acc {acc:.4f}  macroF1 {f1:.4f}")

        models[lang], vecs[lang], tests[lang] = model, vec, te

        # Language-Segment SHAP
        names = list(vec.get_feature_names_out())
        tagger = make_tagger(lang)
        tags = np.array([tagger(t) for t in names])
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xte)

        present_any = Xte != 0
        totals = np.abs(sv) * present_any
        row_tot = totals.sum(axis=1)
        keep = row_tot > 0
        seg = {}
        for tg in TAGS:
            m = tags == tg
            seg[tg] = float((totals[keep][:, m].sum(axis=1) / row_tot[keep]).mean() * 100)
        seg_rows.append({"lang": lang, **{f"{k}_pct": round(v, 2) for k, v in seg.items()}})
        print("    segment attribution: " +
              "  ".join(f"{k} {v:.1f}%" for k, v in sorted(seg.items(), key=lambda x: -x[1])))

        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]
        for i in order:
            top_rows.append({
                "lang": lang, "token": names[i],
                "mean_abs_shap": round(float(mean_abs[i]), 5),
                "mean_signed_shap": round(float(sv[:, i].mean()), 5),
                "language_tag": tags[i],
            })

        per_lang[lang] = {
            "n_train": int(len(tr)), "n_test": int(len(te)), "split": how,
            "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
            "segment_attribution_pct": {k: round(v, 2) for k, v in seg.items()},
            "vocab_tag_counts": {k: int((tags == k).sum()) for k in TAGS},
        }
        print()

    # -------------------------------------------------------------
    # 2. Cross-lingual transfer matrix
    # -------------------------------------------------------------
    print("=== Cross-lingual transfer (rows = trained on, cols = tested on)")
    print("    all languages are label-balanced, so chance = 50%")
    mat = pd.DataFrame(index=langs, columns=langs, dtype=float)
    for a in langs:
        for b in langs:
            Xb = vecs[a].transform(tests[b]["text"]).toarray()
            mat.loc[a, b] = accuracy_score(tests[b]["y"], models[a].predict(Xb))
    print((mat * 100).round(1).to_string())
    mat.to_csv("multilingual_transfer_matrix.csv")

    off = [(a, b, mat.loc[a, b]) for a in langs for b in langs if a != b]
    if off:
        best = max(off, key=lambda x: x[2])
        print(f"\n    mean off-diagonal: {np.mean([x[2] for x in off]) * 100:.1f}%"
              f"   best: {best[0]} -> {best[1]} at {best[2] * 100:.1f}%")

    # -------------------------------------------------------------
    # 3. Pooled model vs the dedicated ones
    # -------------------------------------------------------------
    print("\n=== Pooled multilingual model")
    tr_all = pd.concat([split_language(data[data["lang"] == l])[0] for l in langs])
    te_all = pd.concat([split_language(data[data["lang"] == l])[1] for l in langs])
    pvec, pmodel = fit_one(tr_all, max_features=4000)
    Xp = pvec.transform(te_all["text"]).toarray()
    ppred = pmodel.predict(Xp)
    pacc = accuracy_score(te_all["y"], ppred)
    print(f"    overall accuracy {pacc:.4f} on {len(te_all)} pooled test rows")

    comp = []
    for lang in langs:
        m = te_all["lang"].values == lang
        a_pool = accuracy_score(te_all["y"].values[m], ppred[m])
        a_solo = per_lang[lang]["accuracy"]
        comp.append({"lang": lang, "dedicated": round(a_solo, 4),
                     "pooled": round(float(a_pool), 4),
                     "delta": round(float(a_pool) - a_solo, 4)})
        print(f"    {lang:10s} dedicated {a_solo:.4f}   pooled {a_pool:.4f}"
              f"   delta {a_pool - a_solo:+.4f}")
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv("multilingual_pooled_vs_dedicated.csv", index=False)

    # -------------------------------------------------------------
    pd.DataFrame(seg_rows).to_csv("multilingual_language_segment.csv", index=False)
    pd.DataFrame(top_rows).to_csv("multilingual_top_tokens.csv", index=False)
    pmodel.save_model("multilingual_pooled_model.json")
    joblib.dump(pvec, "multilingual_pooled_vectorizer.joblib")
    for lang in langs:
        models[lang].save_model(f"fakenews_{lang}_model.json")
        joblib.dump(vecs[lang], f"fakenews_{lang}_vectorizer.joblib")

    with open("multilingual_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "languages": langs,
            "per_language": per_lang,
            "transfer_matrix": {a: {b: round(float(mat.loc[a, b]), 4) for b in langs}
                                for a in langs},
            "pooled_overall_accuracy": round(float(pacc), 4),
            "pooled_vs_dedicated": comp,
        }, f, indent=2)

    print("\nSaved per-language models, transfer matrix, segment attribution,"
          " top tokens and summary.")


if __name__ == "__main__":
    main()
