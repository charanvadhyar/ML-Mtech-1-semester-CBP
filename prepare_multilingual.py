"""
Normalises every available fake news dataset into one schema:
    lang, text, label ('fake' / 'real'), y (1 = fake)

Each language comes from a different source with a different format and a
different *kind* of text, which matters for interpretation:

  malayalam  DravidianLangTech EACL-2024 Task 1   social media comments, code-mixed
  tamil      TamilFakeAndReal / news headlines    news headlines, near-monolingual script
  english    IFND (Indian Fake News Dataset)      Indian English news statements
  hindi      Zenodo 11408513                      news articles
  telugu     Zenodo 11408513                      news articles

Because the sources differ, each language is balanced to equal fake/real
counts here. Without that, a pooled multilingual model could score well
purely by identifying the language and exploiting that source's label
prior -- learning nothing about credibility. Balancing removes that
shortcut so any pooled result is about the text, not the provenance.

Languages whose files are absent are skipped with a note, so this runs
before and after the Zenodo download.
"""
import glob
import json
import os
import re

import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

SCRIPT_RANGES = {
    "tamil": r"[஀-௿]",
    "malayalam": r"[ഀ-ൿ]",
    "hindi": r"[ऀ-ॿ]",
    "telugu": r"[ఀ-౿]",
    "english": None,
}


def balance(df, seed=42):
    """Downsample the majority class so fake/real are equal.

    Iterates the groups directly rather than using groupby().apply(),
    which in pandas 3 no longer passes the grouping column through.
    """
    n = int(df["y"].value_counts().min())
    parts = [g.sample(n=n, random_state=seed) for _, g in df.groupby("y")]
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def clean(df):
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 5]
    return df.drop_duplicates(subset="text")


# ---------------------------------------------------------------------------
# Per-language loaders. Each returns a frame with text + y, or None.
# ---------------------------------------------------------------------------
def load_malayalam():
    paths = [os.path.join(RAW, f) for f in
             ["Task-1_trainset.csv", "Task-1_devset.csv", "Task-1-testset.csv"]]
    if not all(os.path.exists(p) for p in paths):
        return None
    # Keep the official split boundaries so the per-language model can
    # still be scored the way the shared task intends.
    frames = []
    for p, split in zip(paths, ["train", "dev", "test"]):
        d = pd.read_csv(p)
        d["label"] = d["label"].astype(str).str.strip().str.lower()
        d = d[d["label"].isin(["fake", "original"])]
        d["y"] = (d["label"] == "fake").astype(int)
        d["split"] = split
        frames.append(d[["text", "y", "split"]])
    return pd.concat(frames, ignore_index=True)


def load_tamil():
    p = os.path.join(RAW, "TamilFakeAndReal.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d.rename(columns={"Tamil": "text", "Is Fake": "y"})
    return d[["text", "y"]]


def load_english():
    p = os.path.join(RAW, "IFND.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, encoding="latin-1")
    d = d.rename(columns={"Statement": "text"})
    d["y"] = (d["Label"].astype(str).str.strip().str.lower() == "fake").astype(int)
    return d[["text", "y"]]


def _load_zenodo(lang):
    """Hindi / Telugu from Zenodo record 11408513.

    Those archives ship one full article per .txt file (~20,600 Hindi,
    ~20,000 Telugu) with the label in the folder name. consolidate_zenodo.py
    collapses them into a CSV, and the text is then truncated to the
    article LEAD (first 1200 characters).

    The truncation is deliberate and worth stating in any write-up: the
    full articles average 702 words (Hindi) and 300 (Telugu), against 11
    for the Malayalam comments and ~13 for the Tamil and English
    headlines. Comparing a 702-word article against an 11-word comment
    measures document length as much as credibility. The lead keeps the
    most informative part of a news article while narrowing that gap;
    the full-text archives remain on disk if a full-length run is wanted.
    """
    path = os.path.join(RAW, f"{lang}_fakenews_lead.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "text" not in d.columns:
        print(f"    [!] {os.path.basename(path)} has no 'text' column: {list(d.columns)}")
        return None
    if "y" in d.columns:
        d["y"] = pd.to_numeric(d["y"], errors="coerce")
    elif "label" in d.columns:
        d["y"] = (d["label"].astype(str).str.strip().str.lower() == "fake").astype(int)
    else:
        return None
    print(f"    [ok] lead-truncated articles from {os.path.basename(path)}")
    return d[["text", "y"]].dropna(subset=["y"])


LOADERS = {
    "malayalam": load_malayalam,
    "tamil": load_tamil,
    "english": load_english,
    "hindi": lambda: _load_zenodo("hindi"),
    "telugu": lambda: _load_zenodo("telugu"),
}


def main():
    out, report = [], {}
    for lang, loader in LOADERS.items():
        print(f"=== {lang}")
        try:
            df = loader()
        except Exception as e:
            print(f"    [!] loader failed: {type(e).__name__}: {e}")
            df = None
        if df is None or df.empty:
            print("    not available -- skipped")
            report[lang] = {"status": "missing"}
            continue

        df = clean(df.dropna(subset=["text", "y"]))
        df["y"] = df["y"].astype(int)
        raw_counts = df["y"].value_counts().to_dict()
        has_split = "split" in df.columns

        # Balance within language (and within split, where splits exist, so
        # the official test split stays a valid held-out set).
        if has_split:
            df = pd.concat([balance(g).assign(split=s)
                            for s, g in df.groupby("split")], ignore_index=True)
        else:
            df = balance(df)

        df["lang"] = lang
        df["label"] = df["y"].map({1: "fake", 0: "real"})
        if not has_split:
            df["split"] = "pool"

        rng = SCRIPT_RANGES[lang]
        native = (100 * df["text"].str.contains(rng, regex=True).mean()) if rng else 0.0
        latin = 100 * df["text"].str.contains(r"[A-Za-z]", regex=True).mean()
        mixed = (100 * (df["text"].str.contains(rng, regex=True)
                        & df["text"].str.contains(r"[A-Za-z]", regex=True)).mean()) if rng else 0.0

        print(f"    {len(df)} rows after balancing (raw {raw_counts})")
        print(f"    native script {native:.0f}%  latin {latin:.0f}%  both {mixed:.0f}%"
              f"  | mean words {df['text'].str.split().str.len().mean():.1f}")

        report[lang] = {
            "status": "ok",
            "n_balanced": int(len(df)),
            "raw_label_counts": {str(k): int(v) for k, v in raw_counts.items()},
            "pct_native_script": round(float(native), 1),
            "pct_latin": round(float(latin), 1),
            "pct_code_mixed": round(float(mixed), 1),
            "mean_words": round(float(df["text"].str.split().str.len().mean()), 1),
            "has_official_splits": bool(has_split),
        }
        out.append(df[["lang", "text", "label", "y", "split"]])

    if not out:
        print("\nNo datasets available at all -- nothing written.")
        return

    allx = pd.concat(out, ignore_index=True)
    allx.to_csv("multilingual_fakenews.csv", index=False)
    with open("multilingual_dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================ COMBINED ================")
    print(allx.groupby(["lang", "label"]).size().unstack(fill_value=0).to_string())
    print(f"\nTotal {len(allx)} rows across {allx['lang'].nunique()} languages")
    print("Saved multilingual_fakenews.csv, multilingual_dataset_report.json")


if __name__ == "__main__":
    main()
