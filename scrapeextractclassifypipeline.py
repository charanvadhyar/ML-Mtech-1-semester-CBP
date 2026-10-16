"""
End-to-End Code-Mixed Language Tagging Pipeline (Fully Local, Rule-Based)
============================================================================
No paid API required. Everything derives iteratively from your own data:

  1. (Optional) Extract named entities from video titles/descriptions via
     regex on credit-line patterns ("Cast:", "Director:", "Music:", etc.)
     -- needs a free YouTube Data API key. Skip this with --skip_metadata
     if you don't have video metadata; the pipeline still works using only
     the statistical method below.
  2. Statistically validate/clean that entity list against real comment
     frequency (drops false positives like generic company-descriptor
     words; drops low-confidence, cross-video-generic words).
  3. Iteratively closes gaps: adds back statistically-confident names the
     extractor missed (non-dictionary words only, to stay safe), flags
     genuinely ambiguous English-word/name overlaps for a quick human call
     instead of guessing.
  4. Automatically derives short-token classification (Tamil particles vs
     leaked name initials vs genuine short English words) straight from
     the dataset -- no hand-typed particle list.
  5. Tags every token in every comment: PROPER_NOUN / TA-script / EN /
     TANGLISH? / TA-particle / NOISE / SHORT-AMBIG.
  6. Outputs a final CSV with per-comment language-mix percentages, AND a
     standalone color-coded HTML report showing every comment with each
     word highlighted by category.

SETUP:
  pip install pandas scikit-learn pyspellchecker
  pip install google-api-python-client   # only needed if NOT using --skip_metadata

USAGE (with metadata extraction):
  python full_pipeline.py --comments_csv comments.csv \
      --youtube_api_key YOUR_KEY \
      --video_ids jpvb4l2RIXU iOMvWcMo8cA SFrKBnj13qI LnmF2vX8Dfo \
      --out_prefix output/tanglish_analysis

USAGE (skip metadata, pure statistical method):
  python full_pipeline.py --comments_csv comments.csv --skip_metadata \
      --out_prefix output/tanglish_analysis
"""

import argparse
import html
import json
import os
import re
import string
from collections import Counter, defaultdict

import pandas as pd
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

spell = SpellChecker()

TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
EMOJI_RANGE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u2000-\u206F"
    "\uFE0F"
    "]+"
)

CREDIT_LABELS = [
    "cast", "starring", "star cast", "director", "directed by",
    "music", "music director", "music composed by", "producer",
    "produced by", "banner", "dop", "cinematography", "editor",
    "story", "screenplay", "presented by",
]
LABEL_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(l) for l in CREDIT_LABELS) + r")\s*[:\-\u2013]\s*(.+)",
    re.IGNORECASE,
)
NAME_SPLIT_PATTERN = re.compile(r"[,&/|]|(?:\band\b)", re.IGNORECASE)
TITLE_PIPE_SPLIT = re.compile(r"\s*\|\s*")

COMPANY_DESCRIPTOR_NOISE = {
    "arts", "factory", "film", "infotainment", "entertainments",
    "others", "trident", "eywa", "ak", "rs", "ync",
}

CATEGORY_COLORS = {
    "PROPER_NOUN": "#D9C2F0",
    "TA-script": "#FFD9A0",
    "EN": "#B8D9F5",
    "TANGLISH?": "#FFB3A7",
    "TA-particle": "#C9F0D9",
    "PROPER_NOUN_CANDIDATE": "#E6D2F5",
    "NOISE": "#E0E0E0",
    "SHORT/AMBIG": "#F0F0C0",
}


# ---------------------------------------------------------------------------
# STEP 1: Rule-based entity extraction from video metadata (optional)
# ---------------------------------------------------------------------------
def clean_name(name):
    name = name.strip()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[#@].*", "", name)
    name = re.sub(r"\s+", " ", name).strip(" .-")
    return name


def extract_from_description(description):
    names = set()
    for line in description.split("\n"):
        match = LABEL_PATTERN.search(line)
        if match:
            for raw in NAME_SPLIT_PATTERN.split(match.group(1)):
                cleaned = clean_name(raw)
                if cleaned and len(cleaned) > 1:
                    names.add(cleaned)
    return names


def extract_from_title(title):
    names = set()
    parts = TITLE_PIPE_SPLIT.split(title)
    for part in parts[1:]:
        cleaned = clean_name(part)
        if cleaned and cleaned.lower() not in {
            "official teaser", "official trailer", "teaser", "trailer",
            "in cinemas", "releasing soon", "4k", "hd",
        }:
            names.add(cleaned)
    return names


def expand_variants(names):
    expanded = set()
    for name in names:
        expanded.add(name.lower())
        words = name.split()
        if len(words) > 1:
            for w in words:
                if len(w) > 2:
                    expanded.add(w.lower())
    return expanded


def fetch_and_extract_entities(youtube_api_key, video_ids):
    from googleapiclient.discovery import build
    youtube = build("youtube", "v3", developerKey=youtube_api_key)

    all_entities = set()
    for vid in video_ids:
        request = youtube.videos().list(part="snippet", id=vid)
        response = request.execute()
        items = response.get("items", [])
        if not items:
            continue
        snippet = items[0]["snippet"]
        title, description = snippet.get("title", ""), snippet.get("description", "")
        combined = extract_from_description(description) | extract_from_title(title)
        all_entities.update(expand_variants(combined))
        print(f"  [{vid}] {title[:70]}  -> {len(combined)} raw names")

    return all_entities


# ---------------------------------------------------------------------------
# Shared tokenization helpers
# ---------------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = EMOJI_RANGE.sub(" ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"^\d+:\d+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def raw_tokenize(text):
    return [t.strip(string.punctuation + "\u2019\u2018\u201c\u201d") for t in clean_text(text).split()]


def build_concentration_table(df, min_count=5):
    """Per-token frequency and cross-video concentration, computed fresh from this dataset."""
    token_video_counts = defaultdict(lambda: defaultdict(int))
    token_case_counts = defaultdict(Counter)

    for _, row in df.iterrows():
        video_id = row["video_id"]
        for raw_tok in raw_tokenize(row["text"]):
            tok = raw_tok.strip(string.punctuation)
            if not tok or not tok.isalpha():
                continue
            lower = tok.lower()
            token_video_counts[lower][video_id] += 1
            token_case_counts[lower][tok] += 1

    rows = []
    for tok, video_counts in token_video_counts.items():
        total = sum(video_counts.values())
        if total < min_count:
            continue
        concentration = max(video_counts.values()) / total
        most_common_case = token_case_counts[tok].most_common(1)[0][0]
        is_dict_word = bool(spell.known([tok]))
        rows.append({
            "token": tok,
            "total_count": total,
            "n_videos_present": len(video_counts),
            "concentration": round(concentration, 2),
            "capitalized": most_common_case[0].isupper() if most_common_case else False,
            "is_dictionary_word": is_dict_word,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STEP 2 + 3: Build and iteratively refine the final exclusion list
# ---------------------------------------------------------------------------
def build_final_exclusion_list(df, extracted_entities, concentration_threshold=0.85):
    candidates = build_concentration_table(df)

    dropped = {}
    kept = set()
    for entity in extracted_entities:
        is_multiword = " " in entity or "." in entity
        if entity in COMPANY_DESCRIPTOR_NOISE:
            dropped[entity] = "company descriptor fragment"
            continue
        row = candidates[candidates["token"] == entity]
        if not is_multiword and not row.empty and row.iloc[0]["concentration"] < 0.75:
            dropped[entity] = f"low concentration in real comments ({row.iloc[0]['concentration']:.2f})"
            continue
        kept.add(entity)

    # Iteratively add back high-confidence statistical candidates the
    # extractor missed entirely -- but only the safe (non-dictionary) ones.
    missed = candidates[
        (candidates["concentration"] >= concentration_threshold)
        & (candidates["capitalized"] == True)
        & (candidates["total_count"] >= 5)
        & (~candidates["token"].isin(kept))
    ]
    safe_to_add = set(missed[~missed["is_dictionary_word"]]["token"])
    flagged_for_review = missed[missed["is_dictionary_word"]][["token", "total_count", "concentration"]]

    final_list = kept | safe_to_add
    return final_list, dropped, safe_to_add, flagged_for_review, candidates


# ---------------------------------------------------------------------------
# STEP 4: Automated short-token classifier
# ---------------------------------------------------------------------------
def build_short_token_classifier(df, exclusion_set, min_count=5, concentration_threshold=0.6):
    token_video_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        video_id = row["video_id"]
        for tok in raw_tokenize(row["text"]):
            lower = tok.lower()
            if not lower or len(lower) > 3 or not lower.isalpha():
                continue
            if lower in ENGLISH_STOP_WORDS or lower in exclusion_set:
                continue
            if TAMIL_RANGE.search(tok):
                continue
            token_video_counts[lower][video_id] += 1

    classification = {}
    for tok, video_counts in token_video_counts.items():
        total = sum(video_counts.values())
        if total < min_count:
            continue
        if spell.known([tok]):
            classification[tok] = "EN"
            continue
        concentration = max(video_counts.values()) / total
        classification[tok] = "TA-particle" if concentration < concentration_threshold else "PROPER_NOUN_CANDIDATE"

    return classification


# ---------------------------------------------------------------------------
# STEP 5: Token classification + full tagging
# ---------------------------------------------------------------------------
def classify_token(token, exclusion_set, short_classifier):
    tok = token.strip(string.punctuation + "\u2019\u2018\u201c\u201d")
    if not tok:
        return None
    lower = tok.lower()

    if lower in exclusion_set:
        return "PROPER_NOUN"
    if TAMIL_RANGE.search(tok):
        return "TA-script"
    if lower in ENGLISH_STOP_WORDS:
        return "EN"
    if not lower.isalpha():
        return "NOISE"
    if len(lower) <= 3 and lower in short_classifier:
        return short_classifier[lower]
    if len(lower) <= 2:
        return "SHORT/AMBIG"
    if spell.known([lower]):
        return "EN"
    return "TANGLISH?"


def tag_comment(text, exclusion_set, short_classifier):
    return [
        (t, classify_token(t, exclusion_set, short_classifier))
        for t in raw_tokenize(text)
        if classify_token(t, exclusion_set, short_classifier)
    ]


# ---------------------------------------------------------------------------
# STEP 6: Color-coded HTML rendering
# ---------------------------------------------------------------------------
def render_comment_html(tags):
    spans = []
    for tok, label in tags:
        color = CATEGORY_COLORS.get(label, "#FFFFFF")
        spans.append(
            f'<span title="{label}" style="background-color:{color};padding:1px 3px;'
            f'border-radius:3px;margin:1px;display:inline-block;">{html.escape(tok)}</span>'
        )
    return " ".join(spans)


def build_html_report(comment_df, path):
    legend_items = "".join(
        f'<span style="background-color:{color};padding:2px 8px;border-radius:3px;margin-right:12px;">{label}</span>'
        for label, color in CATEGORY_COLORS.items()
    )
    rows_html = "\n".join(
        f'<div style="margin-bottom:10px;padding:8px;border:1px solid #ddd;border-radius:4px;">'
        f'<div style="font-size:12px;color:#888;">video: {r["video_id"]} | author: {html.escape(str(r["author"]))} | likes: {r["like_count"]}</div>'
        f'<div style="margin-top:4px;">{r["colored_html"]}</div>'
        f"</div>"
        for _, r in comment_df.iterrows()
    )
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Language Tagging Report</title></head>
<body style="font-family:sans-serif;max-width:900px;margin:20px auto;">
<h2>Code-Mixed Language Tagging - Color-Coded Report</h2>
<div style="margin-bottom:20px;">{legend_items}</div>
{rows_html}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="End-to-end local, rule-based code-mixed tagging pipeline.")
    parser.add_argument("--comments_csv", required=True)
    parser.add_argument("--youtube_api_key", default=None)
    parser.add_argument("--video_ids", nargs="*", default=[])
    parser.add_argument("--skip_metadata", action="store_true", help="Skip video metadata extraction, use statistics only.")
    parser.add_argument("--entities_json", default=None,
                         help="Path to a pre-built entity-extraction JSON (e.g. produced by "
                              "rulebasedextractor.py) containing a 'combined_exclusion_list' field. "
                              "When given, this is used directly instead of calling the YouTube API "
                              "again or falling back to pure statistics.")
    parser.add_argument("--out_prefix", default="tanglish_output")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    df = pd.read_csv(args.comments_csv)
    df["text"] = df["text"].fillna("")
    print(f"Loaded {len(df)} comments across {df['video_id'].nunique()} videos.\n")

    extracted_entities = set()
    if args.entities_json:
        print(f"STEP 1: Loading pre-extracted named entities from {args.entities_json} ...")
        with open(args.entities_json, "r", encoding="utf-8") as f:
            entities_data = json.load(f)
        extracted_entities = set(entities_data.get("combined_exclusion_list", []))
        print(f"  -> {len(extracted_entities)} entity candidates loaded from file\n")
    elif not args.skip_metadata:
        if not args.youtube_api_key or not args.video_ids:
            print("[!] --youtube_api_key and --video_ids required unless --skip_metadata or --entities_json is set. Skipping metadata step.")
        else:
            print("STEP 1: Extracting named entities from video metadata (rule-based, no LLM)...")
            extracted_entities = fetch_and_extract_entities(args.youtube_api_key, args.video_ids)
            print(f"  -> {len(extracted_entities)} raw entity candidates\n")

    print("STEP 2-3: Statistically validating and iteratively refining exclusion list...")
    final_exclusion, dropped, auto_added, flagged, candidates = build_final_exclusion_list(df, extracted_entities)
    print(f"  Dropped {len(dropped)} false positives: {sorted(dropped)}")
    print(f"  Auto-added {len(auto_added)} statistically-confident names missed by extraction: {sorted(auto_added)}")
    if not flagged.empty:
        print(f"  Flagged {len(flagged)} ambiguous English-word/name overlaps for human review:")
        print(flagged.to_string(index=False))
    print(f"  Final exclusion list: {len(final_exclusion)} entries\n")

    print("STEP 4: Auto-deriving short-token classification from this dataset...")
    short_classifier = build_short_token_classifier(df, final_exclusion)
    print(f"  {len(short_classifier)} short tokens classified automatically.\n")

    print("STEP 5-6: Tagging every comment and building color-coded output...")
    comment_rows = []
    for _, row in df.iterrows():
        tags = tag_comment(row["text"], final_exclusion, short_classifier)
        counts = Counter(label for _, label in tags)
        total = sum(counts.values())
        pct_row = {f"{label}_pct": round(100 * c / total, 1) if total else None for label, c in counts.items()}
        pct_row["n_tokens"] = total

        # Three word-list columns: the actual words behind each core category,
        # so the CSV shows not just "60% Tamil" but exactly which words.
        pct_row["english_words"] = ", ".join(t for t, label in tags if label == "EN")
        pct_row["tamil_words"] = ", ".join(t for t, label in tags if label == "TA-script")
        pct_row["tanglish_words"] = ", ".join(t for t, label in tags if label == "TANGLISH?")

        pct_row["colored_html"] = render_comment_html(tags)
        comment_rows.append(pct_row)

    result_df = pd.concat([df, pd.DataFrame(comment_rows)], axis=1)

    csv_path = f"{args.out_prefix}_tagged.csv"
    html_path = f"{args.out_prefix}_colorcoded.html"
    exclusion_path = f"{args.out_prefix}_exclusion_list.json"

    # Drop the raw html column from CSV (kept only in the HTML report) to keep the CSV clean
    result_df.drop(columns=["colored_html"]).to_csv(csv_path, index=False)
    build_html_report(result_df, html_path)
    with open(exclusion_path, "w", encoding="utf-8") as f:
        json.dump({
            "final_exclusion_list": sorted(final_exclusion),
            "dropped": dropped,
            "auto_added_from_statistics": sorted(auto_added),
            "flagged_for_human_review": flagged.to_dict(orient="records"),
            "auto_derived_short_token_classes": short_classifier,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Outputs:")
    print(f"  {csv_path}          -- per-comment language-mix percentages")
    print(f"  {html_path}   -- word-by-word color-coded report, open in a browser")
    print(f"  {exclusion_path}  -- full audit trail of every automated decision")
    print("\nRe-run this on a fresh scrape and every list re-derives itself from scratch.")


if __name__ == "__main__":
    main()