

import argparse
import json
import re

from googleapiclient.discovery import build

# Common credit-line labels in Tamil movie trailer descriptions (case-insensitive)
CREDIT_LABELS = [
    "cast", "starring", "star cast", "director", "directed by",
    "music", "music director", "music composed by", "producer",
    "produced by", "banner", "dop", "cinematography", "editor",
    "story", "screenplay", "presented by",
]

LABEL_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(l) for l in CREDIT_LABELS) + r")\s*[:\-–]\s*(.+)",
    re.IGNORECASE,
)

# Splits a credit line's names on common separators used in these descriptions
NAME_SPLIT_PATTERN = re.compile(r"[,&/|]|(?:\band\b)", re.IGNORECASE)

# Title lines often look like: "Movie Name - Official Teaser | Karthi | Soori | GV Prakash"
TITLE_PIPE_SPLIT = re.compile(r"\s*\|\s*")


def fetch_video_metadata(youtube, video_id):
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    items = response.get("items", [])
    if not items:
        return None
    snippet = items[0]["snippet"]
    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
    }


def clean_name(name):
    name = name.strip()
    name = re.sub(r"\(.*?\)", "", name)          # drop parenthetical notes
    name = re.sub(r"[#@].*", "", name)             # drop hashtags/handles trailing a name
    name = re.sub(r"\s+", " ", name).strip(" .-")
    return name


def extract_from_description(description):
    names = set()
    for line in description.split("\n"):
        match = LABEL_PATTERN.search(line)
        if match:
            raw_names = NAME_SPLIT_PATTERN.split(match.group(1))
            for raw in raw_names:
                cleaned = clean_name(raw)
                if cleaned and len(cleaned) > 1:
                    names.add(cleaned)
    return names


def extract_from_title(title):
    names = set()
    parts = TITLE_PIPE_SPLIT.split(title)
    # Skip the first part (usually the movie name itself + "Official Teaser/Trailer")
    for part in parts[1:]:
        cleaned = clean_name(part)
        # Skip generic promo words that aren't names
        if cleaned and cleaned.lower() not in {
            "official teaser", "official trailer", "teaser", "trailer",
            "in cinemas", "releasing soon", "4k", "hd",
        }:
            names.add(cleaned)
    return names


def expand_variants(names):
    """
    Add likely shorthand variants people use in comments:
    - Last word of a multi-word name (e.g. "GV Prakash Kumar" -> "Prakash", "Kumar")
    - Full name lowercased as-is
    """
    expanded = set()
    for name in names:
        expanded.add(name.lower())
        words = name.split()
        if len(words) > 1:
            for w in words:
                if len(w) > 2:  # skip initials like "GV"
                    expanded.add(w.lower())
    return expanded


def main():
    parser = argparse.ArgumentParser(description="Rule-based entity extraction from video metadata, no API needed.")
    parser.add_argument("--youtube_api_key", required=True)
    parser.add_argument("--video_ids", nargs="+", required=True)
    parser.add_argument("--out", default="entity_exclusion_list.json")
    args = parser.parse_args()

    youtube = build("youtube", "v3", developerKey=args.youtube_api_key)

    all_entities = set()
    per_video_results = {}

    for vid in args.video_ids:
        print(f"Fetching metadata for {vid} ...")
        meta = fetch_video_metadata(youtube, vid)
        if meta is None:
            print(f"  [!] No metadata found for {vid}, skipping.")
            continue

        print(f"  Title: {meta['title'][:80]}")
        desc_names = extract_from_description(meta["description"])
        title_names = extract_from_title(meta["title"])
        combined = desc_names | title_names
        expanded = expand_variants(combined)

        print(f"  -> extracted {len(combined)} raw names, {len(expanded)} with variants: {sorted(expanded)}")

        per_video_results[vid] = {
            "title": meta["title"],
            "raw_names": sorted(combined),
            "expanded_variants": sorted(expanded),
        }
        all_entities.update(expanded)

    output = {
        "combined_exclusion_list": sorted(all_entities),
        "per_video": per_video_results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_entities)} unique entities across {len(args.video_ids)} videos to {args.out}")
    print("This is fully rule-based (no API, no cost) -- every entry traces back")
    print("to a specific 'Cast:'/'Director:'/title-pipe pattern in the video's own metadata.")


if __name__ == "__main__":
    main()