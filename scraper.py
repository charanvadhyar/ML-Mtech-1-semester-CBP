

import argparse
import csv
import re
import sys
import time

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def get_video_ids_from_search(youtube, query, max_videos=5):
    """Find video IDs matching a search query. Costs 100 quota units per call."""
    video_ids = []
    request = youtube.search().list(
        part="id",
        q=query,
        type="video",
        maxResults=min(max_videos, 50),
        relevanceLanguage="ta",  # bias toward Tamil-relevant content
    )
    response = request.execute()
    for item in response.get("items", []):
        video_ids.append(item["id"]["videoId"])
    return video_ids


def fetch_comments_for_video(youtube, video_id, max_comments=500):
    """Fetch top-level comments for a single video, following pagination."""
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",  # prioritizes comments likely to have real engagement/text
            )
            response = request.execute()
        except HttpError as e:
            if e.resp.status == 403:
                print(f"  [!] Comments disabled or forbidden for video {video_id}, skipping.")
            else:
                print(f"  [!] API error on video {video_id}: {e}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "author": snippet.get("authorDisplayName", ""),
                "text": snippet.get("textDisplay", "").replace("\n", " ").strip(),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.1)  # be polite to the API

    return comments


def looks_code_mixed_or_tamil(text, tamil_range=re.compile(r"[\u0B80-\u0BFF]")):
    """
    Cheap pre-filter: keep comments that contain at least some Tamil-script
    characters, OR are short/likely-romanized (heuristic: skip pure long
    English-looking comments to reduce noise before language tagging).
    This is NOT the final language tagger -- just a filter to avoid saving
    thousands of comments that are obviously plain English (e.g. spam, bot
    comments in English only) before you do proper per-token tagging later.
    """
    has_tamil_script = bool(tamil_range.search(text))
    return has_tamil_script or True  # currently keeps everything; tighten later if needed


def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube comments for code-mixed Tamil-English data.")
    parser.add_argument("--api_key", required=True, help="YouTube Data API v3 key")
    parser.add_argument("--video_ids", nargs="*", default=[], help="Specific YouTube video IDs to scrape")
    parser.add_argument("--search", type=str, default=None, help="Search query to find videos (costs more quota)")
    parser.add_argument("--max_videos", type=int, default=5, help="Max videos to pull from search")
    parser.add_argument("--max_comments_per_video", type=int, default=500, help="Max comments per video")
    parser.add_argument("--out", type=str, default="youtube_comments.csv", help="Output CSV path")
    args = parser.parse_args()

    if not args.video_ids and not args.search:
        print("Provide either --video_ids or --search. See --help.")
        sys.exit(1)

    youtube = build("youtube", "v3", developerKey=args.api_key)

    video_ids = list(args.video_ids)
    if args.search:
        print(f"Searching for videos matching: '{args.search}'")
        found = get_video_ids_from_search(youtube, args.search, args.max_videos)
        print(f"Found {len(found)} videos.")
        video_ids.extend(found)

    all_comments = []
    for vid in video_ids:
        print(f"Fetching comments for video {vid} ...")
        comments = fetch_comments_for_video(youtube, vid, args.max_comments_per_video)
        print(f"  -> got {len(comments)} comments")
        all_comments.extend(comments)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "comment_id", "author", "text", "like_count", "published_at"])
        writer.writeheader()
        writer.writerows(all_comments)

    print(f"\nSaved {len(all_comments)} total comments to {args.out}")
    print("Next step: run the language-tagging heuristic over the 'text' column before feeding into BERT.")


if __name__ == "__main__":
    main()