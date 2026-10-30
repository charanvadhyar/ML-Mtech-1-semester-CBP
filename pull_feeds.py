"""
Pulls live news articles from RSS feeds in all five project languages, so
the trained fake news models can be run on real current news rather than
only on benchmark splits.

Uses only the standard library (urllib + ElementTree), so there is nothing
to pip install and nothing to break.

Several candidate feeds are tried per language and the script reports which
ones answered, because publisher feed URLs change often. Each item
contributes its title plus its RSS description.

Run on your own machine -- news sites are not reachable from the Claude
sandbox. Writes feed_articles.csv for Claude to analyse.
"""
import csv
import html
import os
import re
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

socket.setdefaulttimeout(30)

# Google News RSS is listed first for every language. It is the one source
# that reliably serves all five, which also removes a confound: otherwise
# each language would come from different publishers, and any cross-language
# difference could be a publisher effect rather than a language one.
# GN_TOPIC pulls top stories; GN_SEARCH widens the haul with a 7-day window.
def _gn(code, ceid):
    return [
        f"https://news.google.com/rss?hl={code}&gl=IN&ceid=IN:{ceid}",
        f"https://news.google.com/rss/search?q=when:7d&hl={code}&gl=IN&ceid=IN:{ceid}",
    ]


FEEDS = {
    "english": _gn("en-IN", "en") + [
        "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "https://indianexpress.com/section/india/feed/",
    ],
    "hindi": _gn("hi", "hi") + [
        "https://feeds.bbci.co.uk/hindi/rss.xml",
        "https://www.bhaskar.com/rss-v1--category-1061.xml",
    ],
    "tamil": _gn("ta", "ta") + [
        "https://feeds.bbci.co.uk/tamil/rss.xml",
        "https://www.dinamani.com/rss/latest-news.xml",
    ],
    "telugu": _gn("te", "te") + [
        "https://feeds.bbci.co.uk/telugu/rss.xml",
        "https://www.sakshi.com/rss.xml",
    ],
    # Every publisher feed tried for Malayalam returned HTTP errors, so
    # Google News plus a Manorama path are the candidates here.
    "malayalam": _gn("ml", "ml") + [
        "https://www.manoramaonline.com/news/kerala.rss",
        "https://www.madhyamam.com/rss/kerala",
    ],
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")


def strip_html(s):
    if not s:
        return ""
    return WS.sub(" ", html.unescape(TAG.sub(" ", s))).strip()


def fetch(url):
    ctx = ssl.create_default_context()
    # A few Indian publishers have incomplete certificate chains; this is a
    # read-only public feed pull, so fall back rather than give up on them.
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return r.read()
    except (ssl.SSLError, urllib.error.URLError) as e:
        if isinstance(e, urllib.error.HTTPError):
            raise
        lax = ssl.create_default_context()
        lax.check_hostname = False
        lax.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=30, context=lax) as r:
            return r.read()


def parse_items(blob):
    """Handle both RSS <item> and Atom <entry>."""
    root = ET.fromstring(blob)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out = []
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        get = lambda n: (it.findtext(n) or
                         it.findtext(f"atom:{n}", namespaces=ns) or "")
        title = strip_html(get("title"))
        desc = strip_html(get("description") or get("summary") or get("content"))
        link = get("link")
        if not link:
            le = it.find("atom:link", namespaces=ns)
            link = le.get("href", "") if le is not None else ""
        pub = get("pubDate") or get("published") or get("updated")
        if title or desc:
            out.append((title, desc, link.strip(), pub.strip()))
    return out


def main():
    rows, status = [], []
    for lang, urls in FEEDS.items():
        print(f"=== {lang}")
        got = 0
        for url in urls:
            try:
                items = parse_items(fetch(url))
            except Exception as e:
                print(f"    [fail] {url[:70]} -> {type(e).__name__}")
                status.append({"lang": lang, "feed": url, "ok": 0, "n": 0,
                               "error": type(e).__name__})
                continue
            src = urllib.parse.urlparse(url).netloc
            for title, desc, link, pub in items:
                text = (title + ". " + desc).strip() if desc else title
                rows.append({"lang": lang, "source": src, "title": title,
                             "summary": desc, "text": text, "url": link,
                             "published": pub})
            got += len(items)
            print(f"    [ok]   {src:<34} {len(items):>3} items")
            status.append({"lang": lang, "feed": url, "ok": 1, "n": len(items),
                           "error": ""})
        print(f"    total {got}\n")

    if not rows:
        print("No articles pulled at all -- check your internet connection.")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "feed_articles.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "source", "title", "summary",
                                          "text", "url", "published"])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(here, "feed_status.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "feed", "ok", "n", "error"])
        w.writeheader()
        w.writerows(status)

    print("================ PULLED ================")
    by = {}
    for r in rows:
        by[r["lang"]] = by.get(r["lang"], 0) + 1
    for lang in FEEDS:
        print(f"  {lang:<11} {by.get(lang, 0):>4} articles")
    print(f"\n  TOTAL {len(rows)} articles   "
          f"pulled {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"\nWrote {out}")
    print("Tell Claude it's ready and it will run the models over these.")


if __name__ == "__main__":
    main()
