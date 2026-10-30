"""
Fetch a news URL, pull out the article text, and work out which language
it is in -- so the dashboard can score any live article on demand.

Standard library only, so there is nothing to install and nothing to
break mid-demo.

Language detection is by Unicode script, which is exact for the four
Indic languages here (each has its own block) and falls back to English
for Latin script. That is far more reliable than a statistical language
identifier for this particular five-way choice.
"""
import html as _html
import re
import ssl
import urllib.error
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SCRIPTS = {
    "hindi": re.compile(r"[ऀ-ॿ]"),
    "telugu": re.compile(r"[ఀ-౿]"),
    "tamil": re.compile(r"[஀-௿]"),
    "malayalam": re.compile(r"[ഀ-ൿ]"),
}

# Mean words per document in each model's training data, for the
# domain-shift warning. A live article much longer than this is outside
# what the model ever saw, which is exactly what drives the false-flag
# behaviour documented on the live-news page.
TRAIN_WORDS = {"english": 13.5, "hindi": 227.1, "malayalam": 11.3,
               "tamil": 11.2, "telugu": 141.2}

DROP_BLOCKS = re.compile(
    r"<(script|style|noscript|nav|header|footer|aside|form|svg)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")
P_BLOCK = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
H1 = re.compile(r"<h1\b[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
TITLE = re.compile(r"<title\b[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
META = re.compile(
    r"""<meta[^>]+(?:property|name)\s*=\s*["']?(?:og:)?(title|description)["']?"""
    r"""[^>]+content\s*=\s*["'](.*?)["']""", re.IGNORECASE | re.DOTALL)


def _clean(s):
    return WS.sub(" ", _html.unescape(TAG.sub(" ", s or ""))).strip()


def fetch_html(url, timeout=25):
    """Fetch a page. Falls back to an unverified context for the several
    Indian publishers with incomplete certificate chains -- this is a
    read-only fetch of a public news page."""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-IN,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError:
        raise
    except (ssl.SSLError, urllib.error.URLError):
        lax = ssl.create_default_context()
        lax.check_hostname = False
        lax.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=lax) as r:
            raw = r.read()

    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_article(doc):
    """Return (title, text). Paragraph tags first, since that is where news
    bodies live; falls back to whole-page text when a site renders its body
    some other way."""
    body = DROP_BLOCKS.sub(" ", doc)

    metas = {k.lower(): _clean(v) for k, v in META.findall(doc)}
    title = ""
    for cand in (H1.search(body), TITLE.search(doc)):
        if cand:
            title = _clean(cand.group(1))
            if title:
                break
    if not title:
        title = metas.get("title", "")

    paras = [_clean(p) for p in P_BLOCK.findall(body)]
    paras = [p for p in paras if len(p.split()) >= 4]
    text = " ".join(paras)

    # Only fall back when the paragraphs gave us almost nothing. Real news
    # bodies run to hundreds of words, so this threshold is deliberately low
    # -- being too eager here drags in navigation text and duplicates the
    # headline, which then shows up as spurious tokens in the explanation.
    if len(text.split()) < 15:
        flat = _clean(body)
        desc = metas.get("description", "")
        text = max(text, flat, key=lambda s: len(s.split()))
        if desc and desc not in text:
            text = (desc + " " + text).strip()

    # Drop repeats of the headline so it is not counted two or three times.
    if title:
        text = text.replace(title, " ", 2) if text.count(title) > 1 else text
        if text.startswith(title):
            text = text[len(title):]
    return title, WS.sub(" ", text).strip()


def detect_language(text):
    """Pick the language by Unicode script. Returns (lang, confidence) where
    confidence is the share of letters in that script."""
    if not text:
        return "english", 0.0
    counts = {l: len(p.findall(text)) for l, p in SCRIPTS.items()}
    latin = len(re.findall(r"[A-Za-z]", text))
    best = max(counts, key=counts.get)
    total = sum(counts.values()) + latin
    if total == 0:
        return "english", 0.0
    if counts[best] == 0 or counts[best] < 0.10 * total:
        return "english", latin / total
    return best, counts[best] / total


def shift_ratio(lang, text):
    """How many times longer this article is than the model's training
    documents. Above about 2x, the prediction is outside the model's
    experience and should be treated as unreliable."""
    n = len(text.split())
    base = TRAIN_WORDS.get(lang)
    if not base or not n:
        return None, n
    return round(n / base, 2), n


def load_url(url):
    """One call: fetch, extract, detect. Returns a dict."""
    doc = fetch_html(url)
    title, text = extract_article(doc)
    lang, conf = detect_language(title + " " + text)
    ratio, n_words = shift_ratio(lang, text)
    return {"url": url, "title": title, "text": text, "lang": lang,
            "script_confidence": round(conf, 3), "n_words": n_words,
            "shift_ratio": ratio}
