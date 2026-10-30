"""
Fetches the Hindi and Telugu fake news datasets from Zenodo record
11408513, then unzips them into datasets/<lang>/.

The filenames contain a literal '&' ("Hindi_F&R_News.zip"). The first
attempt at this percent-encoded it, which Zenodo answered with 403 --
a bare '&' is legal inside a URL *path* and is what the record actually
stores. So this version tries several URL forms, newest-API first, and
reports which one worked.

MANUAL FALLBACK: if every URL still fails, just download these two in
your browser and drop the .zip files anywhere in this folder, then run
this script again -- it will find and extract them without downloading.

  https://zenodo.org/records/11408513
"""
import glob
import io
import os
import zipfile

import urllib.error
import urllib.parse
import urllib.request

REC = "11408513"
WANTED = {"hindi": "Hindi_F&R_News.zip", "telugu": "Telugu_F&R_News.zip"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "datasets")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"https://zenodo.org/records/{REC}",
}


def url_forms(fname):
    """Candidate download URLs, most likely first. The literal-'&' forms
    come first because that is what the record page's own href uses."""
    enc = urllib.parse.quote(fname)                      # '&' -> '%26'
    enc_safe = urllib.parse.quote(fname, safe="&._-")     # keeps '&'
    return [
        f"https://zenodo.org/records/{REC}/files/{fname}?download=1",
        f"https://zenodo.org/api/records/{REC}/files/{fname}/content",
        f"https://zenodo.org/record/{REC}/files/{fname}?download=1",
        f"https://zenodo.org/records/{REC}/files/{enc_safe}?download=1",
        f"https://zenodo.org/api/records/{REC}/files/{enc}/content",
        f"https://zenodo.org/records/{REC}/files/{enc}?download=1",
    ]


def try_download(fname):
    for i, url in enumerate(url_forms(fname), 1):
        try:
            print(f"    [{i}] {url[:95]}...")
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=900) as r:
                blob = r.read()
            if len(blob) < 100_000:
                print(f"        got only {len(blob)} bytes, not a dataset -- next")
                continue
            print(f"        OK, {len(blob) / 1e6:.1f} MB")
            return blob
        except urllib.error.HTTPError as e:
            print(f"        HTTP {e.code}")
        except Exception as e:
            print(f"        {type(e).__name__}: {e}")
    return None


def extract(blob_or_path, target):
    os.makedirs(target, exist_ok=True)
    z = (zipfile.ZipFile(io.BytesIO(blob_or_path))
         if isinstance(blob_or_path, bytes) else zipfile.ZipFile(blob_or_path))
    with z:
        z.extractall(target)
    print(f"    extracted to {target}")


def find_local_zip(lang):
    """Look for a manually-downloaded archive for this language."""
    for p in glob.glob(os.path.join(HERE, "**", "*.zip"), recursive=True):
        if lang.lower() in os.path.basename(p).lower():
            return p
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = []

    for lang, fname in WANTED.items():
        target = os.path.join(OUT_DIR, lang)
        if os.path.isdir(target) and os.listdir(target):
            print(f"=== {lang}: already extracted, skipping")
            continue

        local = find_local_zip(lang)
        if local:
            print(f"=== {lang}: found local archive {os.path.basename(local)}, "
                  f"extracting (no download needed)")
            try:
                extract(local, target)
                continue
            except Exception as e:
                print(f"    [!] could not unzip it: {type(e).__name__}: {e}")

        print(f"=== {lang}: downloading {fname}")
        blob = try_download(fname)
        if blob is None:
            print(f"    [!] every URL form failed for {lang}")
            failures.append(lang)
            continue
        try:
            extract(blob, target)
        except Exception as e:
            print(f"    [!] downloaded but could not unzip: {type(e).__name__}: {e}")
            failures.append(lang)

    print("\n================ WHAT WE GOT ================")
    found_any = False
    for root, _, files in os.walk(OUT_DIR):
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f"  {os.path.getsize(p):>12,}  {os.path.relpath(p, OUT_DIR)}")
            found_any = True
    if not found_any:
        print("  (nothing yet)")

    if failures:
        print(f"\n[!] Failed: {failures}")
        print("    MANUAL FALLBACK -- open this page in your browser:")
        print(f"      https://zenodo.org/records/{REC}")
        print("    click 'Download' on Hindi_F&R_News.zip and Telugu_F&R_News.zip,")
        print("    drop both .zip files into this MLSHAP folder, and run this again.")
        print("    It will find and extract them without any download.")
    else:
        print("\nAll datasets ready. Paste the listing above to Claude.")


if __name__ == "__main__":
    main()
