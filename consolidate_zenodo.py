"""
Collapses the Zenodo Hindi and Telugu archives into one CSV per language.

Those archives ship one article per .txt file -- about 20,600 Hindi and
20,000 Telugu files, with the label carried by the folder name
(Hindi_fake_news / Hindi_real_news). That is far too many files to move
one at a time, so this reads them locally and writes a single tidy CSV
per language:

    text, label ('fake' / 'real'), y (1 = fake)

Note these are full news ARTICLES, not headlines or comments, so they are
much longer than the other languages in this project. That difference is
reported below and is kept in mind when the models are compared.

Run on your own machine, from the MLSHAP folder.
"""
import csv
import os
import re
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(HERE, "datasets")
SCRIPTS = {"hindi": r"[ऀ-ॿ]", "telugu": r"[ఀ-౿]"}
WS = re.compile(r"\s+")


def read_text(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def collect(lang):
    base = os.path.join(DATASETS, lang)
    if not os.path.isdir(base):
        print(f"=== {lang}: {base} not found, skipping")
        return None

    rows, skipped = [], 0
    for root, dirs, files in os.walk(base):
        # __MACOSX holds resource-fork junk ('._name.txt'), never content.
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        low_root = root.lower()
        if "fake" in low_root:
            label, y = "fake", 1
        elif "real" in low_root or "true" in low_root:
            label, y = "real", 0
        else:
            continue

        for fn in files:
            if fn.startswith("._") or not fn.lower().endswith(".txt"):
                continue
            txt = WS.sub(" ", read_text(os.path.join(root, fn))).strip()
            if len(txt) < 20:
                skipped += 1
                continue
            rows.append((txt, label, y))

    if not rows:
        print(f"=== {lang}: no usable .txt files found")
        return None

    out = os.path.join(HERE, f"{lang}_fakenews.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "y"])
        w.writerows(rows)

    n_fake = sum(1 for r in rows if r[2] == 1)
    native = re.compile(SCRIPTS[lang])
    n_native = sum(1 for r in rows[:4000] if native.search(r[0]))
    n_latin = sum(1 for r in rows[:4000] if re.search(r"[A-Za-z]", r[0]))
    sample = min(4000, len(rows))
    mean_words = sum(len(r[0].split()) for r in rows[:sample]) / sample

    print(f"=== {lang}: {len(rows):,} articles "
          f"({n_fake:,} fake / {len(rows) - n_fake:,} real), {skipped} too short")
    print(f"    native script {100 * n_native / sample:.0f}%  "
          f"latin {100 * n_latin / sample:.0f}%  "
          f"mean words {mean_words:.0f}")
    print(f"    wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


def main():
    print("Consolidating Zenodo per-article archives into one CSV per language.\n")
    made = [p for p in (collect(l) for l in SCRIPTS) if p]
    print("\n================ DONE ================")
    for p in made:
        print(f"  {os.path.getsize(p):>12,}  {os.path.basename(p)}")
    if made:
        print("\nTell Claude these are ready and it will pull them in for training.")
    else:
        print("\nNothing produced -- check that datasets\\hindi and datasets\\telugu exist.")


if __name__ == "__main__":
    main()
