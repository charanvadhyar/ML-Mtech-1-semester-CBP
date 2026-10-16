"""
Downloads mBERT and IndicBERT to a local folder, so the SHAP pipeline can
load them from disk afterwards with no further network access needed.

Run this directly on your own machine (not through the Claude session) --
your machine's normal internet connection can reach huggingface.co even
though the Claude sandbox's network policy blocks it.

ai4bharat/indic-bert is a *gated* HF repo: you need an HF account that has
been granted access to it (https://huggingface.co/ai4bharat/indic-bert ->
"Agree and access repository" if you haven't already), and a token with
at least read access. This script will prompt you to paste that token
the first time it needs it.

Get a token at: https://huggingface.co/settings/tokens (a "Read" token is
enough).
"""
import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "hf_cache"))

from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "indicbert": "ai4bharat/indic-bert",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def ensure_logged_in():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nai4bharat/indic-bert needs a HuggingFace login.")
        print("Paste a token from https://huggingface.co/settings/tokens below")
        print("(a 'Read' token is enough; input is hidden if your terminal supports it):")
        import getpass
        token = getpass.getpass("HF token: ").strip()
    if token:
        login(token=token)
        print("Logged in to HuggingFace.\n")
    else:
        print("No token entered -- gated models will fail to download.\n")


def already_downloaded(target):
    return os.path.isdir(target) and any(
        f.startswith("model") or f.startswith("pytorch_model")
        for f in os.listdir(target)
    )


def download(name, hf_id):
    target = os.path.join(OUT_DIR, name)
    if already_downloaded(target):
        print(f"\n=== {hf_id} already present at {target}, skipping ===")
        return
    print(f"\n=== Downloading {hf_id} -> {target} ===")
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    os.makedirs(target, exist_ok=True)
    tok.save_pretrained(target)
    model.save_pretrained(target)
    print(f"Saved {hf_id} to {target}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    needs_login = any(
        not already_downloaded(os.path.join(OUT_DIR, name))
        for name in MODELS
    )
    if needs_login:
        ensure_logged_in()
    failures = []
    for name, hf_id in MODELS.items():
        try:
            download(name, hf_id)
        except Exception as e:
            print(f"[!] Failed to download {hf_id}: {e}")
            failures.append(hf_id)

    print("\n================ SUMMARY ================")
    if failures:
        print(f"Failed: {failures}")
        print("Everything else downloaded OK; you can re-run this script "
              "to retry just the failures (it skips ones already saved).")
    else:
        print("All models downloaded successfully.")
    print(f"Models are saved under: {OUT_DIR}")
    print("Tell Claude they're ready and it will point the pipeline at this folder.")


if __name__ == "__main__":
    main()
