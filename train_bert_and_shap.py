"""
Part B - Neural half: fine-tune mBERT / IndicBERT (loaded from local disk,
no network needed) on the same balanced DravidianCodeMix Tamil-English
binary sentiment task as the classical TF-IDF+XGBoost model, then compute
PartitionSHAP (shap's own default explainer for text pipelines) on it.

Part C - Language-Segment SHAP Attribution: tag every token the way we
tagged the YouTube pipeline/vocab (TAMIL_RANGE unicode + English dictionary
+ "TANGLISH?" fallback) and aggregate PartitionSHAP's |value| mass per tag,
exactly as we already did for TreeSHAP on the classical model -- so the
two can be compared side by side.

Run this on your own machine (python train_bert_and_shap.py --model mbert),
where models/mbert and models/indicbert already exist locally.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spellchecker import SpellChecker
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TAMIL_RANGE = re.compile(r"[஀-௿]")
spell = SpellChecker()


def tag_term(term: str) -> str:
    if TAMIL_RANGE.search(term):
        return "TA-script"
    if term.lower() in ENGLISH_STOP_WORDS:
        return "EN"
    if not term.isalpha():
        return "NOISE"
    if spell.known([term.lower()]):
        return "EN"
    return "TANGLISH?"


def build_word_tag_map(row):
    """From our OWN pipeline's per-comment output (english_words/tamil_words/
    tanglish_words columns): which words were tagged EN / TA-script / TANGLISH?
    This is the real, entity-exclusion-aware tag -- preferred over tag_term()
    wherever it's available."""
    m = {}
    for col, tag in [("english_words", "EN"), ("tamil_words", "TA-script"), ("tanglish_words", "TANGLISH?")]:
        val = row.get(col, "")
        if val != val or not str(val).strip():  # NaN-safe
            continue
        for w in str(val).split(", "):
            w = w.strip().lower()
            if w:
                m[w] = tag
    return m


def tag_token(token, word_tag_map):
    token_clean = token.strip().lower()
    if token_clean in word_tag_map:
        return word_tag_map[token_clean]
    return tag_term(token.strip())


def apply_to_scraped(model_name, predict_proba, tokenizer, out_prefix,
                      scraped_csv, scraped_tagged_csv, n_per_video):
    import csv as csv_module
    csv_module.field_size_limit(10**7)

    comments = pd.read_csv(scraped_csv)
    comments["text"] = comments["text"].fillna("")
    tagged = pd.read_csv(scraped_tagged_csv, engine="python").iloc[: len(comments)].reset_index(drop=True)

    sample_idx = sorted(
        comments.groupby("video_id", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_video, len(g)), random_state=42))
        .index
    )
    sample_texts = comments.iloc[sample_idx]["text"].tolist()
    print(f"\n[{model_name}] Running inference + PartitionSHAP on "
          f"{len(sample_texts)} REAL scraped comments across "
          f"{comments['video_id'].nunique()} videos ...")

    probs = predict_proba(sample_texts)
    preds = np.where(probs[:, 1] >= 0.5, "Positive", "Negative")

    explainer = shap.Explainer(predict_proba, tokenizer)
    shap_values = explainer(sample_texts)

    rows = []
    for pos, orig_idx in enumerate(sample_idx):
        tokens = shap_values.data[pos]
        vals = shap_values.values[pos][:, 1]
        word_tag_map = build_word_tag_map(tagged.iloc[orig_idx])
        tags = [tag_token(t, word_tag_map) for t in tokens]
        abs_vals = np.abs(vals)
        total = abs_vals.sum()
        row = {
            "video_id": comments.iloc[orig_idx]["video_id"],
            "text": sample_texts[pos],
            "predicted_sentiment": preds[pos],
        }
        for tag in ["EN", "TA-script", "TANGLISH?", "NOISE"]:
            mask = [t == tag for t in tags]
            row[f"{tag}_shap_pct"] = round(100 * abs_vals[mask].sum() / total, 2) if total > 0 else 0.0
        rows.append(row)

    per_instance = pd.DataFrame(rows)
    per_instance.to_csv(f"{out_prefix}_scraped_language_segment_shap.csv", index=False)

    pct_cols = [c for c in per_instance.columns if c.endswith("_shap_pct")]
    print(f"\n[{model_name}] Dataset-level language-segment trend on REAL scraped comments:")
    print(per_instance[pct_cols].mean().round(2).sort_values(ascending=False))
    print(f"\n[{model_name}] Split by video:")
    print(per_instance.groupby("video_id")[pct_cols].mean().round(2))

    with open(f"{out_prefix}_scraped_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "n_instances": len(per_instance),
            "overall_mean_pct": per_instance[pct_cols].mean().round(2).to_dict(),
            "by_video": per_instance.groupby("video_id")[pct_cols].mean().round(2).to_dict(),
        }, f, indent=2)
    print(f"[{model_name}] Saved: {out_prefix}_scraped_language_segment_shap.csv, "
          f"{out_prefix}_scraped_summary.json")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=48):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()} | {"labels": self.labels[idx]}


def fine_tune(model, tokenizer, train_df, epochs, batch_size, lr, device):
    ds = SentimentDataset(train_df["text"].tolist(), train_df["y"].tolist(), tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Linear warmup then linear decay: ALBERT-style models (IndicBERT) are
    # unstable with a flat LR and often collapse to chance without this.
    total_steps = epochs * len(dl)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    model.train()
    for epoch in range(epochs):
        total_loss, correct, seen = 0.0, 0, 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            total_loss += out.loss.item()
            correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
            seen += len(batch["labels"])
        print(f"  epoch {epoch + 1}/{epochs}  mean loss = {total_loss / len(dl):.4f}"
              f"  train acc = {correct / seen:.3f}")
    model.eval()
    return model


def evaluate(model, tokenizer, test_df, device, batch_size=16):
    ds = SentimentDataset(test_df["text"].tolist(), test_df["y"].tolist(), tokenizer)
    dl = DataLoader(ds, batch_size=batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dl:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mbert", "indicbert"])
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--train_csv", default="dravidian_sentiment_train.csv")
    parser.add_argument("--test_csv", default="dravidian_sentiment_test.csv")
    parser.add_argument("--n_train", type=int, default=6000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_shap", type=int, default=150, help="How many test comments to run PartitionSHAP on.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out_prefix", default=None)
    parser.add_argument("--apply_to_scraped", action="store_true", default=True,
                         help="Also run inference + PartitionSHAP on our own real scraped YouTube comments.")
    parser.add_argument("--no_apply_to_scraped", dest="apply_to_scraped", action="store_false")
    parser.add_argument("--scraped_csv", default="comments.csv")
    parser.add_argument("--scraped_tagged_csv", default="output/tanglish_tagged.csv")
    parser.add_argument("--n_scraped_per_video", type=int, default=40,
                         help="PartitionSHAP on real text is slow on CPU but fine on GPU.")
    args = parser.parse_args()

    out_prefix = args.out_prefix or f"bert_{args.model}"
    model_path = f"{args.models_dir}/{args.model}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading {args.model} from {model_path} (local, no network) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)

    # Tokenizer sanity check: a broken tokenizer (e.g. built via a fallback
    # extractor) shows up as a high [UNK] rate or nonsense pieces.
    probe = "Semma mass padam da, thalaiva vera level acting! படம் சூப்பர்"
    probe_ids = tokenizer(probe)["input_ids"]
    probe_toks = tokenizer.convert_ids_to_tokens(probe_ids)
    unk_rate = sum(1 for i in probe_ids if i == tokenizer.unk_token_id) / len(probe_ids)
    print(f"Tokenizer probe: {probe_toks}")
    print(f"Tokenizer probe [UNK] rate: {unk_rate:.0%}")
    if unk_rate > 0.3:
        print("[!] High [UNK] rate -- the saved tokenizer looks broken. "
              "Re-run download_models.py after deleting models\\" + args.model)

    train_df = pd.read_csv(args.train_csv).sample(n=args.n_train, random_state=42).reset_index(drop=True)
    test_df = pd.read_csv(args.test_csv).sample(n=args.n_test, random_state=42).reset_index(drop=True)
    print(f"Train subset: {len(train_df)}  Test subset: {len(test_df)}")

    print(f"\nFine-tuning {args.model} for {args.epochs} epoch(s) ...")
    model = fine_tune(model, tokenizer, train_df, args.epochs, args.batch_size, args.lr, device)

    acc = evaluate(model, tokenizer, test_df, device)
    print(f"\n{args.model} test accuracy (subset of {len(test_df)}): {acc:.4f}")

    # -----------------------------------------------------------------
    # PartitionSHAP -- shap's own default explainer for text pipelines
    # -----------------------------------------------------------------
    def predict_proba(texts):
        enc = tokenizer(list(texts), truncation=True, padding=True, max_length=48, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    print(f"\nComputing PartitionSHAP on {args.n_shap} test comments (this is the slow part) ...")
    shap_texts = test_df["text"].tolist()[: args.n_shap]
    explainer = shap.Explainer(predict_proba, tokenizer)
    shap_values = explainer(shap_texts)

    # shap_values.values shape: (n_instances, n_tokens, n_classes) for text maskers
    # We explain class 1 ("Positive").
    per_instance_rows = []
    for i, text in enumerate(shap_texts):
        tokens = shap_values.data[i]
        vals = shap_values.values[i][:, 1]  # contribution to "Positive" class
        tags = [tag_term(t.strip()) for t in tokens]
        abs_vals = np.abs(vals)
        total = abs_vals.sum()
        row = {"text": text, "label": test_df.iloc[i]["label"]}
        for tag in ["EN", "TA-script", "TANGLISH?", "NOISE"]:
            mask = [t == tag for t in tags]
            row[f"{tag}_shap_pct"] = round(100 * abs_vals[mask].sum() / total, 2) if total > 0 else 0.0
        per_instance_rows.append(row)

    per_instance = pd.DataFrame(per_instance_rows)
    per_instance.to_csv(f"{out_prefix}_language_segment_shap.csv", index=False)

    pct_cols = [c for c in per_instance.columns if c.endswith("_shap_pct")]
    print(f"\n{args.model} -- dataset-level language-segment SHAP trend "
          f"(mean % of |PartitionSHAP| mass, {len(per_instance)} comments):")
    print(per_instance[pct_cols].mean().round(2).sort_values(ascending=False))

    with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "n_train": len(train_df), "n_test": len(test_df), "epochs": args.epochs,
            "test_accuracy": acc,
            "n_shap_instances": len(per_instance),
            "mean_language_segment_shap_pct": per_instance[pct_cols].mean().round(2).to_dict(),
        }, f, indent=2)

    print(f"\nSaved: {out_prefix}_language_segment_shap.csv, {out_prefix}_summary.json")

    # -----------------------------------------------------------------
    # The actual demo: explain this DravidianCodeMix-trained model on
    # our OWN real scraped YouTube comments, using our own pipeline's
    # real entity-aware language tags.
    # -----------------------------------------------------------------
    if args.apply_to_scraped:
        apply_to_scraped(
            args.model, predict_proba, tokenizer, out_prefix,
            args.scraped_csv, args.scraped_tagged_csv, args.n_scraped_per_video,
        )


if __name__ == "__main__":
    main()
