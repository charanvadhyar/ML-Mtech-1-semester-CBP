
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
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optim.step()
            total_loss += out.loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  mean loss = {total_loss / len(dl):.4f}")
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
    parser.add_argument("--n_train", type=int, default=1500)
    parser.add_argument("--n_test", type=int, default=300)
    parser.add_argument("--n_shap", type=int, default=60, help="How many test comments to run PartitionSHAP on (slow).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out_prefix", default=None)
    args = parser.parse_args()

    out_prefix = args.out_prefix or f"bert_{args.model}"
    model_path = f"{args.models_dir}/{args.model}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading {args.model} from {model_path} (local, no network) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)

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


if __name__ == "__main__":
    main()
