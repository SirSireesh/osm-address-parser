"""Fine-tune ModernBERT-base for address token classification.

Trains on the BIO JSONL from osm_address_extractor's `dataset`
subcommand, then reports per-label span F1 on the test file.

  python train.py --train data/sz-train.jsonl \
      --test data/sz-test.jsonl --epochs 2 --out model
"""
import argparse
import json
import os
import random
import time

import eval as ev
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = ev.MAX_LEN
LR = 5e-5
WARMUP_FRAC = 0.06
LOG_EVERY = 50


def load_jsonl(path, limit=None):
    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break

    return rows


# pre-tokenized examples, unpadded; collate pads per batch
class AddrDataset(Dataset):
    def __init__(self, rows, tokenizer, label2id):
        self.tok = tokenizer(
            [r["tokens"] for r in rows],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
        )

        # label of each subword: first subword of a word carries the
        # word's tag, the rest are ignored (-100)
        self.labels = []

        for i, r in enumerate(rows):
            labs = [-100] * len(self.tok["input_ids"][i])
            prev = None

            for pos, wid in enumerate(self.tok.word_ids(i)):
                if wid is None or wid == prev:
                    continue

                labs[pos] = label2id[r["tags"][wid]]
                prev = wid

            self.labels.append(labs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.tok["input_ids"][i],
            "attention_mask": self.tok["attention_mask"][i],
            "labels": self.labels[i],
        }


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []

    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        torch.set_num_threads(4)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.train)
    random.Random(123).shuffle(rows)

    if args.limit:
        rows = rows[: args.limit]

    # label set from the data: landmark joined since the POC
    label2id = {t: i for i, t in enumerate(sorted({t for r in rows for t in r["tags"]}))}
    print(f"examples={len(rows)} labels={len(label2id)} device={device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label={i: t for t, i in label2id.items()},
    ).to(device)

    ds = AddrDataset(rows, tokenizer, label2id)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        pin_memory=device.type == "cuda",
    )

    steps = len(dl) * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, int(steps * WARMUP_FRAC), steps)

    # bf16 autocast on GPU; CPU trains in fp32 (POC-proven)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    model.train()
    t0 = time.time()
    step = running = n_r = 0

    for epoch in range(args.epochs):
        for batch in dl:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            with amp:
                out = model(**batch)

            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running += out.loss.item()
            n_r += 1
            step += 1

            if step % LOG_EVERY == 0:
                rate = step / (time.time() - t0)
                print(f"epoch {epoch} step {step}/{steps} "
                    f"loss={running / n_r:.4f} {rate:.2f} it/s", flush=True)
                running = n_r = 0

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    with open(os.path.join(args.out, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print(f"saved to {args.out} in {time.time() - t0:.0f}s", flush=True)

    if not args.test:
        return

    test_rows = ev.load_jsonl(args.test)
    preds = ev.predict(model, tokenizer, test_rows, device)
    metrics = ev.evaluate(test_rows, preds)
    ev.report(metrics)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
