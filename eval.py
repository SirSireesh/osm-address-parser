"""Evaluate the address token classifier.

Metrics on a BIO JSONL test file:
- span F1 per label (exact (start, end, label) span match) + micro
- token label accuracy (B-/I- prefix stripped)
- field exact match: reconstructed per-label text, normalized

  python eval.py --model runs/base/model --test data/sz-test.jsonl
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# must match train.py truncation; longer examples are skipped
MAX_LEN = 96
BATCH = 64
WORD_RE = re.compile(r"\w+|[^\w\s]")


def load_jsonl(path, limit=None):
    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break

    return rows


# one tag per input word: the tag of the first subword
@torch.no_grad()
def predict(model, tokenizer, rows, device):
    out = []
    model.eval()

    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]

        enc = tokenizer(
            [r["tokens"] for r in chunk],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        ).to(device)

        preds = model(**enc).logits.argmax(-1).tolist()

        for j, row in enumerate(chunk):
            tags, prev = [], None

            for pos, wid in enumerate(enc.word_ids(j)):
                if wid is None or wid == prev:
                    continue

                tags.append(model.config.id2label[preds[j][pos]])
                prev = wid

            out.append(tags)

    return out


# word-index spans: [(start, end, label)]
def spans(tags):
    res = []
    i = 0

    while i < len(tags):
        if not tags[i].startswith("B-"):
            i += 1
            continue

        label = tags[i][2:]
        j = i + 1

        while j < len(tags) and tags[j] == "I-" + label:
            j += 1

        res.append((i, j, label))
        i = j

    return res


# per-label text; punctuation-only tokens dropped, multi-span labels joined
def fields(tokens, span_list):
    words_by_label = {}

    for s, e, label in span_list:
        words = [w for w in tokens[s:e] if re.search(r"\w", w)]

        if words:
            words_by_label.setdefault(label, []).extend(words)

    return {k: " ".join(v) for k, v in words_by_label.items()}


def norm(text):
    return " ".join(w for w in WORD_RE.findall(text) if re.search(r"\w", w))


# token/whitespace joins can't fake an error: "16/3" == "16 / 3"
def norm_tokens(words):
    return norm(" ".join(words))


def evaluate(rows, preds):
    # per label: [tp, fn, fp, field_em, gold_fields]
    counts = {}
    micro = [0, 0, 0]
    tok_ok = tok_n = skipped = 0

    def bucket(label):
        return counts.setdefault(label, [0, 0, 0, 0, 0])

    for row, pred in zip(rows, preds):
        gold = row["tags"]

        # tokenizer truncation: cannot align, drop the example
        if len(pred) != len(gold):
            skipped += 1
            continue

        gspans, pspans = spans(gold), spans(pred)
        gset, pset = set(gspans), set(pspans)

        for s in gspans:
            b = bucket(s[2])
            hit = s in pset
            b[0] += hit
            b[1] += not hit

        for s in pspans:
            if s not in gset:
                bucket(s[2])[2] += 1

        micro[0] += len(gset & pset)
        micro[1] += len(pset) - len(gset & pset)
        micro[2] += len(gset) - len(gset & pset)

        # field exact match: only where the gold row has the field
        gfields = fields(row["tokens"], gspans)
        pfields = fields(row["tokens"], pspans)

        for label, gtext in gfields.items():
            b = bucket(label)
            b[4] += 1
            ptext = pfields.get(label)
            b[3] += ptext is not None and norm(ptext) == norm(gtext)

        # token accuracy sans B-/I- prefix
        for gtag, ptag in zip(gold, pred):
            tok_ok += gtag.lstrip("BI-") == ptag.lstrip("BI-")
            tok_n += 1

    metrics = {
        "examples": len(rows),
        "skipped": skipped,
        "token_acc": tok_ok / max(tok_n, 1),
        "labels": {},
    }

    for label, (tp, fn, fp, em, present) in counts.items():
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        metrics["labels"][label] = {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "field_em": em / present if present else 0.0,
            "support": tp + fn,
        }

    tp, fp, fn = micro
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    metrics["micro"] = {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }

    return metrics


def report(metrics):
    print(f"examples={metrics['examples']} skipped={metrics['skipped']} "
        f"token_acc={metrics['token_acc']:.4f}")
    print(f"{'label':15} {'support':>7} {'P':>6} {'R':>6} {'F1':>6} {'EM':>6}")

    for label, m in sorted(metrics["labels"].items()):
        print(f"{label:15} {m['support']:>7} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['field_em']:>6.3f}")

    mi = metrics["micro"]
    print(f"{'micro':15} {'':>7} {mi['precision']:>6.3f} "
        f"{mi['recall']:>6.3f} {mi['f1']:>6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="write metrics.json here")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)

    rows = load_jsonl(args.test, args.limit)
    preds = predict(model, tokenizer, rows, device)
    metrics = evaluate(rows, preds)

    report(metrics)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

