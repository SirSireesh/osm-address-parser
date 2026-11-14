"""Parse free-text addresses with a trained checkpoint.

Reads one address per line, prints the model's per-label
fields for hand evaluation.

  python predict.py --model runs/base/model \
      --input addresses.txt
"""
import argparse

import eval
import torch
from transformers import AutoModelForTokenClassification, \
    AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(args.input) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    if args.limit:
        lines = lines[: args.limit]

    rows = [{"tokens": eval.WORD_RE.findall(l)} for l in lines]
    preds = eval.predict(model, tokenizer, rows, device)

    for line, row, pred in zip(lines, rows, preds):
        fields = eval.fields(row["tokens"], eval.spans(pred))

        print(f"addr: {line}")

        for label in sorted(fields):
            print(f"  {label:13} {fields[label]}")

        print()


if __name__ == "__main__":
    main()
