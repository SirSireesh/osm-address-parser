"""Parse free-text addresses with a trained checkpoint.

Reads one address per line, prints the model's per-label
fields for hand evaluation.

  python predict.py --model runs/base/model \
      --input addresses.txt

Without --input, reads addresses from stdin one per line:
on a tty it is a REPL (line-edited via readline, each
address predicted on enter); piped stdin predicts in batches.
"""

import argparse
import readline  # noqa: F401 — line editing for input() on a tty
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import eval


def line_iter(path, prompt):
    if path:
        with open(path) as f:
            for l in f:
                l = l.rstrip("\n")
                if l.strip():
                    yield l
        return

    try:
        while True:
            l = input(prompt)
            if l.strip():
                yield l
    except EOFError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # tty stdin is a REPL: predict each line on enter (batch 1).
    # piped stdin batches, no prompt noise.
    prompt = "> " if sys.stdin.isatty() else ""
    flush_size = 1 if prompt else eval.BATCH
    lines = []
    rows = []

    def flush():
        if not rows:
            return
        preds = eval.predict(model, tokenizer, rows, device)
        for line, row, pred in zip(lines, rows, preds):
            fields = eval.fields(row["tokens"], eval.spans(pred))

            print(f"addr: {line}")

            for label in sorted(fields):
                print(f"  {label:13} {fields[label]}")

            print()

        lines.clear()
        rows.clear()

    for consumed, line in enumerate(line_iter(args.input, prompt), 1):
        lines.append(line)
        rows.append({"tokens": eval.WORD_RE.findall(line)})

        if args.limit and consumed >= args.limit:
            break
        if len(rows) >= flush_size:
            flush()

    flush()


if __name__ == "__main__":
    main()
