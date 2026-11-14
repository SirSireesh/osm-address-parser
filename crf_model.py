"""Linear-chain CRF address parser on python-crfsuite.

Trains on the same BIO JSONL as train.py with no external data:
token shapes, affixes, context windows and a train-set vocabulary
replace libpostal's dictionary, numex and transliteration tables.
Predictions align 1:1 with input tokens, so eval.py metrics apply
unchanged.

  python crf_model.py train --train data/sz-train.jsonl --out runs/crf-base
  python crf_model.py eval --model runs/crf-base --test data/sz-test.jsonl
  python crf_model.py hand50 --model runs/crf-base \
      --input trade_hand50b_input.txt --gold trade_hand50b_gold.txt
"""
import argparse
import json
import os
import re

import eval as ev
import pycrfsuite

MAX_NGRAM = 4
VOCAB_MIN_SUPPORT = 3
VOCAB_MAX_LABELS = 2
MIN_FEATURE_FREQ = 2
MODEL_FILE = "model.crfsuite"
VOCAB_FILE = "vocab.json"


def clean(word):
    low = word.lower()
    return low if re.search(r"\w", low) else f"p{low}"


def shape(word):
    # char-class runs collapsed: "R1A" -> "XdX", "605802" -> "d"
    clas = []

    for c in word:
        c_cls = ("d" if c.isdigit() else "X" if c.isupper()
            else "x" if c.isalpha() else "p")

        if not clas or c_cls != clas[-1]:
            clas.append(c_cls)

    return "".join(clas)


# vocab substitutes libpostal's global dictionary: word -> labels
# seen at training time, low-support noise pruned
def vocab_from_rows(rows):
    counts = {}

    for r in rows:
        for tok, tag in zip(r["tokens"], r["tags"]):
            low = clean(tok)

            if len(low) < 2 or low.isdigit():
                continue

            label = tag[2:]
            labs = counts.setdefault(low, {})
            labs[label] = labs.get(label, 0) + 1

    vocab = {}

    for low, labs in counts.items():
        top = sorted(labs.items(), key=lambda kv: (-kv[1], kv[0]))[:VOCAB_MAX_LABELS]
        keep = sorted(l for l, n in top if n >= VOCAB_MIN_SUPPORT)

        if keep:
            vocab[low] = keep

    return vocab


def token_features(tokens, i, vocab):
    w = tokens[i]
    low = clean(w)
    n = len(tokens)
    feat = [f"w={low}", f"s={shape(w)}"]

    if w.isupper() and len(w) > 1:
        feat.append("up")
    elif w.istitle():
        feat.append("ti")

    if w.isdigit():
        feat.append("dg")
    elif not w.isalpha() and re.search(r"\d", w):
        feat.append("mx")

    for k in range(2, MAX_NGRAM + 1):
        if len(low) >= k:
            feat.append(f"suf{k}={low[-k:]}")
            feat.append(f"pre{k}={low[:k]}")

    for lab in vocab.get(low, ()):
        feat.append(f"v={lab}")

    if i == 0:
        feat.append("bos")
    else:
        feat.append(f"c-1={clean(tokens[i - 1])}")

    if i == n - 1:
        feat.append("eos")
    else:
        feat.append(f"c+1={clean(tokens[i + 1])}")

    if i >= 2:
        feat.append(f"c-2={clean(tokens[i - 2])}")

    if i + 2 < n:
        feat.append(f"c+2={clean(tokens[i + 2])}")

    # neighbor shapes decode glued mixes like "S22" or "12-14"
    if i > 0:
        feat.append(f"s-1={shape(tokens[i - 1])}")

    if i + 1 < n:
        feat.append(f"s+1={shape(tokens[i + 1])}")

    # tail anchoring: city and postcode cluster near the end
    feat.append(f"e={min(n - i - 1, 4)}")

    return feat


def train_model(rows, vocab, path, iters, seed):
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.select("l2sgd")
    trainer.set_params({
        "c2": 0.1,
        "max_iterations": iters,
        "delta": 1e-5,
        "feature.minfreq": MIN_FEATURE_FREQ,
        "feature.possible_transitions": True,
    })

    for r in rows:
        seq = [token_features(r["tokens"], i, vocab) for i in range(len(r["tokens"]))]
        trainer.append(seq, r["tags"])

    trainer.train(path)


def load_tagger(model_dir):
    tagger = pycrfsuite.Tagger()
    tagger.open(os.path.join(model_dir, MODEL_FILE))

    with open(os.path.join(model_dir, VOCAB_FILE)) as f:
        vocab = json.load(f)

    return tagger, vocab


def predict_tags(rows, tagger, vocab):
    out = []

    for r in rows:
        seq = [token_features(r["tokens"], i, vocab) for i in range(len(r["tokens"]))]
        out.append(tagger.tag(seq))

    return out


# hand50 gold blocks: "addr: <raw>" plus "  <label>  <words>" lines
def parse_gold(path):
    blocks = []
    block = None

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            if line.startswith("addr:"):
                block = {"fields": {}}
                blocks.append(block)
            else:
                # gold format: "  <label>  <text>"; single-space labels exist
                parts = re.match(r"\s+([\w_]+)\s+(.+)\n", line).groups()
                block["fields"][parts[0]] = parts[1].strip()

    return blocks


def model_fields(line, tagger, vocab):
    tokens = ev.WORD_RE.findall(line)
    tags = predict_tags([{"tokens": tokens}], tagger, vocab)[0]

    return {k: ev.norm(v) for k, v in
        ev.fields(tokens, ev.spans(tags)).items()}


# per-label and row EM against the gold blocks
def hand50_report(blocks, tagger, vocab, lines):
    per_label = {}
    rows_ok = 0

    for block, line in zip(blocks, lines):
        got = model_fields(line, tagger, vocab)
        ok = True

        for label, gtext in block["fields"].items():
            c = per_label.setdefault(label, [0, 0])
            c[0] += got.get(label) == ev.norm(gtext)
            c[1] += 1
            ok = ok and got.get(label) == ev.norm(gtext)

        rows_ok += ok

    print(f"rows {rows_ok}/{len(blocks)}")
    print(f"{'label':15} {'EM':>7} {'n':>5}")

    for label in sorted(per_label):
        hits, n = per_label[label]
        print(f"{label:15} {hits:>7}/{n}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--iters", type=int, default=15)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--test", required=True)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--out", default=None)

    h = sub.add_parser("hand50")
    h.add_argument("--model", required=True)
    h.add_argument("--input", required=True)
    h.add_argument("--gold", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        rows = ev.load_jsonl(args.train, args.limit)
        vocab = vocab_from_rows(rows)
        os.makedirs(args.out, exist_ok=True)

        model_path = os.path.join(args.out, MODEL_FILE)
        train_model(rows, vocab, model_path, args.iters, args.seed)

        with open(os.path.join(args.out, VOCAB_FILE), "w") as f:
            json.dump(vocab, f)

        tagger, _ = load_tagger(args.out)
        print(f"saved {model_path}: vocab={len(vocab)} "
            f"labels={len(tagger.labels())}")

        return

    if args.cmd == "eval":
        tagger, vocab = load_tagger(args.model)
        rows = ev.load_jsonl(args.test, args.limit)
        preds = predict_tags(rows, tagger, vocab)
        metrics = ev.evaluate(rows, preds)
        ev.report(metrics)

        if args.out:
            with open(args.out, "w") as f:
                json.dump(metrics, f, indent=2)

        return

    with open(args.input) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    blocks = parse_gold(args.gold)
    tagger, vocab = load_tagger(args.model)
    hand50_report(blocks, tagger, vocab, lines)


if __name__ == "__main__":
    main()
