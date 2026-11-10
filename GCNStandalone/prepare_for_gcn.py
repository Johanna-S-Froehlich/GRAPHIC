"""Convert R52 or WOS-46985 into Yao's Text-GCN expected file layout.

Reads from r52/ or wos46985/ at the repo root, writes:

    text_gcn_tf2/data/<NAME>.txt        metadata: <doc_id>\\t<train|test>\\t<label>
    text_gcn_tf2/data/corpus/<NAME>.txt raw text per line (same order as metadata)

After this, the standard Yao pipeline runs:

    python remove_words.py <NAME>
    python build_graph.py  <NAME>
    python train.py        <NAME>

R52 is re-stratified with the same seed as CNNStandalone/data.py (see load_r52
below). For WOS-46985, which has no canonical split, a stratified 80 / 20 split
is generated (seed=42 by default; pass --seed to change).
"""

import argparse
import os
import sys

from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
GCN_DATA_DIR = os.path.join(THIS_DIR, "text_gcn_tf2", "data")


def _read_label_text(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, text = line.split("\t", 1)
            texts.append(text)
            labels.append(label)
    return texts, labels


def load_r52(seed=42, val_frac=0.2):
    """Return (texts, labels, splits, doc_ids).

    Doc-id convention: position in the concatenation of
    r52-train-all-terms.txt then r52-test-all-terms.txt.

    R52 is re-stratified (rather than using Cachopo's canonical split)
    with the same seed as CNNStandalone/data.py:load_r52, so the two
    pipelines agree on which doc lands in NN-train vs NN-val. See the
    docstring there for the methodology rationale.

    Texts are written in train-then-test order (Yao's pipeline reads
    them positionally for graph construction), but the metadata file's
    doc_id column carries the original position so cross-pipeline
    doc_id-keyed splits stay aligned.
    """
    base = os.path.join(REPO_ROOT, "r52")
    if not os.path.exists(base):
        sys.exit("r52/ not found — run `python download_r52.py` first.")
    train_texts, train_labels = _read_label_text(
        os.path.join(base, "r52-train-all-terms.txt"))
    test_texts, test_labels = _read_label_text(
        os.path.join(base, "r52-test-all-terms.txt"))
    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels
    all_doc_ids = list(range(len(all_texts)))

    train_idx, test_idx = train_test_split(
        all_doc_ids, test_size=val_frac, stratify=all_labels,
        random_state=seed)
    order = train_idx + test_idx  # original positions, train-first
    texts_ordered = [all_texts[i] for i in order]
    labels_ordered = [all_labels[i] for i in order]
    splits = ["train"] * len(train_idx) + ["test"] * len(test_idx)
    return texts_ordered, labels_ordered, splits, order


def load_wos46985(seed=42, val_frac=0.2):
    """Return (texts, labels, splits, doc_ids).

    Doc-id convention for WOS-46985: original X.txt line number. The
    NN-train/NN-val partition is the same stratified train_test_split
    as the CNN side (matching seed), so doc_ids align across pipelines.
    Texts are written to Yao's corpus in train-then-test order, but the
    metadata file records the *original* X.txt position as the doc_id.
    """
    base = os.path.join(REPO_ROOT, "wos46985")
    if not os.path.exists(base):
        sys.exit("wos46985/ not found — run `python download_wos46985.py` first.")
    with open(os.path.join(base, "X.txt"), encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    with open(os.path.join(base, "Y.txt"), encoding="utf-8") as f:
        y = [int(x) for x in f.read().split()]
    labels = [f"class_{i}" for i in y]
    idx = list(range(len(texts)))
    train_idx, test_idx = train_test_split(
        idx, test_size=val_frac, stratify=y, random_state=seed)
    order = train_idx + test_idx  # original X.txt positions, train first
    texts_ordered = [texts[i] for i in order]
    labels_ordered = [labels[i] for i in order]
    splits = ["train"] * len(train_idx) + ["test"] * len(test_idx)
    doc_ids = order  # original X.txt positions, in train-then-test order
    return texts_ordered, labels_ordered, splits, doc_ids


def write_yao_format(texts, labels, splits, doc_ids, dataset_name):
    os.makedirs(os.path.join(GCN_DATA_DIR, "corpus"), exist_ok=True)
    meta = [f"{doc_ids[i]}\t{splits[i]}\t{labels[i]}"
            for i in range(len(texts))]
    meta_path = os.path.join(GCN_DATA_DIR, f"{dataset_name}.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta))
    corpus_path = os.path.join(GCN_DATA_DIR, "corpus", f"{dataset_name}.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    print(f"Wrote {meta_path} ({len(meta)} docs)")
    print(f"Wrote {corpus_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["R52", "WOS46985"], required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the stratified train/test split.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Val fraction for the stratified split.")
    args = p.parse_args()

    if args.dataset == "R52":
        texts, labels, splits, doc_ids = load_r52(args.seed, args.val_frac)
    else:
        texts, labels, splits, doc_ids = load_wos46985(args.seed, args.val_frac)
    write_yao_format(texts, labels, splits, doc_ids, args.dataset)


if __name__ == "__main__":
    main()
