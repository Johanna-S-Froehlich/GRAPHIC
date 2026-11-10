"""Download R52 (Reuters-21578 single-label, 52 classes) for the CNN pipeline.

Pulls the dataset from Yao et al.'s `text_gcn` repository, which redistributes
Cardoso-Cachopo's R52 split, and writes two files in `r52/` at the repo root:

    r52-train-all-terms.txt    <label>\\t<text>
    r52-test-all-terms.txt     <label>\\t<text>
    R52_labels.txt             one label per line, sorted

If you use this dataset, please cite:

    Ana Cardoso-Cachopo. Improving Methods for Single-label Text Categorization.
    PhD Thesis, Universidade Tecnica de Lisboa, 2007.
"""

import os
import urllib.request

REPO = "yao8839836/text_gcn"
RAW = f"https://raw.githubusercontent.com/{REPO}/master"
META_URL = f"{RAW}/data/R52.txt"
CORPUS_URL = f"{RAW}/data/corpus/R52.txt"
OUT_DIR = "r52"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Downloading metadata from {META_URL}")
    meta_lines = urllib.request.urlopen(META_URL).read().decode().splitlines()
    print(f"Downloading corpus from {CORPUS_URL}")
    corpus_lines = urllib.request.urlopen(CORPUS_URL).read().decode().splitlines()

    if len(meta_lines) != len(corpus_lines):
        raise RuntimeError(
            f"metadata has {len(meta_lines)} lines but corpus has "
            f"{len(corpus_lines)} — mismatch between Yao's R52 files."
        )

    train_path = os.path.join(OUT_DIR, "r52-train-all-terms.txt")
    test_path = os.path.join(OUT_DIR, "r52-test-all-terms.txt")
    labels = set()

    with open(train_path, "w") as ftrain, open(test_path, "w") as ftest:
        for meta, text in zip(meta_lines, corpus_lines):
            _doc_id, split, label = meta.split("\t")
            labels.add(label)
            target = ftrain if split == "train" else ftest
            target.write(f"{label}\t{text}\n")

    labels_path = os.path.join(OUT_DIR, "R52_labels.txt")
    with open(labels_path, "w") as f:
        for label in sorted(labels):
            f.write(f"{label}\n")

    print(f"Wrote {train_path}, {test_path}, {labels_path}")
    print(f"  {len(labels)} classes, "
          f"{sum(1 for line in open(train_path))} train, "
          f"{sum(1 for line in open(test_path))} test docs")


if __name__ == "__main__":
    main()
