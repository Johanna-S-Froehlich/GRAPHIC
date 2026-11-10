"""Shared LC split function for the GRAPHIC text-classification pipeline.

Both ``CNNStandalone/data.py`` and ``confusion_matrices_gcn.py`` call
:func:`lc_split_assignment` with a ``labels_by_doc_id`` mapping. Identical
input -> identical output regardless of how the calling pipeline ordered
its data, which guarantees the CNN and GCN LC splits put the same docs
into the same buckets for cross-architecture comparison.

Per-class manual allocation (replaces sklearn's rounded stratified
train_test_split, which collapses eval_train to 0 for classes with
N <= 4 at test_size=0.2):

    N == 1:  -> LC-train only (singleton; eval_train empty for that class)
    N >= 2:  max(1, round(eval_train_frac * N)) -> eval_train,
             rest                                 -> LC-train

Per-class shuffle is seeded by ``hash((seed, label))`` so the assignment
is invariant to the order in which classes appear in the input dict.
"""

from __future__ import annotations

import random


def lc_split_assignment(
    labels_by_doc_id: dict[int, int],
    seed: int = 42,
    eval_train_frac: float = 0.2,
) -> dict[int, str]:
    """Map each NN-train doc_id to 'lc_train' or 'eval_train'.

    Parameters
    ----------
    labels_by_doc_id : dict[int, int]
        doc_id -> class label, covering every NN-train document.
    seed : int
        Per-class RNG seed (combined with the class label via ``hash``).
    eval_train_frac : float
        Target fraction of each class to put into eval_train (rounded up
        to at least 1 for classes with N >= 2).

    Returns
    -------
    dict[int, str]
        ``{doc_id: 'lc_train' | 'eval_train'}`` covering every input doc_id.
        Singleton classes (N == 1) map to 'lc_train'.
    """
    by_class: dict[int, list[int]] = {}
    for doc_id, label in labels_by_doc_id.items():
        by_class.setdefault(label, []).append(doc_id)

    assignment: dict[int, str] = {}
    for label in sorted(by_class):
        ids = sorted(by_class[label])  # canonical order regardless of input order
        rng = random.Random(hash((seed, label)))
        rng.shuffle(ids)
        n = len(ids)
        if n == 1:
            assignment[ids[0]] = "lc_train"
            continue
        n_eval = max(1, round(eval_train_frac * n))
        n_train = n - n_eval
        for did in ids[:n_train]:
            assignment[did] = "lc_train"
        for did in ids[n_train:]:
            assignment[did] = "eval_train"
    return assignment
