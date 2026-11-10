"""Data loaders for R52 and WOS-46985 in two embedding modes (GloVe / BERT).

Split convention:
    train       80 % of NN-train per class (the LC training set)
    eval_train  20 % of NN-train per class (for the training-side CMs)
    val         20 % of the dataset, stratified by label (the test set)

Classes with fewer than 2 samples in NN-train are folded entirely into
``train`` (see text_split.py) and a warning is printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Repo root holds text_split.py (shared with confusion_matrices_gcn.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from text_split import lc_split_assignment  # noqa: E402


# ---------------------------------------------------------------------------
# Raw text loading
# ---------------------------------------------------------------------------

def _read_label_text(path: str) -> tuple[list[str], list[str]]:
    """Read a label\\ttext file (R52 format)."""
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


def load_r52(data_dir: str, val_frac: float = 0.2, seed: int = 42):
    """Return (train_texts, train_y, val_texts, val_y, classes,
    train_doc_ids, val_doc_ids).

    Doc-id convention: position in the concatenation (train file lines,
    then test file lines), regardless of which split each doc lands in.

    R52 ships a canonical Cachopo train/test split, but we re-stratify it
    here to guarantee every class with >= 3 total samples lands in all
    three GRAPHIC splits (LC-train / eval_train / NN-val). Under
    Cachopo's split ``platinum`` has only 1 NN-train sample, which leaves
    its eval_train CM row empty by construction. Re-stratifying with the
    same seed used elsewhere is reproducible and is documented in the
    paper's methodology section. WOS-46985 is unaffected (it never had a
    canonical split).
    """
    train_texts, train_label_strs = _read_label_text(
        os.path.join(data_dir, "r52-train-all-terms.txt"))
    test_texts, test_label_strs = _read_label_text(
        os.path.join(data_dir, "r52-test-all-terms.txt"))
    classes = sorted(set(train_label_strs + test_label_strs))
    label_to_int = {lbl: i for i, lbl in enumerate(classes)}

    # Pool then stratified split. doc_ids are positions in the pooled
    # (train_file, then test_file) corpus, identical to what
    # GCNStandalone/prepare_for_gcn.py:load_r52 sees with the same seed.
    all_texts = train_texts + test_texts
    all_labels = [label_to_int[l] for l in train_label_strs + test_label_strs]
    all_doc_ids = list(range(len(all_texts)))
    train_doc_ids, val_doc_ids = train_test_split(
        all_doc_ids, test_size=val_frac, stratify=all_labels,
        random_state=seed)
    train_texts_out = [all_texts[i] for i in train_doc_ids]
    val_texts_out = [all_texts[i] for i in val_doc_ids]
    train_y = [all_labels[i] for i in train_doc_ids]
    val_y = [all_labels[i] for i in val_doc_ids]
    return (train_texts_out, train_y, val_texts_out, val_y, classes,
            train_doc_ids, val_doc_ids)


def load_wos46985(data_dir: str, val_frac: float = 0.2, seed: int = 42):
    """Return (train_texts, train_y, val_texts, val_y, classes,
    train_doc_ids, val_doc_ids).

    WOS-46985 has no canonical train/test split. We carve out ``val_frac``
    documents stratified by label as the val set.

    Doc-id convention for WOS-46985: original X.txt line number. The
    indices returned by ``train_test_split`` are exactly those positions,
    so we keep them as the doc_ids and use them on both CNN and GCN
    sides for cross-pipeline LC-split consistency.
    """
    with open(os.path.join(data_dir, "X.txt"), encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    with open(os.path.join(data_dir, "Y.txt"), encoding="utf-8") as f:
        raw_labels = [int(x) for x in f.read().split()]
    assert len(texts) == len(raw_labels), "X.txt and Y.txt length mismatch"
    # Map labels to lexicographically-sorted "class_<N>" indices so the
    # KimCNN CM ordering matches the GCN side (which sorts the unique
    # label strings) and wos46985_mappings.py. With raw integer labels,
    # CM row N would index class N, but the label list goes
    # class_0, class_1, class_10, class_100, class_101, ..., causing a
    # permutation of every node identity for WoS. (R52 is unaffected:
    # class strings sort the same regardless of source.)
    classes = sorted({f"class_{i}" for i in set(raw_labels)})
    name_to_idx = {name: i for i, name in enumerate(classes)}
    # Stratify on the RAW integer labels (the same values GCN's
    # prepare_for_gcn.py uses), then remap to sorted-name indices afterwards.
    # sklearn's train_test_split iterates classes in sorted-unique-value
    # order, so changing the label values before the split changes the
    # per-class RNG draws and produces a different partition. Stratifying
    # on raw_labels keeps the CNN doc-id partition byte-identical to the
    # GCN's shuffle file, so both pipelines put a document into the same LC
    # sub-split. The remap is then a pure relabel and preserves the
    # sorted-name CM ordering.
    idx_all = list(range(len(texts)))
    train_doc_ids, val_doc_ids = train_test_split(
        idx_all, test_size=val_frac, stratify=raw_labels, random_state=seed)
    train_texts = [texts[i] for i in train_doc_ids]
    val_texts = [texts[i] for i in val_doc_ids]
    train_y = [name_to_idx[f"class_{raw_labels[i]}"] for i in train_doc_ids]
    val_y = [name_to_idx[f"class_{raw_labels[i]}"] for i in val_doc_ids]
    return (train_texts, train_y, val_texts, val_y, classes,
            train_doc_ids, val_doc_ids)


def load_dataset(name: str, data_dir: str, **kwargs):
    """Returns a 7-tuple: (train_texts, train_y, val_texts, val_y, classes,
    train_doc_ids, val_doc_ids)."""
    if name == "r52":
        return load_r52(data_dir, **kwargs)
    if name == "wos46985":
        return load_wos46985(data_dir, **kwargs)
    raise ValueError(f"unknown dataset {name!r}")


# ---------------------------------------------------------------------------
# Splitting NN-train into LC train + eval_train
# ---------------------------------------------------------------------------

def stratified_train_eval_train_split(
    texts: list, labels: list[int], eval_train_frac: float = 0.2, seed: int = 42
) -> tuple[list, list, list[int], list[int]]:
    """Per-class deterministic allocation into LC-train / eval_train.

    Thin wrapper around :func:`text_split.lc_split_assignment` that operates on
    parallel ``texts``/``labels`` lists (positional doc_ids 0..len-1) instead
    of an explicit doc_id mapping. Use ``lc_split_assignment`` directly when
    you need cross-pipeline-consistent splits keyed by stable doc_ids.
    """
    doc_ids = list(range(len(labels)))
    labels_by_doc_id = dict(zip(doc_ids, labels))

    counts = Counter(labels)
    singletons = sorted(c for c, n in counts.items() if n == 1)
    if singletons:
        warnings.warn(
            f"{len(singletons)} class(es) have only 1 NN-train sample and were "
            f"placed entirely in LC-train (no eval_train representation): "
            f"{singletons[:10]}{'...' if len(singletons) > 10 else ''}",
            stacklevel=2,
        )

    bucket = lc_split_assignment(labels_by_doc_id, seed=seed,
                                 eval_train_frac=eval_train_frac)
    train_indices = [did for did in doc_ids if bucket[did] == "lc_train"]
    eval_indices = [did for did in doc_ids if bucket[did] == "eval_train"]
    return (
        [texts[i] for i in train_indices],
        [texts[i] for i in eval_indices],
        [labels[i] for i in train_indices],
        [labels[i] for i in eval_indices],
    )


# ---------------------------------------------------------------------------
# GloVe path: tokenize, build vocab, load GloVe vectors
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], max_vocab: int) -> dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


def load_glove(path: str, vocab: dict[str, int], embed_dim: int) -> torch.Tensor:
    matrix = np.random.uniform(
        -0.25, 0.25, (len(vocab), embed_dim)).astype(np.float32)
    matrix[vocab["<PAD>"]] = 0.0
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab and len(parts) == embed_dim + 1:
                matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"GloVe coverage: {found}/{len(vocab) - 2} non-special vocab words")
    return torch.from_numpy(matrix)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.x = torch.tensor(
            [encode(t, vocab, max_len) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BertDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels):
        self.x = embeddings
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Cache is stored as fp16 to halve disk + RAM; upcast per-sample to
        # fp32 for the CNN head. The cast is on a single max_len*hidden slice
        # (~1.5 MB), so per-batch overhead is negligible. Old fp32 caches
        # still work -- .float() on an fp32 tensor is a no-op.
        return self.x[idx].float(), self.y[idx]


# ---------------------------------------------------------------------------
# BERT cache management
# ---------------------------------------------------------------------------


def _bert_cache_fingerprint(texts: list[str], model_name: str,
                            max_len: int) -> str:
    """Content fingerprint of a BERT cache: hashes the exact text sequence
    (order-sensitive) plus the encoder settings. Any re-split, re-seed,
    dataset fix, or max_len change yields a different fingerprint."""
    h = hashlib.sha256()
    h.update(f"{model_name}|{max_len}|{len(texts)}".encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def get_or_build_bert_cache(
    texts: list[str], cache_path: str, model_name: str, max_len: int,
    device: torch.device, batch_size: int = 64,
) -> torch.Tensor:
    """Return a tensor of shape (N, max_len, hidden) for ``texts``.

    Reads from ``cache_path`` if present; otherwise encodes with ``model_name``
    and writes the result to ``cache_path``.
    """
    sidecar_path = cache_path + ".json"
    if os.path.exists(cache_path):
        print(f"Loading BERT cache from {cache_path}")
        # mmap=True memory-maps the cache file instead of loading it fully into
        # CPU RAM (~37 GB for WoS-46985 at max_len=512). With mmap each batch
        # reads only its own slice from disk on demand. Requires PyTorch >= 2.1.
        cached = torch.load(cache_path, map_location="cpu", mmap=True)
        if cached.shape[0] != len(texts):
            raise RuntimeError(
                f"BERT cache has {cached.shape[0]} entries, expected {len(texts)}")
        if cached.shape[1] != max_len:
            raise RuntimeError(
                f"BERT cache {cache_path} stores max_len={cached.shape[1]} "
                f"but max_len={max_len} was requested. The filename does not "
                f"control the length; rebuild the cache or fix --max-len.")
        # Sidecar fingerprint: row count alone cannot detect a re-ordered or
        # re-split text sequence, whose rows would silently misalign.
        fp = _bert_cache_fingerprint(texts, model_name, max_len)
        if os.path.exists(sidecar_path):
            with open(sidecar_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != fp:
                raise RuntimeError(
                    f"BERT cache {cache_path} was built for a different text "
                    f"sequence or settings (fingerprint mismatch). The rows "
                    f"would silently misalign. Delete the cache (and its "
                    f".json sidecar) and rebuild.")
        else:
            warnings.warn(
                f"{cache_path} has no .json sidecar (older cache); "
                f"cannot verify text alignment. Writing a sidecar that "
                f"asserts the CURRENT texts match this cache -- if the data "
                f"or split changed since the cache was built, delete the "
                f"cache and rebuild instead.")
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({"model_name": model_name, "max_len": max_len,
                           "n_texts": len(texts), "fingerprint": fp}, f)
        return cached

    print(f"Building BERT cache with {model_name} (max_len={max_len}, fp16)")
    from transformers import AutoModel, AutoTokenizer  # lazy import
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    # Probe hidden size from the model config so we can pre-allocate the
    # output tensor and fill it slice-by-slice. The previous list+torch.cat
    # layout briefly held 2x the cache in RAM and OOM-killed WoS46985 even
    # on A100 nodes (~200 GiB peak). Pre-allocating + fp16 keeps peak RAM at
    # half the cache size (~36 GiB for WoS46985 vs the old ~144 GiB peak).
    hidden = encoder.config.hidden_size
    embeddings = torch.empty(
        (len(texts), max_len, hidden), dtype=torch.float16, pin_memory=False)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]
            enc = tokenizer(batch, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            out = encoder(**enc).last_hidden_state.to(torch.float16).cpu()
            embeddings[start:end] = out
            print(f"  encoded {end}/{len(texts)}", end="\r")
    print()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(embeddings, cache_path)
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "max_len": max_len,
                   "n_texts": len(texts),
                   "fingerprint": _bert_cache_fingerprint(
                       texts, model_name, max_len)}, f)
    return embeddings


# ---------------------------------------------------------------------------
# Top-level loader: returns the three datasets and any auxiliary tensors
# ---------------------------------------------------------------------------

@dataclass
class TextData:
    # train: the LC's training partition (= lc_train, 80% of NN-train).
    # nn_train: the model's training partition (= lc_train + eval_train,
    #           the full ~80% NN-train slice of the corpus). The model
    #           must train on nn_train so that the "training-side" CM
    #           (LC predictions on eval_train) reflects model behaviour
    #           on data the model has actually seen.
    train: Dataset
    eval_train: Dataset
    val: Dataset
    nn_train: Dataset
    num_classes: int
    classes: list[str]
    embed_dim: int
    pretrained_embeddings: torch.Tensor | None = None  # GloVe only
    # True (un-padded) token counts per sample, clamped to [1, max_len];
    # consumed by the CM driver's masked-mean embedding probe so PAD
    # positions do not dilute the per-document mean. None for callers
    # that never probe (training ignores these).
    train_lengths: torch.Tensor | None = None
    eval_train_lengths: torch.Tensor | None = None
    val_lengths: torch.Tensor | None = None


def _partition_nn_train_by_doc_id(
    train_texts, train_y, train_doc_ids, eval_train_frac, seed
):
    """Apply lc_split_assignment, return parallel LC-train / eval_train lists."""
    labels_by_doc_id = dict(zip(train_doc_ids, train_y))
    counts = Counter(train_y)
    singletons = sorted(c for c, n in counts.items() if n == 1)
    if singletons:
        warnings.warn(
            f"{len(singletons)} class(es) have only 1 NN-train sample and were "
            f"placed entirely in LC-train (no eval_train representation): "
            f"{singletons[:10]}{'...' if len(singletons) > 10 else ''}",
            stacklevel=3,
        )
    bucket = lc_split_assignment(labels_by_doc_id, seed=seed,
                                 eval_train_frac=eval_train_frac)
    lc_train_idx, eval_train_idx = [], []
    for i, did in enumerate(train_doc_ids):
        (lc_train_idx if bucket[did] == "lc_train" else eval_train_idx).append(i)
    return (
        [train_texts[i] for i in lc_train_idx],
        [train_texts[i] for i in eval_train_idx],
        [train_y[i] for i in lc_train_idx],
        [train_y[i] for i in eval_train_idx],
    )



def _glove_lengths(texts: list[str], max_len: int) -> torch.Tensor:
    """True token count per text under the GloVe tokenizer, in [1, max_len]."""
    return torch.tensor(
        [max(1, min(len(tokenize(t)), max_len)) for t in texts],
        dtype=torch.long)


def _bert_lengths(texts: list[str], model_name: str,
                  max_len: int) -> torch.Tensor:
    """Per-text attention length (real tokens incl. [CLS]/[SEP]) under the
    BERT tokenizer with the same truncation/padding as the cache build, so
    the masked mean covers exactly the positions BERT actually encodes."""
    from transformers import AutoTokenizer  # lazy import
    tok = AutoTokenizer.from_pretrained(model_name)
    out: list[int] = []
    bs = 512
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding="max_length", truncation=True,
                  max_length=max_len)
        out.extend(sum(m) for m in enc["attention_mask"])
    return torch.tensor(out, dtype=torch.long)


def build_glove_data(
    dataset: str, data_dir: str, glove_path: str,
    embed_dim: int = 300, max_len: int = 256, max_vocab: int = 50_000,
    eval_train_frac: float = 0.2, seed: int = 42,
) -> TextData:
    train_texts, train_y, val_texts, val_y, classes, train_doc_ids, _ = \
        load_dataset(dataset, data_dir, seed=seed)
    lc_train_texts, eval_train_texts, lc_train_y, eval_train_y = \
        _partition_nn_train_by_doc_id(
            train_texts, train_y, train_doc_ids, eval_train_frac, seed)
    # Vocab is built on the union of all texts so val tokens don't all become
    # <UNK>. The classifier never sees val labels during training.
    vocab = build_vocab(
        lc_train_texts + eval_train_texts + val_texts, max_vocab)
    embeddings = load_glove(glove_path, vocab, embed_dim)
    return TextData(
        train=TokenDataset(lc_train_texts, lc_train_y, vocab, max_len),
        eval_train=TokenDataset(eval_train_texts, eval_train_y, vocab, max_len),
        val=TokenDataset(val_texts, val_y, vocab, max_len),
        nn_train=TokenDataset(
            lc_train_texts + eval_train_texts,
            lc_train_y + eval_train_y, vocab, max_len),
        num_classes=len(classes),
        classes=classes,
        embed_dim=embed_dim,
        pretrained_embeddings=embeddings,
        train_lengths=_glove_lengths(lc_train_texts, max_len),
        eval_train_lengths=_glove_lengths(eval_train_texts, max_len),
        val_lengths=_glove_lengths(val_texts, max_len),
    )


def build_bert_data(
    dataset: str, data_dir: str, bert_cache: str,
    model_name: str = "bert-base-uncased",
    max_len: int = 256, eval_train_frac: float = 0.2, seed: int = 42,
    device: torch.device | str = "cuda",
) -> TextData:
    train_texts, train_y, val_texts, val_y, classes, train_doc_ids, _ = \
        load_dataset(dataset, data_dir, seed=seed)
    lc_train_texts, eval_train_texts, lc_train_y, eval_train_y = \
        _partition_nn_train_by_doc_id(
            train_texts, train_y, train_doc_ids, eval_train_frac, seed)
    all_texts = lc_train_texts + eval_train_texts + val_texts
    tensor = get_or_build_bert_cache(
        all_texts, bert_cache, model_name, max_len, torch.device(device))
    n_train = len(lc_train_texts)
    n_eval = len(eval_train_texts)
    return TextData(
        train=BertDataset(tensor[:n_train], lc_train_y),
        eval_train=BertDataset(tensor[n_train:n_train + n_eval], eval_train_y),
        val=BertDataset(tensor[n_train + n_eval:], val_y),
        nn_train=BertDataset(tensor[:n_train + n_eval],
                             lc_train_y + eval_train_y),
        num_classes=len(classes),
        classes=classes,
        embed_dim=tensor.shape[-1],
        train_lengths=_bert_lengths(lc_train_texts, model_name, max_len),
        eval_train_lengths=_bert_lengths(eval_train_texts, model_name, max_len),
        val_lengths=_bert_lengths(val_texts, model_name, max_len),
    )
