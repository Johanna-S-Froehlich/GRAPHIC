"""Per-layer confusion matrices for Text-GCN checkpoints on R52 / WOS-46985.

The GCN's per-epoch activations are pre-written to disk by the modified
``train.py`` inside ``GCNStandalone/text_gcn_tf2/``. This driver only has
to:
  1. Load the activations for a given epoch and layer
  2. Recover labels via the shuffle file (graph-order -> (doc, split, label))
  3. Stratified 80 / 20 split the NN-train activations into LC train / eval_train
  4. Train a linear classifier, build two CMs per (epoch, layer):

       epoch_<N>_iteration_0_block_<layer>_use_trainset.npy   built on eval_train
       epoch_<N>_iteration_0_block_<layer>_use_testset.npy     built on val (NN-test)

(The ``iteration_0_block_`` infix matches the existing ResNet / ViT
naming convention so metrics.py's file_matrix indexer picks them up.)

By default only ``graph_conv_1`` is probed: the second layer is the
classifier itself (a linear probe on it would be degenerate).
Pass ``--probes 1 2`` to include the second layer anyway.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from confusion_matrix_pipeline import (
    build_confusion_matrix,
    plot_train_curves,
    train_linear_classifier_fixed_epochs,
)
from custom_datasets.activation_dataset import ActivationDataset
from models.CustomLossFunction import CustomCrossEntropyLoss
from text_split import lc_split_assignment

# LC hyperparameter defaults; overridable via --lc-lr / --lc-epochs.
# The paper uses lr=0.3/wd=0 for R52 and lr=0.01/wd=0 for WoS46985, both at
# 100 cosine epochs. The defaults here match R52; for WoS pass --lc-lr 0.01 on
# the command line.
CLASSIFIER_LR = 0.3
CLASSIFIER_EPOCHS = 100
DEFAULT_PROBES = ("1",)


# ---------------------------------------------------------------------------
# Loaders for Yao's activation file format
# ---------------------------------------------------------------------------

def read_shuffle(path: str) -> list[tuple[int, str, str]]:
    """Parse ``<DATASET>_shuffle.txt`` -> [(orig_doc_id, 'train'|'test', label_str)]."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            doc_id, split, label = line.split("\t")
            rows.append((int(doc_id), split, label))
    return rows


def read_activations(path: str) -> torch.Tensor:
    """One activation vector per line, whitespace-separated floats."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    return torch.tensor(rows, dtype=torch.float32)


def find_epochs(activations_dir: Path) -> list[int]:
    pattern = re.compile(r"^epoch_(\d+)_train_activations_1\.txt$")
    epochs = []
    for name in os.listdir(activations_dir):
        match = pattern.match(name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def build_label_mapping(shuffle_rows) -> dict[str, int]:
    return {lbl: i for i, lbl in enumerate(
        sorted({label for _, _, label in shuffle_rows}))}


def split_indices_by_mask(shuffle_rows) -> tuple[list[int], list[int]]:
    """Return (train_positions, test_positions) in graph order — i.e. the row
    indices in the *_train_activations_* and *_test_activations_* files
    correspond, in order, to these graph positions filtered to the matching
    split."""
    train_positions = [i for i, (_, s, _) in enumerate(shuffle_rows) if s == "train"]
    test_positions = [i for i, (_, s, _) in enumerate(shuffle_rows) if s == "test"]
    return train_positions, test_positions


# ---------------------------------------------------------------------------
# LC training shared with the CNN driver
# ---------------------------------------------------------------------------

def train_lc_and_build_cms(act_train, act_eval_train, act_val, *,
                           num_classes, feature_dim, alpha, weight_decay,
                           lc_lr, lc_epochs, device, log_prefix):
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(),
                                 lr=lc_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=lc_epochs, eta_min=1e-7)
    criterion = CustomCrossEntropyLoss(alpha).to(device)

    def process_batch(batch):
        feats, true_labels, pred_labels = batch
        feats, true_labels, pred_labels = (
            feats.to(device), true_labels.to(device), pred_labels.to(device))
        return classifier(feats), true_labels, pred_labels

    (losses, test_losses, train_accs, test_accs,
     train_f1s, test_f1s) = train_linear_classifier_fixed_epochs(
        classifier=classifier,
        train_loader=act_train,
        # Per-epoch monitoring runs on the val (NN-test) partition so the
        # curves match the test-side numbers reported in the paper.
        # val is held out from both model and LC; no early stopping or
        # selection feeds back from these per-epoch values.
        test_loader=act_val,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=lc_epochs,
        device=device,
        process_batch=process_batch,
        log_prefix=log_prefix,
    )

    def feats_labels(batch):
        return batch[0], batch[1]

    C_train = build_confusion_matrix(
        classifier, act_eval_train, num_classes, device, feats_labels)
    C_val = build_confusion_matrix(
        classifier, act_val, num_classes, device, feats_labels)
    return (classifier, C_train, C_val,
            losses, test_losses, train_accs, test_accs,
            train_f1s, test_f1s)


# ---------------------------------------------------------------------------
# Per-class text dumps of misclassified val docs
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_confusions(classifier, val_loader, val_texts, classes_inv, out_dir, device):
    """val_texts[i] must correspond to the i-th sample emitted by val_loader."""
    os.makedirs(out_dir, exist_ok=True)
    by_true = defaultdict(list)
    cursor = 0
    classifier.eval()
    for batch in val_loader:
        feats, labels, _preds = batch
        feats = feats.to(device)
        preds = classifier(feats).argmax(dim=1).cpu()
        for j in range(labels.size(0)):
            true_y = int(labels[j])
            pred_y = int(preds[j])
            if true_y != pred_y:
                by_true[true_y].append((cursor + j, pred_y))
        cursor += labels.size(0)

    for true_y, samples in by_true.items():
        true_name = classes_inv[true_y].replace("/", "_")
        with open(os.path.join(out_dir, f"{true_name}.txt"), "w",
                  encoding="utf-8") as f:
            for idx, pred_y in samples:
                pred_name = classes_inv[pred_y]
                f.write(f"Pred: {pred_name} [{idx}]:\n{val_texts[idx]}\n\n")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["R52", "WOS46985"], required=True)
    p.add_argument("--activations-dir", required=True,
                   help="Directory of per-epoch activation files, typically "
                        "GCNStandalone/text_gcn_tf2/activations/.")
    p.add_argument("--corpus-path", default=None,
                   help="Path to raw text corpus for save_confusions "
                        "(e.g. GCNStandalone/text_gcn_tf2/data/corpus/R52.txt).")
    p.add_argument("--meta-path", default=None,
                   help="Path to the unshuffled meta file written alongside "
                        "the corpus (data/<DS>.txt; doc_id per line in corpus "
                        "order). Default: derived from --corpus-path.")
    p.add_argument("--output-dir", default="gcn_conf_matrices")
    p.add_argument("--probes", nargs="+", default=list(DEFAULT_PROBES),
                   help="Layer indices to probe ('1', '2'). Default: 1 only "
                        "(layer 2 is the classifier head -- linear-on-linear).")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="LC weight decay. Default 0 (tuned).")
    p.add_argument("--lc-lr", type=float, default=CLASSIFIER_LR,
                   help="LC peak learning rate (cosine annealing).")
    p.add_argument("--lc-epochs", type=int, default=CLASSIFIER_EPOCHS,
                   help="LC training epochs (cosine T_max).")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-train-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iteration", type=int, default=0,
                   help="LC repetition index. Seeds the LC stage with "
                        "seed + 1000*iteration and names outputs "
                        "iteration_<N>; data splits are iteration-"
                        "independent. Iteration 0 is the canonical run "
                        "(graphs, confusion text dumps).")
    p.add_argument("--start-epoch", type=int, default=0)
    p.add_argument("--end-epoch", type=int, default=None)
    p.add_argument("--epoch-step", type=int, default=1)
    p.add_argument("--save-confusions", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pprint(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    activations_dir = Path(args.activations_dir)

    shuffle_path = activations_dir / f"{args.dataset}_shuffle.txt"
    if not shuffle_path.exists():
        raise SystemExit(f"shuffle file not found at {shuffle_path}")

    shuffle = read_shuffle(str(shuffle_path))
    label_to_int = build_label_mapping(shuffle)
    classes_inv = {v: k for k, v in label_to_int.items()}
    num_classes = len(label_to_int)
    train_positions, test_positions = split_indices_by_mask(shuffle)
    # Defensive only: our build_graph.py sets val_size = 0 (Yao's 10%
    # early-stopping slice is disabled; fixed-epoch cosine training), so the
    # activation row count normally equals the shuffle's train block and no
    # clipping happens. The guard stays so a future re-enabled val slice
    # cannot silently misalign rows and labels.
    all_epochs_probe = find_epochs(activations_dir)
    if not all_epochs_probe:
        raise SystemExit(f"no activation files found in {activations_dir}")
    probe_path = activations_dir / f"epoch_{all_epochs_probe[0]}_train_activations_1.txt"
    real_train_size = read_activations(str(probe_path)).shape[0]
    if real_train_size < len(train_positions):
        print(f"clipping train_positions {len(train_positions)} -> "
              f"{real_train_size} (Yao val slice excluded from activations)")
        train_positions = train_positions[:real_train_size]
    train_labels_full = [label_to_int[shuffle[i][2]] for i in train_positions]
    val_labels_full = [label_to_int[shuffle[i][2]] for i in test_positions]
    print(f"{args.dataset}: {len(train_positions)} NN-train docs, "
          f"{len(test_positions)} NN-val docs, {num_classes} classes")

    # LC sub-split keyed by the original doc_id (first column of the shuffle
    # file). lc_split_assignment is the SAME function called by the CNN side,
    # which is keyed by the same doc_id convention -- so a given doc lands in
    # the same bucket on both pipelines regardless of how each loaded it.
    train_doc_ids_per_row = [shuffle[i][0] for i in train_positions]
    labels_by_doc_id = dict(zip(train_doc_ids_per_row, train_labels_full))

    counts = Counter(train_labels_full)
    singletons = sorted(c for c, n in counts.items() if n == 1)
    if singletons:
        import warnings  # local
        warnings.warn(
            f"{len(singletons)} class(es) have only 1 NN-train sample and "
            f"were placed entirely in LC-train (no eval_train representation): "
            f"{singletons[:10]}{'...' if len(singletons) > 10 else ''}",
            stacklevel=2,
        )

    bucket = lc_split_assignment(
        labels_by_doc_id, seed=args.seed,
        eval_train_frac=args.eval_train_frac)
    lc_train_idx: list[int] = []
    eval_train_idx: list[int] = []
    for k, did in enumerate(train_doc_ids_per_row):
        if bucket[did] == "lc_train":
            lc_train_idx.append(k)
        else:
            eval_train_idx.append(k)
    lc_train_y = [train_labels_full[i] for i in lc_train_idx]
    eval_train_y = [train_labels_full[i] for i in eval_train_idx]

    # Load val texts (raw corpus) if writing per-class misclassification dumps.
    val_texts: list[str] = []
    if args.save_confusions:
        if not args.corpus_path:
            raise SystemExit("--corpus-path required when --save-confusions is set")
        with open(args.corpus_path, encoding="utf-8") as f:
            corpus = [line.rstrip("\n") for line in f]
        # Corpus-line resolution: corpus line k
        # and meta line k (data/<DS>.txt) describe the same document, in the
        # UNSHUFFLED tts order written by prepare_for_gcn.py; the shuffle
        # file is a permutation produced later by build_graph.py. So neither
        # the original doc_id nor the graph position indexes the corpus
        # directly: the right line is the meta-file position of the doc_id.
        meta_path = (Path(args.meta_path) if args.meta_path
                     else Path(args.corpus_path).parent.parent
                     / Path(args.corpus_path).name)
        if not meta_path.exists():
            raise SystemExit(
                f"meta file {meta_path} not found (needed to map doc_ids to "
                f"corpus lines); pass --meta-path explicitly")
        pos: dict[int, int] = {}
        with open(meta_path, encoding="utf-8") as f:
            for k, line in enumerate(f):
                if line.strip():
                    pos[int(line.split("\t")[0])] = k
        val_texts = [corpus[pos[shuffle[i][0]]] for i in test_positions]

    # Seed the LC stage explicitly: iteration N reproducibly trains the N-th
    # LC repetition (iteration 0 = the canonical run used for the graphs).
    torch.manual_seed(args.seed + 1000 * args.iteration)
    torch.cuda.manual_seed_all(args.seed + 1000 * args.iteration)
    it = f"iteration_{args.iteration}"

    out_dir = Path(args.output_dir) / f"alpha_{args.alpha}"
    curves_dir = out_dir / "train_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(exist_ok=True)

    all_epochs = find_epochs(activations_dir)
    end = args.end_epoch if args.end_epoch is not None else max(all_epochs) + 1
    in_range = [e for e in all_epochs if args.start_epoch <= e < end]
    # Always include the final available epoch. train.py writes the final
    # epoch's activations after the training loop, but the mod-`epoch_step`
    # grid below would otherwise drop it whenever the last epoch is off the
    # step boundary.
    final_epoch = max(in_range) if in_range else None
    selected = sorted({
        e for e in in_range
        if (e - args.start_epoch) % args.epoch_step == 0 or e == 1
        or e == final_epoch
    })
    print(f"Processing epochs: {selected}")

    # Head-layer model-output CM (test side only): the second graph-conv
    # layer already emits class scores, so a linear probe on it would be
    # degenerate. We instead read the network's own argmax on the test
    # partition. No training-side CM since the network was fit on that data.
    def head_test_cm(epoch: int) -> np.ndarray:
        head_test = read_activations(str(
            activations_dir / f"epoch_{epoch}_test_activations_2.txt"))
        preds = head_test.argmax(dim=1).numpy()
        truth = np.asarray(val_labels_full)
        C = np.zeros((num_classes, num_classes), dtype=np.float64)
        for t, p in zip(truth, preds):
            C[int(t), int(p)] += 1
        row = C.sum(axis=1, keepdims=True)
        return np.where(row > 0, C / np.where(row > 0, row, 1), C)

    for epoch in selected:
        head_path = activations_dir / f"epoch_{epoch}_test_activations_2.txt"
        if head_path.exists():
            np.save(
                out_dir / f"epoch_{epoch}_{it}_block_2_use_testset.npy",
                head_test_cm(epoch))

        # Frozen-model predictions on the corresponding partition. Used as
        # pred_labels in CustomCrossEntropyLoss so that alpha=0 actually
        # mimics the GCN's own argmax instead of degenerating into "predict
        # class 0 for everything". The GCN's class scores live in the
        # layer-2 activations (already softmax-pre-image of num_classes
        # dimensions); take argmax for hard predictions.
        head_train_acts = read_activations(str(
            activations_dir / f"epoch_{epoch}_train_activations_2.txt"))
        head_test_acts = read_activations(str(
            activations_dir / f"epoch_{epoch}_test_activations_2.txt"))
        train_preds_full = head_train_acts.argmax(dim=1).long()
        val_model_preds = head_test_acts.argmax(dim=1).long()

        for layer in args.probes:
            train_path = activations_dir / f"epoch_{epoch}_train_activations_{layer}.txt"
            test_path = activations_dir / f"epoch_{epoch}_test_activations_{layer}.txt"
            train_acts = read_activations(str(train_path))
            val_acts = read_activations(str(test_path))
            feature_dim = train_acts.shape[1]

            lc_train_acts = train_acts[lc_train_idx]
            eval_train_acts = train_acts[eval_train_idx]
            lc_train_labels = torch.tensor(lc_train_y, dtype=torch.long)
            eval_train_labels = torch.tensor(eval_train_y, dtype=torch.long)
            val_labels = torch.tensor(val_labels_full, dtype=torch.long)
            lc_train_preds = train_preds_full[lc_train_idx]
            eval_train_preds = train_preds_full[eval_train_idx]
            val_preds = val_model_preds

            at = DataLoader(
                ActivationDataset(lc_train_acts, lc_train_labels, lc_train_preds),
                batch_size=args.batch_size, shuffle=True)
            aet = DataLoader(
                ActivationDataset(eval_train_acts, eval_train_labels, eval_train_preds),
                batch_size=args.batch_size, shuffle=False)
            av = DataLoader(
                ActivationDataset(val_acts, val_labels, val_preds),
                batch_size=args.batch_size, shuffle=False)

            log_prefix = f"[epoch {epoch} layer {layer}] "
            (clf, C_train, C_val,
             losses, tl, ta, te_acc, tr_f1, te_f1) = train_lc_and_build_cms(
                at, aet, av,
                num_classes=num_classes,
                feature_dim=feature_dim,
                alpha=args.alpha,
                weight_decay=args.weight_decay,
                lc_lr=args.lc_lr,
                lc_epochs=args.lc_epochs,
                device=device,
                log_prefix=log_prefix)

            np.save(out_dir / f"epoch_{epoch}_{it}_block_{layer}_use_trainset.npy",
                    C_train.numpy())
            np.save(out_dir / f"epoch_{epoch}_{it}_block_{layer}_use_testset.npy",
                    C_val.numpy())
            plot_train_curves(
                losses, tl, ta, te_acc,
                output_path=str(curves_dir / f"epoch_{epoch}_{it}_layer_{layer}_metrics.png"),
                title_suffix=f"epoch {epoch} {it} layer {layer}",
                train_macro_f1=tr_f1, val_macro_f1=te_f1, heldout_name="Test")
            # Text dumps only for the canonical iteration; the per-class
            # misclassification listings of repetitions add no insight.
            if args.save_confusions and args.iteration == 0:
                save_confusions(
                    clf, av, val_texts, classes_inv,
                    str(out_dir / f"epoch_{epoch}_layer_{layer}_confusions"),
                    device)


if __name__ == "__main__":
    main()
