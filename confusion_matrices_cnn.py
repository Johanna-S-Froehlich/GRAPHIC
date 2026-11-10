"""Per-layer confusion matrices for KimCNN-static checkpoints on R52 / WOS-46985.

Mirrors the Vit pipeline: probe activations at named layers
(``embedding``, ``convs.0``, ``convs.1``, ``convs.2``, ``pool``), train a
linear classifier on the cached features, and build two normalized
confusion matrices per checkpoint and layer:

    <ckpt>_iteration_0_block_<layer>_use_trainset.npy   built on eval_train
    <ckpt>_iteration_0_block_<layer>_use_testset.npy     built on val

The ``iteration_0_block_`` infix mirrors the existing ResNet / ViT
convention so the unified ``metrics.py`` (file_matrix glob) picks them
up unchanged.

The ``embedding`` probe is frozen by construction in CNN-static (GloVe is
``freeze=True``; BERT inputs are pre-encoded), so its activations are
identical at every checkpoint. The LC is nevertheless retrained per
checkpoint: at alpha < 1 the loss targets include the checkpoint's own
predictions, which change across training; at alpha = 1 the repeated CMs
serve as an LC-initialization sanity check.

For each misclassified sample in val, a per-true-class text file is written
listing the predicted class and the document text.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from natsort import natsorted
from torch import nn
from torch.utils.data import DataLoader

# Make CNNStandalone importable for data + model factories.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "CNNStandalone"))

from data import build_bert_data, build_glove_data, load_dataset  # noqa: E402
from kimcnn import KimCNN  # noqa: E402

from confusion_matrix_pipeline import (  # noqa: E402
    build_confusion_matrix,
    plot_train_curves,
    train_linear_classifier_fixed_epochs,
)
from custom_datasets.activation_dataset import ActivationDataset  # noqa: E402
from feature_extraction.hooks import register_hooks  # noqa: E402
from models.CustomLossFunction import CustomCrossEntropyLoss  # noqa: E402

DEFAULT_PROBES = ("embedding", "convs.0", "convs.1", "convs.2", "pool")
# LC hyperparameter defaults; overridable via --lc-lr / --lc-epochs.
# The paper uses lr=0.01/wd=1e-3 for R52 and lr=0.003/wd=1e-3 for WoS46985,
# both at 100 cosine epochs. The defaults here match R52; for WoS pass
# --lc-lr 0.003 on the command line.
CLASSIFIER_LR = 0.01
CLASSIFIER_EPOCHS = 100
_SPLIT_NAMES = ("train", "eval_train", "val")


# ---------------------------------------------------------------------------
# Feature pooling per probe
# ---------------------------------------------------------------------------

def _pool_activation(layer: str, activation: torch.Tensor,
                     lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Reduce one layer's activation to a per-sample vector.

    - embedding: (B, L, E) -> masked mean over the first ``lengths[i]``
      positions of each sample. Both tokenizers right-pad, and for BERT the
      PAD positions carry real nonzero encoder outputs, so an unmasked mean
      dilutes short documents with padding mass (~90% for a 50-token doc at
      max_len 512). Falls back to the plain mean when lengths is None.
    - convs.N:   (B, K, L') -> max over seq -> (B, K)
    - pool:      (B, N*K) -> identity
    """
    if layer == "embedding":
        if lengths is None:
            return activation.mean(dim=1)
        L = activation.shape[1]
        mask = (torch.arange(L, device=activation.device)[None, :]
                < lengths.to(activation.device)[:, None])
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(activation.dtype)
        return (activation * mask.unsqueeze(-1)).sum(dim=1) / denom
    if layer.startswith("convs."):
        return activation.max(dim=2).values
    if layer == "pool":
        return activation
    raise ValueError(f"unsupported probe layer {layer!r}")


def _probe_dim(layer: str, model: KimCNN, kernel_num: int) -> int:
    if layer == "embedding":
        return model.embed_dim
    if layer.startswith("convs."):
        return kernel_num
    if layer == "pool":
        return model.fc.in_features
    raise ValueError(f"unsupported probe layer {layer!r}")


# ---------------------------------------------------------------------------
# Feature extraction + cache
# ---------------------------------------------------------------------------

def extract_features(model, loader, device, probes, lengths=None,
                     activations=None):
    """Run the model once and return {layer: (N, d_layer)} plus labels & preds.

    Predictions from the frozen KimCNN are also returned for the GRAPHIC
    pred-label-aware loss (alpha != 1.0). ``lengths`` is the per-sample
    true-token-count tensor for the whole split (loader order, which is
    deterministic: all extraction loaders use shuffle=False); it drives the
    masked-mean embedding probe. ``activations`` is the hook dict from a
    prior ``register_hooks(model)``; pass it when extracting several splits
    from the same model, because hooks.py never removes hooks and repeated
    registration stacks redundant hook calls on the model.
    """
    if activations is None:
        activations, _ = register_hooks(model, only=list(probes))
    pooled = {layer: [] for layer in probes}
    labels_buf, preds_buf = [], []
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds_buf.append(logits.argmax(dim=1).cpu())
            labels_buf.append(y)
            batch_lengths = (lengths[offset:offset + y.size(0)]
                             if lengths is not None else None)
            offset += y.size(0)
            for layer in probes:
                pooled[layer].append(_pool_activation(
                    layer, activations[layer], batch_lengths).cpu())
    return (
        {layer: torch.cat(chunks) for layer, chunks in pooled.items()},
        torch.cat(labels_buf),
        torch.cat(preds_buf),
    )


def cache_split(model, loader, device, probes, cache_dir, ckpt_name,
                split_name, lengths=None, activations=None):
    feats, labels, preds = extract_features(model, loader, device, probes,
                                            lengths=lengths,
                                            activations=activations)
    for layer, tensor in feats.items():
        torch.save(tensor, cache_dir / f"{ckpt_name}_{split_name}_features_{layer}.pt")
    torch.save(labels, cache_dir / f"{ckpt_name}_{split_name}_labels.pt")
    torch.save(preds, cache_dir / f"{ckpt_name}_{split_name}_preds.pt")


def load_loaders_for_probe(cache_dir, ckpt_name, layer, batch_size):
    def _loader(split):
        feats = torch.load(cache_dir / f"{ckpt_name}_{split}_features_{layer}.pt")
        labels = torch.load(cache_dir / f"{ckpt_name}_{split}_labels.pt")
        preds = torch.load(cache_dir / f"{ckpt_name}_{split}_preds.pt")
        # The LC TRAIN loader reshuffles per epoch, matching the GCN
        # driver's protocol (a fixed batch partition on one side would make
        # the iteration bands measure different noise sources across
        # architectures). Evaluation loaders stay ordered: CM building and the confusion text dumps
        # index into them positionally.
        return DataLoader(ActivationDataset(feats, labels, preds),
                          batch_size=batch_size, shuffle=(split == "train"))
    return tuple(_loader(s) for s in _SPLIT_NAMES)


def cleanup_per_layer(cache_dir, ckpt_name, layer):
    for split in _SPLIT_NAMES:
        path = cache_dir / f"{ckpt_name}_{split}_features_{layer}.pt"
        if path.exists():
            path.unlink()


def cleanup_per_ckpt(cache_dir, ckpt_name):
    for split in _SPLIT_NAMES:
        for suffix in ("labels", "preds"):
            path = cache_dir / f"{ckpt_name}_{split}_{suffix}.pt"
            if path.exists():
                path.unlink()


# ---------------------------------------------------------------------------
# LC training + CM building for one (checkpoint, layer)
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
        # curves and printed Test Loss/Acc/F1 numbers track the same
        # quantity reported in the paper (LC test-side accuracy from
        # _use_testset.npy). val is genuinely held out from both the
        # model and the LC; using it for monitoring is fine because no
        # early stopping or hyperparameter selection feeds back from
        # these per-epoch values.
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
# save_confusions: per-class text dumps of misclassified val samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_confusions(classifier, val_loader, val_texts, classes, out_dir, device):
    """Write per-true-class text files listing every misclassified val sample.

    Sample index within the loader maps directly to position in ``val_texts``
    (loader is constructed with ``shuffle=False`` and there are no other
    transforms — no hardcoded corpus offsets needed).
    """
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
        true_name = classes[true_y].replace("/", "_")
        with open(os.path.join(out_dir, f"{true_name}.txt"), "w",
                  encoding="utf-8") as f:
            for idx, pred_y in samples:
                pred_name = classes[pred_y]
                text = val_texts[idx]
                f.write(f"Pred: {pred_name} [{idx}]:\n{text}\n\n")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory of KimCNN .pt checkpoints.")
    p.add_argument("--output-dir", default="kimcnn_conf_matrices")
    p.add_argument("--dataset", choices=["r52", "wos46985"], required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--embedding-type", choices=["glove", "bert"], required=True)
    p.add_argument("--glove-path", default=None)
    p.add_argument("--bert-cache", default=None)
    p.add_argument("--bert-model", default="bert-base-uncased")
    p.add_argument("--embed-dim", type=int, default=300)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--max-vocab", type=int, default=50_000)
    p.add_argument("--eval-train-frac", type=float, default=0.2)
    p.add_argument("--kernel-num", type=int, default=100,
                   help="Must match the trained KimCNN's kernel_num.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iteration", type=int, default=0,
                   help="LC repetition index. Seeds the LC stage with "
                        "seed + 1000*iteration and names outputs "
                        "iteration_<N>; data splits are iteration-"
                        "independent. Iteration 0 is the canonical run "
                        "(graphs, confusion text dumps).")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--probes", nargs="+", default=list(DEFAULT_PROBES))
    p.add_argument("--alpha", type=float, default=1.0,
                   help="CE mix between true and pred labels (alpha=1 -> true only).")
    p.add_argument("--weight-decay", type=float, default=1e-3,
                   help="LC weight decay. Default 1e-3 (tuned).")
    p.add_argument("--lc-lr", type=float, default=CLASSIFIER_LR,
                   help="LC peak learning rate (cosine annealing).")
    p.add_argument("--lc-epochs", type=int, default=CLASSIFIER_EPOCHS,
                   help="LC training epochs (cosine T_max).")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--epoch-step", type=int, default=1)
    p.add_argument("--save-confusions", action="store_true",
                   help="Write per-class misclassification text files.")
    return p.parse_args()


def build_data(args, device):
    if args.embedding_type == "glove":
        if not args.glove_path:
            raise SystemExit("--glove-path required for glove embedding")
        return build_glove_data(
            args.dataset, args.data_dir, args.glove_path,
            embed_dim=args.embed_dim, max_len=args.max_len,
            max_vocab=args.max_vocab, eval_train_frac=args.eval_train_frac,
            seed=args.seed)
    if not args.bert_cache:
        raise SystemExit("--bert-cache required for bert embedding")
    return build_bert_data(
        args.dataset, args.data_dir, args.bert_cache,
        model_name=args.bert_model, max_len=args.max_len,
        eval_train_frac=args.eval_train_frac, seed=args.seed, device=device)


def select_checkpoints(args):
    paths = natsorted(glob(os.path.join(args.checkpoint_dir, "*.pt")))
    return paths[args.start_index:args.end_index:args.epoch_step]


def reconstruct_model(checkpoint, data, device):
    saved = checkpoint.get("args", {})
    model = KimCNN(
        num_classes=data.num_classes,
        embed_dim=data.embed_dim,
        kernel_sizes=tuple(saved.get("kernel_sizes", [3, 4, 5])),
        kernel_num=saved.get("kernel_num", 100),
        dropout=saved.get("dropout", 0.5),
        pretrained_embeddings=data.pretrained_embeddings,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    args = parse_args()
    pprint(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = build_data(args, device)
    train_loader = DataLoader(data.train, batch_size=args.batch_size, shuffle=False)
    eval_train_loader = DataLoader(data.eval_train, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(data.val, batch_size=args.batch_size, shuffle=False)
    # For save_confusions we need the raw val strings; TokenDataset/BertDataset
    # only carry encoded tensors. Re-load via load_dataset and stash the texts.
    val_texts: list[str] = []
    if args.save_confusions:
        # seed must match build_data's split or the dumped texts misalign.
        loaded = load_dataset(args.dataset, args.data_dir, seed=args.seed)
        val_texts = loaded[2]  # val_texts is 3rd element of the 7-tuple

    # Seed the LC stage explicitly: iteration N reproducibly trains the N-th
    # LC repetition (iteration 0 = the canonical run used for the graphs).
    torch.manual_seed(args.seed + 1000 * args.iteration)
    torch.cuda.manual_seed_all(args.seed + 1000 * args.iteration)
    it = f"iteration_{args.iteration}"

    alpha_dir = Path(args.output_dir) / f"alpha_{args.alpha}"
    # Per-iteration feature cache: iterations may run as parallel jobs and
    # the per-layer caches are transient, so sharing them would race.
    cache_dir = alpha_dir / f"cache_it{args.iteration}"
    curves_dir = alpha_dir / "train_curves"
    for d in (alpha_dir, cache_dir, curves_dir):
        d.mkdir(parents=True, exist_ok=True)

    checkpoints = select_checkpoints(args)
    if not checkpoints:
        raise SystemExit(f"no .pt checkpoints found in {args.checkpoint_dir}")
    print(f"Using {len(checkpoints)} checkpoints:")
    pprint(checkpoints)

    # ---- per-checkpoint, per-probe loop ----
    # NB: the embedding probe is processed per-checkpoint like any other layer.
    # Its activations are constant (the embedding layer is frozen), but the
    # LC's CrossEntropyLoss target mixes in the model's predictions when
    # alpha < 1.0, so the resulting CM still depends on the checkpoint. Under
    # alpha=1.0 this redundantly produces identical embedding CMs across
    # checkpoints, which is a small price for correctness everywhere else.
    processed: set[str] = set()
    for ckpt_path in checkpoints:
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = reconstruct_model(ckpt, data, device)
        ckpt_name = ckpt_path.stem
        # The final checkpoint is saved as kimcnn_final.pt; name its CMs by
        # the stored epoch number instead, as file_matrix expects numeric
        # epochs.
        if ckpt_name == "kimcnn_final" and "epoch" in ckpt:
            ckpt_name = f"kimcnn_epoch_{int(ckpt['epoch'])}"
        # A checkpoint dir may carry both kimcnn_epoch_<N>.pt and
        # kimcnn_final.pt for the same epoch (identical weights); skip the
        # duplicate rather than overwriting its CMs with a second RNG draw.
        if ckpt_name in processed:
            print(f"skipping {ckpt_path.name}: {ckpt_name} already processed")
            continue
        processed.add(ckpt_name)
        print(f"\n=== checkpoint {ckpt_name} ===")

        if not args.probes:
            continue

        # Register the forward hooks once per (fresh) model and reuse the
        # dict across the three splits; hooks.py never removes hooks.
        activations, _ = register_hooks(model, only=list(args.probes))
        cache_split(model, train_loader, device, args.probes, cache_dir,
                    ckpt_name, "train", lengths=data.train_lengths,
                    activations=activations)
        cache_split(model, eval_train_loader, device, args.probes, cache_dir,
                    ckpt_name, "eval_train", lengths=data.eval_train_lengths,
                    activations=activations)
        cache_split(model, val_loader, device, args.probes, cache_dir,
                    ckpt_name, "val", lengths=data.val_lengths,
                    activations=activations)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Head-layer model-output CM (test side only): the fc head emits
        # class scores directly, so a linear probe on it would be degenerate.
        # We instead build the CM from the network's own argmax on the test
        # partition. No training-side CM since the network was fit on that
        # data.
        head_labels = torch.load(cache_dir / f"{ckpt_name}_val_labels.pt")
        head_preds = torch.load(cache_dir / f"{ckpt_name}_val_preds.pt")
        C_head = torch.zeros(data.num_classes, data.num_classes)
        for t, p in zip(head_labels, head_preds):
            C_head[int(t), int(p)] += 1
        row_sum = C_head.sum(dim=1, keepdim=True)
        C_head = torch.where(row_sum > 0, C_head / row_sum, C_head)
        np.save(alpha_dir / f"{ckpt_name}_{it}_block_fc_use_testset.npy",
                C_head.numpy())

        for layer in args.probes:
            at, aet, av = load_loaders_for_probe(
                cache_dir, ckpt_name, layer, args.batch_size)
            (clf, C_train, C_val,
             losses, tl, ta, te_acc, tr_f1, te_f1) = train_lc_and_build_cms(
                at, aet, av,
                num_classes=data.num_classes,
                feature_dim=_probe_dim(layer, model, args.kernel_num),
                alpha=args.alpha, weight_decay=args.weight_decay,
                lc_lr=args.lc_lr, lc_epochs=args.lc_epochs,
                device=device, log_prefix=f"[{ckpt_name} {layer}] ")
            np.save(alpha_dir / f"{ckpt_name}_{it}_block_{layer}_use_trainset.npy",
                    C_train.numpy())
            np.save(alpha_dir / f"{ckpt_name}_{it}_block_{layer}_use_testset.npy",
                    C_val.numpy())
            plot_train_curves(
                losses, tl, ta, te_acc,
                output_path=str(curves_dir / f"{ckpt_name}_{it}_{layer}_metrics.png"),
                title_suffix=f"{ckpt_name} {layer}",
                train_macro_f1=tr_f1, val_macro_f1=te_f1, heldout_name="Test")
            # Text dumps only for the canonical iteration; the per-class
            # misclassification listings of repetitions add no insight.
            if args.save_confusions and args.iteration == 0:
                save_confusions(clf, av, val_texts, data.classes,
                                str(alpha_dir / f"{ckpt_name}_{layer}_confusions"),
                                device)
            cleanup_per_layer(cache_dir, ckpt_name, layer)

        cleanup_per_ckpt(cache_dir, ckpt_name)


if __name__ == "__main__":
    main()
