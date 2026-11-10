"""Train KimCNN-static on R52 or WOS-46985 with GloVe or BERT embeddings.

Usage examples:
    python train_kimcnn.py --dataset r52 --data-dir ../r52 \\
        --embedding-type glove --glove-path ../glove_embeddings/glove.840B.300d.txt

    python train_kimcnn.py --dataset wos46985 --data-dir ../wos46985 \\
        --embedding-type bert --bert-cache ../wos46985_bert_cache/bert_uncased_512.pt

Checkpoints are saved every ``--checkpoint-every`` epochs (named
``kimcnn_epoch_N.pt``) so the confusion-matrix driver can probe them later,
plus a final-epoch checkpoint ``kimcnn_final.pt``. The val partition is the
held-out test set under the cosine, no-early-stop regime: it is reported each
epoch for monitoring but never used to select a checkpoint.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import F1Score

# Ensure the package can be imported when this script is run directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_bert_data, build_glove_data  # noqa: E402
from kimcnn import KimCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=["r52", "wos46985"], required=True)
    p.add_argument("--data-dir", required=True,
                   help="Directory with R52 or WOS-46985 files.")
    p.add_argument("--embedding-type", choices=["glove", "bert"], required=True)
    p.add_argument("--glove-path", default=None,
                   help="Path to the GloVe vectors, e.g. glove.840B.300d.txt "
                        "(required for glove).")
    p.add_argument("--bert-cache", default=None,
                   help="Path to (or destination for) the BERT cache .pt file "
                        "(required for bert).")
    p.add_argument("--bert-model", default="bert-base-uncased")
    p.add_argument("--embed-dim", type=int, default=300,
                   help="Embedding dim for GloVe (ignored for BERT — inferred).")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--max-vocab", type=int, default=50_000)
    p.add_argument("--eval-train-frac", type=float, default=0.2)

    p.add_argument("--kernel-sizes", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--kernel-num", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["none", "plateau", "cosine"],
                   default="cosine",
                   help="LR schedule. 'cosine' decays smoothly over the full "
                        "training run independently of val (no val leakage); "
                        "'plateau' watches val_loss to decide when to reduce; "
                        "'none' keeps LR fixed.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="kimcnn_checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=3,
                   help="Save a checkpoint every N epochs (0 = final only).")
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    f1 = F1Score(task="multiclass", num_classes=num_classes,
                 average="macro").to(device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        f1.update(preds, y)
    return total_loss / total, correct / total, f1.compute().item()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device}")

    if args.embedding_type == "glove":
        if args.glove_path is None:
            raise SystemExit("--glove-path is required for --embedding-type glove")
        data = build_glove_data(
            args.dataset, args.data_dir, args.glove_path,
            embed_dim=args.embed_dim, max_len=args.max_len,
            max_vocab=args.max_vocab, eval_train_frac=args.eval_train_frac,
            seed=args.seed)
    else:
        if args.bert_cache is None:
            raise SystemExit("--bert-cache is required for --embedding-type bert")
        data = build_bert_data(
            args.dataset, args.data_dir, args.bert_cache,
            model_name=args.bert_model, max_len=args.max_len,
            eval_train_frac=args.eval_train_frac, seed=args.seed,
            device=device)

    # Model trains on the FULL NN-train partition (lc_train + eval_train),
    # so the training-side CM (LC predictions on eval_train) measures data
    # the model has seen.
    train_loader = DataLoader(
        data.nn_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        data.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    model = KimCNN(
        num_classes=data.num_classes,
        embed_dim=data.embed_dim,
        kernel_sizes=args.kernel_sizes,
        kernel_num=args.kernel_num,
        dropout=args.dropout,
        pretrained_embeddings=data.pretrained_embeddings,
    ).to(device)
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=np.sqrt(0.1), patience=7, min_lr=5e-7)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-7)

    os.makedirs(args.output_dir, exist_ok=True)

    print(
        "CONFIG: "
        f"dataset={args.dataset}, embedding_type={args.embedding_type}, "
        f"kernel_sizes={'+'.join(str(k) for k in args.kernel_sizes)}, "
        f"kernel_num={args.kernel_num}, dropout={args.dropout}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, "
        f"scheduler={args.scheduler}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, max_len={args.max_len}, "
        f"seed={args.seed}",
        flush=True)

    final_path = os.path.join(args.output_dir, "kimcnn_final.pt")
    final_metrics = None

    # Snapshot the random-init state so GRAPHIC can probe the untrained
    # network as an "epoch 0" reference point in the per-checkpoint sweep.
    if args.checkpoint_every > 0:
        torch.save({
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "val_acc": 0.0,
            "val_f1": 0.0,
            "args": vars(args),
        }, os.path.join(args.output_dir, "kimcnn_epoch_0.pt"))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device, data.num_classes)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save every --checkpoint-every epochs, plus epoch 1 explicitly so
        # GRAPHIC has an "after one SGD epoch" point alongside epoch 0.
        if args.checkpoint_every > 0 and epoch != args.epochs and (
                epoch == 1 or epoch % args.checkpoint_every == 0):
            ckpt = os.path.join(args.output_dir, f"kimcnn_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "args": vars(args),
            }, ckpt)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f}  "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}  "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )

        final_metrics = (train_loss, train_acc, val_loss, val_acc, val_f1)

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": final_metrics[3],
        "val_f1": final_metrics[4],
        "args": vars(args),
    }, final_path)

    print(
        "RESULT: "
        f"dataset={args.dataset}, embedding_type={args.embedding_type}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, "
        f"scheduler={args.scheduler}, epochs={args.epochs}, "
        f"train_loss_final={final_metrics[0]:.5f}, "
        f"train_acc_final={final_metrics[1]:.5f}, "
        f"test_loss={final_metrics[2]:.5f}, "
        f"test_acc={final_metrics[3]:.5f}, "
        f"test_f1={final_metrics[4]:.5f}",
        flush=True)
    print(f"Final checkpoint at {final_path}")


if __name__ == "__main__":
    main()
