"""Shared helpers for the confusion-matrix entry points.

All confusion-matrix scripts train a small linear classifier on top of a
frozen feature representation and then build a class x class confusion
matrix from the classifier's predictions. The pipelines differ in how they
expose those features (ResNet wraps a backbone slice in
``LinearClassifier``; the Vit, KimCNN and Text GCN pipelines pre-extract the
activations), but the classifier-training and matrix-construction steps are
the same in structure.

The image scripts (ResNet, Vit) train the classifier with validation-loss
early stopping (``train_linear_classifier``). The text scripts (KimCNN,
Text GCN) train it for a fixed number of epochs under a cosine schedule
(``train_linear_classifier_fixed_epochs``).

This module captures the shared steps as a tiny set of callbacks-driven
helpers so each script can stay focused on its model-specific setup.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# A callback that converts one batch from the loader into the three tensors
# the loss needs: classifier logits, true labels, pred labels (from the
# frozen model). Hides the difference between "images go in, backbone runs
# inline" (ResNet) and "features already extracted, just classify them" (Vit).
BatchProcessor = Callable[[object], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class BestWeightsTracker:
    """Tracks the lowest validation loss seen and the weights that produced it.

    Also implements simple patience-based early stopping: ``should_stop()``
    returns True once ``patience`` epochs have passed without improvement.

    The weights are copied: ``state_dict()`` returns the live parameter
    tensors, which further training would overwrite.
    """

    def __init__(self, initial_weights: dict, patience: int):
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_weights = copy.deepcopy(initial_weights)
        self.patience = patience
        self._epochs_since_improvement = 0

    def update(self, val_loss: float, epoch: int, weights: dict) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(weights)
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1

    def should_stop(self) -> bool:
        return self._epochs_since_improvement >= self.patience


def train_linear_classifier(
    classifier: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    patience: int,
    device: torch.device,
    process_batch: BatchProcessor,
    train_mode_setup: Callable[[nn.Module], None] = None,
    log_prefix: str = "",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train a linear classifier with validation-loss early stopping.

    Args:
        classifier: the classifier to train.
        train_loader / val_loader: data sources. The shape of each batch is
            opaque to this function; ``process_batch`` interprets them.
        criterion: a CustomCrossEntropyLoss-style callable that takes
            (logits, true_labels, pred_labels) and returns a scalar loss.
        optimizer / scheduler: standard PyTorch instances. The scheduler is
            stepped once per epoch with ``avg_val_loss``.
        epochs: maximum number of epochs to train.
        patience: stop after this many epochs without validation improvement.
        device: where everything lives (assumed already moved).
        process_batch: callback that turns one batch into
            ``(classifier_logits, true_labels, pred_labels)``.
        train_mode_setup: optional callback for any per-epoch setup the caller
            needs to run after putting the classifier in train() mode (the
            ResNet classifier needs to keep the wrapped backbone in eval()).
        log_prefix: string prepended to per-epoch print lines.

    Returns:
        (losses, val_losses, train_accuracies, val_accuracies) over the
        epochs actually trained, and loads the best weights back into the
        classifier in place.
    """
    losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []

    tracker = BestWeightsTracker(classifier.state_dict(), patience)

    for epoch in range(epochs):
        # --- train ---
        classifier.train()
        if train_mode_setup is not None:
            train_mode_setup(classifier)

        epoch_loss = 0.0
        train_correct = 0
        train_pred_correct = 0
        train_total = 0
        for batch in train_loader:
            logits, true_labels, pred_labels = process_batch(batch)
            loss = criterion(logits, true_labels, pred_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == true_labels).sum().item()
            train_pred_correct += (preds == pred_labels).sum().item()
            train_total += true_labels.size(0)

        avg_loss = epoch_loss / len(train_loader)
        true_train_acc = train_correct / train_total
        pred_train_acc = train_pred_correct / train_total
        # Match the reporting convention used in both original scripts.
        train_accuracy = criterion.lam * true_train_acc + (1 - criterion.lam) * pred_train_acc

        # --- validate ---
        classifier.eval()
        eval_loss = 0.0
        true_correct = 0
        pred_correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, true_labels, pred_labels = process_batch(batch)
                loss = criterion(logits, true_labels, pred_labels)
                eval_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                true_correct += (preds == true_labels).sum().item()
                pred_correct += (preds == pred_labels).sum().item()
                total += true_labels.size(0)

        avg_val_loss = eval_loss / len(val_loader)
        true_val_acc = true_correct / total
        pred_val_acc = pred_correct / total
        val_accuracy = criterion.lam * true_val_acc + (1 - criterion.lam) * pred_val_acc

        losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        scheduler.step(avg_val_loss)
        tracker.update(avg_val_loss, epoch, classifier.state_dict())

        print(
            f"{log_prefix}Epoch {epoch + 1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6g} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}",
            flush=True,
        )

        if tracker.should_stop():
            break

    print(f"{log_prefix}Best classifier epoch was {tracker.best_epoch + 1}. Loading best weights.", flush=True)
    classifier.load_state_dict(tracker.best_weights)
    return losses, val_losses, train_accuracies, val_accuracies


def train_linear_classifier_fixed_epochs(
    classifier: nn.Module,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    process_batch: BatchProcessor,
    train_mode_setup: Callable[[nn.Module], None] = None,
    log_prefix: str = "",
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Train a linear classifier for a fixed number of epochs (text pipelines).

    No early stopping, no best-weight selection: the classifier's final-epoch
    state is the trained model. The held-out loader is evaluated each epoch
    for monitoring only; it never feeds back into selection or scheduling,
    so it can be a true test set.

    Args:
        classifier: the classifier to train.
        train_loader / test_loader: data sources. The shape of each batch is
            opaque to this function; ``process_batch`` interprets it.
        criterion: a CustomCrossEntropyLoss-style callable that takes
            (logits, true_labels, pred_labels) and returns a scalar loss.
        optimizer / scheduler: standard PyTorch instances. The scheduler is
            stepped once per epoch with no argument (cosine-style).
        epochs: number of epochs to train.
        device: where everything lives (assumed already moved).
        process_batch: callback that turns one batch into
            ``(classifier_logits, true_labels, pred_labels)``.
        train_mode_setup: optional callback for any per-epoch setup the caller
            needs to run after putting the classifier in train() mode.
        log_prefix: string prepended to per-epoch print lines.

    Returns:
        (train_losses, test_losses, train_accuracies, test_accuracies,
        train_macro_f1s, test_macro_f1s). Macro F1 is computed against the
        ground-truth labels each epoch on both partitions.
    """
    # Imported here so that the image pipelines do not load scikit-learn.
    from sklearn.metrics import f1_score

    train_losses: List[float] = []
    test_losses: List[float] = []
    train_accuracies: List[float] = []
    test_accuracies: List[float] = []
    train_macro_f1s: List[float] = []
    test_macro_f1s: List[float] = []

    for epoch in range(epochs):
        classifier.train()
        if train_mode_setup is not None:
            train_mode_setup(classifier)

        epoch_loss = 0.0
        train_correct = 0
        train_pred_correct = 0
        train_total = 0
        train_true_buf, train_pred_buf = [], []
        for batch in train_loader:
            logits, true_labels, pred_labels = process_batch(batch)
            loss = criterion(logits, true_labels, pred_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == true_labels).sum().item()
            train_pred_correct += (preds == pred_labels).sum().item()
            train_total += true_labels.size(0)
            train_true_buf.append(true_labels.detach().cpu().numpy())
            train_pred_buf.append(preds.detach().cpu().numpy())

        avg_train_loss = epoch_loss / len(train_loader)
        true_train_acc = train_correct / train_total
        pred_train_acc = train_pred_correct / train_total
        train_accuracy = criterion.lam * true_train_acc + (1 - criterion.lam) * pred_train_acc
        train_macro_f1 = float(f1_score(
            np.concatenate(train_true_buf), np.concatenate(train_pred_buf),
            average="macro", zero_division=0))

        classifier.eval()
        test_loss_sum = 0.0
        test_true_correct = 0
        test_pred_correct = 0
        test_total = 0
        test_true_buf, test_pred_buf = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits, true_labels, pred_labels = process_batch(batch)
                loss = criterion(logits, true_labels, pred_labels)
                test_loss_sum += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_true_correct += (preds == true_labels).sum().item()
                test_pred_correct += (preds == pred_labels).sum().item()
                test_total += true_labels.size(0)
                test_true_buf.append(true_labels.detach().cpu().numpy())
                test_pred_buf.append(preds.detach().cpu().numpy())

        avg_test_loss = test_loss_sum / len(test_loader)
        true_test_acc = test_true_correct / test_total
        pred_test_acc = test_pred_correct / test_total
        test_accuracy = criterion.lam * true_test_acc + (1 - criterion.lam) * pred_test_acc
        test_macro_f1 = float(f1_score(
            np.concatenate(test_true_buf), np.concatenate(test_pred_buf),
            average="macro", zero_division=0))

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_macro_f1s.append(train_macro_f1)
        test_macro_f1s.append(test_macro_f1)

        scheduler.step()

        print(
            f"{log_prefix}Epoch {epoch + 1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6g} | "
            f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | "
            f"Train F1: {train_macro_f1:.4f} | Test F1: {test_macro_f1:.4f}",
            flush=True,
        )

    return (train_losses, test_losses, train_accuracies, test_accuracies,
            train_macro_f1s, test_macro_f1s)


InputsLabelsExtractor = Callable[[object], Tuple[torch.Tensor, torch.Tensor]]


def build_confusion_matrix(
    classifier: nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    get_inputs_and_labels: InputsLabelsExtractor,
) -> torch.Tensor:
    """Run a trained classifier over ``loader`` and return a row-normalised
    confusion matrix of shape (num_classes, num_classes).

    ``get_inputs_and_labels(batch)`` returns ``(classifier_inputs, true_labels)``
    already on ``device`` (or, if not, this function will move them).
    """
    classifier.eval()
    C = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for batch in loader:
            inputs, labels = get_inputs_and_labels(batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=1)
            for true_label, pred in zip(labels.cpu(), preds.cpu()):
                C[int(true_label), int(pred)] += 1
    # Row-normalise so each row sums to 1 (or 0 if the row is empty).
    row_sums = C.sum(dim=1, keepdim=True)
    C = torch.where(row_sums > 0, C / row_sums, C)
    return C


def plot_train_curves(losses, val_losses, train_accuracy, val_accuracy,
                      output_path: str, title_suffix: str = "",
                      train_macro_f1=None, val_macro_f1=None,
                      heldout_name: str = "Validation") -> None:
    """Save side-by-side loss and accuracy curves for a classifier training run.

    With ``train_macro_f1`` and ``val_macro_f1`` a third panel shows macro F1.
    ``heldout_name`` labels the held-out curves ("Validation" for the image
    pipelines, "Test" for the text pipelines).
    """
    with_f1 = train_macro_f1 is not None and val_macro_f1 is not None
    fig, axes = plt.subplots(1, 3 if with_f1 else 2, figsize=(21 if with_f1 else 15, 6))

    axes[0].plot(losses, label="Training Loss", color="blue")
    axes[0].plot(val_losses, label=f"{heldout_name} Loss", color="orange")
    axes[0].set_title(f"Loss Curves - {title_suffix}" if title_suffix else "Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accuracy, label="Training Accuracy", color="green")
    axes[1].plot(val_accuracy, label=f"{heldout_name} Accuracy", color="red")
    axes[1].set_title(f"Accuracy Curves - {title_suffix}" if title_suffix else "Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    if with_f1:
        axes[2].plot(train_macro_f1, label="Training Macro F1", color="green")
        axes[2].plot(val_macro_f1, label=f"{heldout_name} Macro F1", color="red")
        axes[2].set_title(f"Macro F1 Curves - {title_suffix}" if title_suffix else "Macro F1 Curves")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1")
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
