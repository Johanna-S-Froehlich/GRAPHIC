import torch.nn as nn


class CustomCrossEntropyLoss(nn.Module):
    """Convex combination of cross-entropy on true and predicted labels.

    Used by the confusion-matrix scripts. ``lam=1.0`` recovers standard
    cross-entropy on the true labels.
    """

    def __init__(self, lam: float = 1.0):
        """
        Args:
            lam: Mixing weight in [0, 1] for the true-label cross-entropy.
                ``1.0`` is pure true-label CE; ``0.0`` is pure predicted-label CE.
        """
        super().__init__()
        self.lam = lam
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels, pred_labels):
        """
        Args:
            logits: Model outputs (logits).
            labels: True labels.
            pred_labels: Predicted labels from the frozen reference model.

        Returns:
            Tensor: Combined loss.
        """
        loss_true = self.loss_fn(logits, labels)
        loss_pred = self.loss_fn(logits, pred_labels)
        return self.lam * loss_true + (1 - self.lam) * loss_pred
