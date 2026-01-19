import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, lam=1.0, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        """
        Custom CrossEntropyLoss using nn.CrossEntropyLoss instances.

        Args:
            lam (float): Weight for the true label loss. If lam=1, behaves like nn.CrossEntropyLoss.
            weight (Tensor, optional): Rescaling weight for each class.
            ignore_index (int, optional): Specifies a target value to ignore.
            reduction (str, optional): 'none', 'mean', or 'sum'.
            label_smoothing (float, optional): Label smoothing factor (default 0.0).
        """
        super(CustomCrossEntropyLoss, self).__init__()
        self.lam = lam

        # Initialize nn.CrossEntropyLoss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels, pred_labels):
        """
        Compute the custom loss.

        Args:
            logits (Tensor): Model outputs (logits).
            labels (Tensor): True labels.
            pred_labels (Tensor): Additional predicted labels.

        Returns:
            Tensor: Computed loss.
        """
        loss_true = self.loss_fn(logits, labels)
        loss_pred = self.loss_fn(logits, pred_labels)

        # Weighted sum of both losses
        return self.lam * loss_true + (1 - self.lam) * loss_pred
