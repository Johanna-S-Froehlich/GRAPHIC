from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    def __init__(self, activations, labels, pred_labels=None):
        """
        A custom dataset class for handling activations, labels, and optional predicted labels.
        For use in combination with the CustomLossFunction for computing confusion matrices.
        Args:
            activations (list or tensor): The activation data, typically a list or tensor of model outputs.
            labels (list or tensor): The true labels corresponding to the activations.
            pred_labels (list or tensor, optional): The predicted labels, if available. Defaults to None.
        """
        self.activations = activations
        self.labels = labels.long()
        self.pred_labels = pred_labels.long() if pred_labels is not None else None

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return (self.activations[idx], self.labels[idx]) if self.pred_labels is None else (
            self.activations[idx], self.labels[idx], self.pred_labels[idx])