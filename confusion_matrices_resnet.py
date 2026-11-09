"""Build per-layer confusion matrices for a trained ResNet-50 on CIFAR-100.

For each requested layer of the (frozen) backbone, this script:
    1. Trains a small linear classifier attached at that layer using the
       paper's custom cross-entropy mix between true and predicted labels.
    2. Saves the row-normalised confusion matrix on the eval portion of the
       training set AND on the validation set (the CIFAR test split in this
       project; see the README methodology section).
    3. Saves loss / accuracy curves for the classifier training run.

Three splits are used:
    - ``trainloader``       : 40k training images, used to train the classifier.
    - ``eval_trainloader``  : 10k held-out evaluation portion of the training
                              set, used for classifier validation during training
                              and to build the matrix saved as
                              ``_use_trainset.npy``.
    - ``valloader``         : the validation set, used to build the matrix
                              saved as ``_use_valset.npy``.

Shared classifier-training and matrix-construction logic lives in
``confusion_matrix_pipeline``; this file only contains the ResNet-specific
data setup and per-checkpoint orchestration.
"""

from __future__ import annotations

import argparse
import os
import re
from glob import glob
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from natsort import natsorted
from torchinfo import summary

from confusion_matrix_pipeline import (
    build_confusion_matrix,
    plot_train_curves,
    train_linear_classifier,
)
from feature_extraction.hooks import capture_output_dims, register_hooks
from models.CustomLossFunction import CustomCrossEntropyLoss
from models.LinearClassifier import LinearClassifier
from utils import initialize_model, set_seeders


# Per-layer learning rates for the linear classifier. Keys match the layer names
# exposed by torchvision's ResNet (``model.layer1`` etc.).
RESNET_LAYER_LRS = {
    "layer1": 0.0001,
    "layer2": 0.0002,
    "layer3": 0.0006,
    "layer4": 0.001,
}
DEFAULT_LR = 0.001  # used for any layer not in the dict above
CLASSIFIER_EPOCHS = 110
CLASSIFIER_PATIENCE = 5
NUM_CLASSES = 100
BATCH_SIZE = 2000


def create_confusion_matrix(model, output_dim, trainloader, valloader,
                            eval_trainloader, alpha, layer, weight_decay,
                            device):
    """Train a linear classifier at the given layer and return both confusion matrices.

    Returns:
        Tuple of (C_train, C_val, losses, val_losses, train_accs, val_accs).
        ``C_train`` is built on the eval portion of the training set (saved
        with suffix ``_use_trainset.npy``); ``C_val`` is built on the
        validation set (saved with suffix ``_use_valset.npy``).
    """
    print(f"Initialize the linear classifier on layer {layer!r}.")
    classifier = LinearClassifier(model, layer, output_dim, NUM_CLASSES).to(device)

    lr = RESNET_LAYER_LRS.get(layer, DEFAULT_LR)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=np.sqrt(0.1), patience=3
    )
    criterion = CustomCrossEntropyLoss(alpha)

    model.eval()  # backbone is frozen; keep it that way through the run.

    def process_batch(batch):
        """Run the classifier and the frozen backbone on one image batch."""
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        logits = classifier(images)
        with torch.no_grad():
            pred_labels = torch.argmax(model(images), dim=1)
        return logits, labels, pred_labels

    def keep_backbone_in_eval(lc):
        """The linear classifier's wrapped backbone slice must stay in eval()."""
        lc.features.eval()

    losses, val_losses, train_accs, val_accs = train_linear_classifier(
        classifier=classifier,
        train_loader=trainloader,
        val_loader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=CLASSIFIER_EPOCHS,
        patience=CLASSIFIER_PATIENCE,
        device=device,
        process_batch=process_batch,
        train_mode_setup=keep_backbone_in_eval,
        log_prefix=f"[layer {layer}] ",
    )

    def unpack_images_labels(batch):
        return batch[0], batch[1]

    C_train = build_confusion_matrix(
        classifier, eval_trainloader, NUM_CLASSES, device, unpack_images_labels
    )
    C_val = build_confusion_matrix(
        classifier, valloader, NUM_CLASSES, device, unpack_images_labels
    )

    return C_train, C_val, losses, val_losses, train_accs, val_accs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-layer confusion matrices for a frozen ResNet using a linear classifier."
    )
    parser.add_argument("-m", "--model", type=str, help="Model to be used (resnet50).")
    parser.add_argument("--small_conv1", action="store_true",
                        help="Use 3x3 conv1 instead of 7x7 as first layer.")
    parser.add_argument("--only", type=str, nargs="*",
                        help="Restrict to these layer names.")
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory of .pth checkpoints; matrices are generated for each.")
    parser.add_argument("--print_modules_only", action="store_true",
                        help="Print the model summary + layer names and exit.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save confusion matrices and curves to.")
    parser.add_argument("--epoch_step", type=int, default=1,
                        help="Number of epochs between checkpoints to process.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="First epoch index to process (inclusive).")
    parser.add_argument("--end_index", type=int, default=None,
                        help="Last epoch index to process (exclusive).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CE mix between true and predicted labels.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of training repetitions per layer.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes in the dataset.")
    parser.add_argument("--weight_decay", type=float, nargs="?", const=1e-4, default=0.0,
                        help="Weight decay (L2) for the classifier optimizer.")
    return parser.parse_args()


def build_cifar100_loaders():
    """Datasets/loaders for the ResNet classifiers. CIFAR-100 only."""
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=base_transform
    )
    train_trainset = torch.utils.data.Subset(trainset, range(40_000))
    eval_trainset = torch.utils.data.Subset(trainset, range(40_000, 50_000))
    valset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=base_transform
    )

    trainloader = torch.utils.data.DataLoader(train_trainset, batch_size=BATCH_SIZE, shuffle=True)
    eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=BATCH_SIZE, shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
    return trainloader, eval_trainloader, valloader


def select_checkpoints(checkpoint_dir: Path, start_index, end_index, step):
    """Return the ``.pth`` checkpoints whose epoch index matches the requested range."""
    all_checkpoints = natsorted(glob(str(checkpoint_dir / "*.pth"), recursive=True))
    epoch_to_ckpt = {}
    for path in all_checkpoints:
        match = re.search(r"epoch[_-]?(\d+)", path)
        if match:
            epoch_to_ckpt[int(match.group(1))] = path

    if end_index is None:
        end_index = max(epoch_to_ckpt.keys(), default=-1) + 1
    selected_epochs = list(range(start_index, end_index, step))
    return [epoch_to_ckpt[e] for e in selected_epochs if e in epoch_to_ckpt]


def main():
    set_seeders()
    args = parse_args()
    pprint(args)

    if not args.checkpoint_dir:
        raise ValueError("You must provide --checkpoint_dir.")

    trainloader, eval_trainloader, valloader = build_cifar100_loaders()
    checkpoint_paths = select_checkpoints(
        Path(args.checkpoint_dir), args.start_index, args.end_index, args.epoch_step
    )
    print(f"Using {len(checkpoint_paths)} checkpoints:")
    pprint(checkpoint_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha_dir = f"{args.output_dir}/alpha_{args.alpha}"
    os.makedirs(alpha_dir, exist_ok=True)
    os.makedirs(f"{alpha_dir}/train_curves", exist_ok=True)

    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        model = initialize_model(args)
        model.eval().to(device)

        output_dims = capture_output_dims(model, only=args.only, device=device)
        _, modules = register_hooks(model, only=args.only)
        layers = list(modules.keys())
        model.to(device)
        print(f"Layers to extract: {layers}")

        if args.print_modules_only:
            summary(model, input_size=(BATCH_SIZE, 3, 32, 32))
            print("Modules in model:")
            pprint(modules.keys())
            print("Output dimensions:")
            pprint(output_dims)
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        checkpoint_name = checkpoint_path.stem

        for iteration in range(args.iterations):
            print(f"\n---------------------- Iteration {iteration} ----------------------")
            for layer in layers:
                print(f"\nGenerating confusion matrix for layer {layer} "
                      f"in iteration {iteration} (checkpoint {checkpoint_name}).")
                C_train, C_val, losses, val_losses, train_acc, val_acc = create_confusion_matrix(
                    model=model,
                    output_dim=output_dims[layer],
                    trainloader=trainloader,
                    valloader=valloader,
                    eval_trainloader=eval_trainloader,
                    alpha=args.alpha,
                    layer=layer,
                    weight_decay=args.weight_decay,
                    device=device,
                )
                np.save(
                    f"{alpha_dir}/{checkpoint_name}_iteration_{iteration}_{layer}_use_valset.npy",
                    C_val.numpy(),
                )
                np.save(
                    f"{alpha_dir}/{checkpoint_name}_iteration_{iteration}_{layer}_use_trainset.npy",
                    C_train.numpy(),
                )
                plot_train_curves(
                    losses, val_losses, train_acc, val_acc,
                    output_path=f"{alpha_dir}/train_curves/{checkpoint_name}_iteration_{iteration}_layer_{layer}_metrics.png",
                    title_suffix=f"{checkpoint_name} {layer} iteration {iteration}",
                )


if __name__ == "__main__":
    main()
