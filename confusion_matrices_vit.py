"""Build per-block confusion matrices for a trained EffVit on CIFAR-100 or Tiny ImageNet.

Unlike the ResNet pipeline, this script first extracts per-block class-token
features for every image in the train / eval_train / val loaders and caches
them to disk. A linear classifier is then trained on those cached features per
block. The two-stage design is required because the backbone is expensive
to run repeatedly.

Three splits are used:
    - ``trainloader``       : 40k training images, used to train the classifier.
    - ``eval_trainloader``  : 10k held-out evaluation portion of the training
                              set, used for classifier validation during training
                              and to build the matrix saved as
                              ``_use_trainset.npy``.
    - ``valloader``         : the validation set (the CIFAR test split in
                              this project), used to build the matrix saved
                              as ``_use_valset.npy``.

Shared classifier-training and matrix-construction logic lives in
``confusion_matrix_pipeline``; this file only contains the Vit-specific
feature extraction, dataset setup, and per-checkpoint orchestration.
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from natsort import natsorted
from torchinfo import summary

from confusion_matrix_pipeline import (
    build_confusion_matrix,
    plot_train_curves,
    train_linear_classifier,
)
from custom_datasets.activation_dataset import ActivationDataset
from feature_extraction.hooks import register_hooks
from models.CustomLossFunction import CustomCrossEntropyLoss
from utils import initialize_model, set_seeders


# Probe hyperparameters for the Vit. These differ from the ResNet classifier
# because the input features are higher-dimensional and a fixed (rather
# than per-block) LR works in practice.
CLASSIFIER_LR = 0.01
CLASSIFIER_EPOCHS = 200
CLASSIFIER_PATIENCE = 15

# Block indices to extract features from, keyed by the model's depth.
DEPTH_TO_BLOCKS = {
    12: [0, 2, 5, 8, 11],
    8: [0, 1, 3, 5, 7],
    4: [0, 1, 2, 3],
}

# Tags used in the on-disk feature cache file names.
_CACHE_SPLITS = ("train", "eval_train", "val")


def extract_features_for_blocks(model, dataloader, device, blocks):
    """Run the model once and collect per-block class-token features.

    Returns:
        (features_per_block, labels, predictions)
    """
    activation_dict, _ = register_hooks(model, only=[f"blocks.{b}" for b in blocks])
    features = {b: torch.empty(0, model.embed_dim) for b in blocks}
    labels = torch.empty(0)
    predictions = torch.empty(0)
    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            preds = model(images)
            predicted_class = preds.argmax(1)
            labels = torch.cat([labels, label])
            predictions = torch.cat([predictions, predicted_class.to("cpu")])
            for block in blocks:
                # Use the class token only (index 0 in the sequence).
                features[block] = torch.cat([
                    features[block],
                    torch.flatten(activation_dict[f"blocks.{block}"][:, 0].to("cpu"), 1),
                ])
    return features, labels, predictions


def cache_features_for_split(model, dataloader, device, blocks, alpha_dir, checkpoint_name, split_name):
    """Run feature extraction for one split and write the result to disk."""
    feats, labels, preds = extract_features_for_blocks(model, dataloader, device, blocks)
    for block, tensor in feats.items():
        torch.save(tensor, f"{alpha_dir}/{checkpoint_name}_{split_name}_features_{block}.pt")
    torch.save(labels, f"{alpha_dir}/{checkpoint_name}_{split_name}_labels.pt")
    torch.save(preds, f"{alpha_dir}/{checkpoint_name}_{split_name}_preds.pt")


def load_activation_loaders(alpha_dir, checkpoint_name, block, batch_size):
    """Load cached features+labels+preds for one block as DataLoaders.

    Returns ``(act_trainloader, act_eval_trainloader, act_valloader)`` —
    one loader per split, matching the order of ``_CACHE_SPLITS``.
    """
    def _loader(split):
        feats = torch.load(f"{alpha_dir}/{checkpoint_name}_{split}_features_{block}.pt")
        labels = torch.load(f"{alpha_dir}/{checkpoint_name}_{split}_labels.pt")
        preds = torch.load(f"{alpha_dir}/{checkpoint_name}_{split}_preds.pt")
        return torch.utils.data.DataLoader(
            ActivationDataset(feats, labels, preds), batch_size=batch_size, shuffle=False
        )

    return _loader("train"), _loader("eval_train"), _loader("val")


def create_confusion_matrix(args, model, block, act_trainloader, act_eval_trainloader,
                            act_valloader, alpha, weight_decay, device):
    """Train a linear classifier on the cached features for one block.

    Returns:
        Tuple of (C_train, C_val, losses, val_losses, train_accs, val_accs).
        ``C_train`` is built on the eval portion of the training set (saved
        with suffix ``_use_trainset.npy``); ``C_val`` is built on the
        validation set (saved with suffix ``_use_valset.npy``).
    """
    classifier = nn.Linear(in_features=model.embed_dim,
                           out_features=args.num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=CLASSIFIER_LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=np.sqrt(0.1), patience=3
    )
    criterion = CustomCrossEntropyLoss(alpha).to(device)

    def process_batch(batch):
        """Cached features already include pred_labels; just classify."""
        features, labels, pred_labels = batch
        features = features.to(device)
        labels = labels.to(device)
        pred_labels = pred_labels.to(device)
        logits = classifier(features)
        return logits, labels, pred_labels

    losses, val_losses, train_accs, val_accs = train_linear_classifier(
        classifier=classifier,
        train_loader=act_trainloader,
        val_loader=act_eval_trainloader,  # eval portion of train steers classifier selection
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=CLASSIFIER_EPOCHS,
        patience=CLASSIFIER_PATIENCE,
        device=device,
        process_batch=process_batch,
        log_prefix=f"[block {block}] ",
    )

    def unpack_features_labels(batch):
        return batch[0], batch[1]

    C_train = build_confusion_matrix(
        classifier, act_eval_trainloader, args.num_classes, device, unpack_features_labels
    )
    C_val = build_confusion_matrix(
        classifier, act_valloader, args.num_classes, device, unpack_features_labels
    )
    return C_train, C_val, losses, val_losses, train_accs, val_accs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-block confusion matrices for a frozen Vit using a linear classifier."
    )
    parser.add_argument("-m", "--model", type=str, help="Model to be used (efficient-vit).")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a single model checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory of .pth checkpoints; matrices are generated for each.")
    parser.add_argument("--print_modules_only", action="store_true",
                        help="Print the model summary and exit.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save confusion matrices and curves to.")
    parser.add_argument("--epoch_step", type=int, default=1,
                        help="Number of checkpoints to skip between processed ones.")
    parser.add_argument("--start_index", type=int, default=0,
                        help="First checkpoint index to process.")
    parser.add_argument("--end_index", type=int, default=None,
                        help="Last checkpoint index to process.")
    parser.add_argument("--use_n_blocks", type=int, default=None,
                        help="Number of transformer blocks before the cut-off.")
    parser.add_argument("--label_type", type=str, default="true",
                        help="Label type for the confusion matrix ('true' or 'pred').")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CE mix between true and predicted labels.")
    parser.add_argument("--num_classes", type=int, default=100,
                        help="Number of classes in the dataset.")
    parser.add_argument("--depth", type=int, default=12,
                        help="Depth parameter for the VisionTransformer.")
    parser.add_argument("--dataset", type=str, default="cifar",
                        help="Dataset to use (cifar / cifar100 / tinyimagenet).")
    parser.add_argument("--batch_size", type=int, default=400,
                        help="Batch size for training and feature extraction.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Spatial size of the model input (matches the dataset Resize).")
    parser.add_argument("--weight_decay", type=float, nargs="?", const=1e-4, default=0.0,
                        help="Weight decay (L2) for the classifier optimizer.")
    return parser.parse_args()


def build_loaders(args, generator):
    """CIFAR-100 or Tiny ImageNet loaders for the Vit pipeline.

    Returns ``(trainloader, eval_trainloader, valloader)``.
    """
    if args.dataset in ("cifar", "cifar100", "CIFAR", "CIFAR100"):
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=3),
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

        trainloader = torch.utils.data.DataLoader(train_trainset, batch_size=args.batch_size, shuffle=True)
        eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        return trainloader, eval_trainloader, valloader

    if args.dataset in ("tinyimgnet", "tinyimagenet"):
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=3),
            transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                 std=torch.tensor([0.229, 0.224, 0.225])),
        ])
        full_trainset = torchvision.datasets.ImageFolder("./tiny-imagenet-200/train", transform=base_transform)
        indices = torch.randperm(len(full_trainset), generator=generator).tolist()
        shuffled = torch.utils.data.Subset(full_trainset, indices)

        split = int(0.8 * len(shuffled))
        train_trainset = torch.utils.data.Subset(shuffled, range(split))
        eval_trainset = torch.utils.data.Subset(shuffled, range(split, len(shuffled)))
        valset = torchvision.datasets.ImageFolder("./tiny-imagenet-200/val", transform=base_transform)

        trainloader = torch.utils.data.DataLoader(train_trainset, batch_size=args.batch_size, shuffle=False)
        eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        return trainloader, eval_trainloader, valloader

    raise ValueError(f"Unsupported dataset: {args.dataset!r}")


def cleanup_block_cache(alpha_dir, checkpoint_name, block):
    """Delete per-block per-checkpoint feature caches after we're done with them."""
    for split in _CACHE_SPLITS:
        path = f"{alpha_dir}/{checkpoint_name}_{split}_features_{block}.pt"
        if os.path.exists(path):
            os.remove(path)


def cleanup_checkpoint_cache(alpha_dir, checkpoint_name):
    """Delete the per-checkpoint label/pred caches."""
    for split in _CACHE_SPLITS:
        for suffix in ("labels", "preds"):
            path = f"{alpha_dir}/{checkpoint_name}_{split}_{suffix}.pt"
            if os.path.exists(path):
                os.remove(path)


def select_checkpoints(args):
    if args.checkpoint_path:
        return [args.checkpoint_path]
    if not args.checkpoint_dir:
        return []
    all_checkpoints = natsorted(glob(os.path.join(args.checkpoint_dir, "*.pth")))
    return all_checkpoints[args.start_index:args.end_index:args.epoch_step]


def main():
    set_seeders()
    generator = torch.Generator().manual_seed(42)
    args = parse_args()
    pprint(args)

    trainloader, eval_trainloader, valloader = build_loaders(args, generator)
    checkpoint_paths = select_checkpoints(args)
    print(f"Using {len(checkpoint_paths)} checkpoints:")
    pprint(checkpoint_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha_dir = f"{args.output_dir}/alpha_{args.alpha}"
    os.makedirs(alpha_dir, exist_ok=True)
    os.makedirs(f"{alpha_dir}/train_curves", exist_ok=True)

    if args.depth not in DEPTH_TO_BLOCKS:
        raise ValueError(
            f"Unsupported --depth {args.depth}; supported depths are "
            f"{sorted(DEPTH_TO_BLOCKS.keys())}"
        )
    blocks = DEPTH_TO_BLOCKS[args.depth]

    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        model = initialize_model(args)
        model.eval().to(device)

        if args.print_modules_only:
            _, modules = register_hooks(model)
            summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size))
            print("Modules:")
            pprint(modules)
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=False)["model"]
        )
        checkpoint_name = checkpoint_path.stem

        # Pre-extract features for all three splits and cache them on disk so
        # classifier training can iterate over them many times cheaply.
        cache_features_for_split(model, trainloader, device, blocks, alpha_dir, checkpoint_name, "train")
        cache_features_for_split(model, eval_trainloader, device, blocks, alpha_dir, checkpoint_name, "eval_train")
        cache_features_for_split(model, valloader, device, blocks, alpha_dir, checkpoint_name, "val")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for block in blocks:
            print(f"\n---------------------- Block {block} ----------------------")
            act_trainloader, act_eval_trainloader, act_valloader = load_activation_loaders(
                alpha_dir, checkpoint_name, block, args.batch_size
            )

            C_train, C_val, losses, val_losses, train_acc, val_acc = create_confusion_matrix(
                args=args,
                model=model,
                block=block,
                act_trainloader=act_trainloader,
                act_eval_trainloader=act_eval_trainloader,
                act_valloader=act_valloader,
                alpha=args.alpha,
                weight_decay=args.weight_decay,
                device=device,
            )

            np.save(
                f"{alpha_dir}/{checkpoint_name}_iteration_0_block_{block}_use_valset.npy",
                C_val.numpy(),
            )
            np.save(
                f"{alpha_dir}/{checkpoint_name}_iteration_0_block_{block}_use_trainset.npy",
                C_train.numpy(),
            )
            plot_train_curves(
                losses, val_losses, train_acc, val_acc,
                output_path=f"{alpha_dir}/train_curves/{checkpoint_name}_iteration_0_layer_{block}_metrics.png",
                title_suffix=f"{checkpoint_name} block {block}",
            )
            cleanup_block_cache(alpha_dir, checkpoint_name, block)

        cleanup_checkpoint_cache(alpha_dir, checkpoint_name)


if __name__ == "__main__":
    main()
