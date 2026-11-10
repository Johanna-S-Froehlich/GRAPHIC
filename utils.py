import os
import random
import re

import torch
from natsort import natsorted
from numpy.dtypes import StringDType
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from models.resnet import create_resnet
from VitStandalone.models import deit_small_patch16_224
from transforms.cifar100 import cifar_transforms

from torch import optim as optim, nn
import numpy as np

# Remove BatchNorm layers from weight decay and layer adaptation.
# Adapted from https://theaisummer.com/simclr/.
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def define_param_groups(model, weight_decay):
    """Split model parameters into two optimizer groups: decayed and BN.

    BatchNorm scale/shift parameters should not be weight-decayed — doing so
    interferes with the normalization statistics and tends to degrade
    performance. We identify BN parameters by checking the type of the
    parent module (so the classification is robust to module naming).
    """
    bn_param_ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, _BN_TYPES)
        for p in module.parameters(recurse=False)
    }
    decayed = [p for p in model.parameters() if id(p) not in bn_param_ids]
    bn_params = [p for p in model.parameters() if id(p) in bn_param_ids]

    return [
        {
            "params": decayed,
            "weight_decay": weight_decay,
            "layer_adaptation": True,
        },
        {
            "params": bn_params,
            "weight_decay": 0.0,
            "layer_adaptation": False,
        },
    ]

def initialize_optimizer(args, model):
    """
    Initializes and returns the optimizer based on the provided arguments.

    Args:
        args (Namespace): A namespace object containing model and optimizer configurations.
        model (torch.nn.Module): The model to be optimized.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Explanation:
    - Supports Adam optimizer.
    - Uses parameter groups to exclude BatchNorm layers from weight decay.
    - Raises an error if an unsupported optimizer is specified.

    Why use parameter groups?
    - Allows fine-grained control over different subsets of parameters (e.g., applying weight decay only where needed).
    """
    # Select the appropriate optimizer and weight decay
    optimizer_name = args.optimizer
    weight_decay = args.weight_decay

    # Define parameter groups, excluding BatchNorm layers and optionally bias terms
    param_groups = define_param_groups(model, weight_decay)

    # Initialize the optimizer based on the specified name
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,                    # Use custom parameter groups
            lr=args.lr,  # Learning rate
            weight_decay=weight_decay        # Global weight decay (handled by parameter groups)
        )
    else:
        # Raise an error if an invalid optimizer is specified
        raise ValueError('Invalid optimizer')

    return optimizer

def initialize_scheduler(args, optimizer):
    """
    Initializes and returns a learning rate scheduler based on the provided arguments.

    Args:
        args (Namespace): A namespace object containing model hyperparameters and configurations.
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler will adjust the learning rate.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: A learning rate scheduler if a valid one is specified, otherwise None.

    """
    # Choose the scheduler
    scheduler_name = args.scheduler

    # If no scheduler is specified, return None (no learning rate adjustments)
    if scheduler_name != 'None':

        if scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=np.sqrt(0.1),          # Multiplicative factor for reducing LR (≈0.316)
                patience=5,                   # Number of epochs with no improvement before reducing LR
                min_lr=0.5e-6                 # Minimum learning rate to maintain
            )
        else:
            raise ValueError('Invalid scheduler')

        return scheduler

    else:
        # Return None if no scheduler is specified
        return None


def initialize_model(args):
    """
    Initializes and returns a model based on the provided arguments.

    Args:
        args (Namespace): A namespace object containing model configuration parameters.

    Returns:
        torch.nn.Module: The initialized model.

    Explanation:
    - Supports ResNet-50 and EfficientVit model initialization.
    - Optionally modifies the first convolutional layer (`conv1`) for smaller inputs (e.g., CIFAR-100).
    - Raises an error if an invalid model is specified.

    Example:
    - For CIFAR-100, the `small_conv1` flag adjusts the input convolution to a 3x3 kernel with stride 1.
    """
    if args.model == 'resnet50':
        # Initialize a ResNet-50 model with 100 output classes
        model = create_resnet(num_classes=args.num_classes, model_name=args.model)

        # Adjust the first convolution layer for smaller inputs (e.g., CIFAR-100 images of size 32x32)
        if args.small_conv1: 
            model.conv1 = nn.Conv2d(
                3,               # Input channels (e.g., RGB images have 3 channels)
                64,              # Output channels (ResNet-50 uses 64 channels in the first layer)
                kernel_size=(3, 3),  # 3x3 convolution (smaller receptive field for smaller images)
                stride=(1, 1),   # No downsampling (preserves input dimensions)
                padding=(1, 1),  # Ensures output spatial size remains the same
                bias=False       # No bias needed due to Batch Normalization
            )
    elif args.model == 'efficient-vit':
        # depth and use_n_blocks live on the Vit confusion-matrix script's
        # argparse but not on every entry point that calls initialize_model;
        # fall back to sensible defaults so resnet-only callers don't crash.
        model = deit_small_patch16_224(
            img_size=224,
            depth=getattr(args, 'depth', 12),
            localvit=True,
            localvit_act='hs',
            postnorm=False,
            layerscale=False,
            ls_init=1e-5,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            return_after_n_blocks=getattr(args, 'use_n_blocks', None),
        )
    else:
        # Raise an error if an unsupported model is specified
        raise ValueError('Invalid model')

    return model


def build_dataset(args):
    """Build CIFAR-100 training and validation DataLoaders.

    The dataset is downloaded to (and cached under) ``./data``. Practitioners
    don't get to pick the path on purpose — one less knob to misconfigure.

    Returns:
        (train_loader, val_loader)
    """
    transform_train, transform_val = cifar_transforms(args)

    train_dataset = CIFAR100(root='./data', train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True, worker_init_fn=seed_worker)

    val_dataset = CIFAR100(root='./data', train=False, transform=transform_val, download=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)

    return train_loader, val_loader


def set_seeders(seed=42):
    """Set all relevant seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    :param worker_id: worker_id [0, ..., NUM_WORKER - 1]
    :return:
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def file_matrix(matrix_files):
    """Index confusion-matrix files by (epoch, iteration, layer, use_val).

    The boolean axis distinguishes ``_use_trainset.npy`` (False) from the
    held-out partition (True). The held-out partition's filename suffix is
    ``_use_valset.npy`` for the image-classification pipeline (resnet/vit;
    a true validation set) and ``_use_testset.npy`` for the text-
    classification pipeline (CNN/GCN under the no-early-stop methodology
    where the held-out partition is a genuine test set).

    Layers that exist only for the held-out partition (e.g. the output layer
    of the text models) get an empty string for the training partition.

    :param matrix_files: a list of file paths to the confusion matrices.
    :return: (file_matrix, unique_epochs, unique_iterations, unique_layers)
    """
    unique_epochs = set()
    unique_iterations = set()
    unique_layers = set()

    matrix_dict = {}
    for f in matrix_files:
        # Only the file name counts, the folders may be named anything.
        name = os.path.basename(f)
        epoch_match = re.search(r'epoch_([^_]+)', name)
        if epoch_match is not None:
            epoch = epoch_match.group(1)
        else:
            epoch = 'random'
        iteration = re.search(r'iteration_([^_-]+)', name).group(1)
        layer = re.search(r'layer([^_]+)', name)
        if layer is None:
            layer = re.search(r'block_([^_]+)', name)
        if layer is None:
            raise ValueError(f'Could not find layer in file name {f}')
        layer = layer.group(1)
        use_val = name.endswith(('_use_valset.npy', '_use_testset.npy'))
        unique_epochs.add(epoch)
        unique_iterations.add(iteration)
        unique_layers.add(layer)

        matrix_dict[(epoch, iteration, layer, use_val)] = f

    unique_epochs = natsorted(list(unique_epochs))
    unique_iterations = natsorted(list(unique_iterations))
    unique_layers = natsorted(list(unique_layers))
    fm = np.zeros((len(unique_epochs), len(unique_iterations), len(unique_layers), 2), dtype=StringDType)

    for i, epoch in enumerate(unique_epochs):
        for j, iteration in enumerate(unique_iterations):
            for k, layer in enumerate(unique_layers):
                for l, use_val in enumerate([False, True]):
                    fm[i, j, k, l] = matrix_dict.get((epoch, iteration, layer, use_val), "")

    return fm, unique_epochs, unique_iterations, unique_layers
