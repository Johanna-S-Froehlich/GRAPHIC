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
# Code taken from https://theaisummer.com/simclr/
def define_param_groups(model, weight_decay):
    """
    Defines parameter groups for the optimizer, excluding BatchNorm layers from weight decay and layer adaptation.

    Args:
        model (torch.nn.Module): The model whose parameters need to be grouped.
        weight_decay (float): The weight decay value for regularization.

    Returns:
        List[Dict[str, Any]]: A list of parameter groups for the optimizer.

    Explanation:
    - BatchNorm layers are excluded from weight decay and layer adaptation.
    Why exclude BatchNorm from weight decay?
    - BatchNorm parameters (scale and shift) should not be regularized because doing so may interfere with
      the statistical normalization process and degrade performance.
    """
    def exclude_from_wd_and_adaptation(name):
        """
        Helper function to determine whether a parameter should be excluded from weight decay and layer adaptation.

        Args:
            name (str): The name of the parameter.

        Returns:
            bool: True if the parameter should be excluded, False otherwise.

        Conditions:
        - Excludes BatchNorm layers (names containing 'bn').
        """
        # Exclude BatchNorm layers
        if 'bn' in name:
            return True
        return False
    
    # Create two parameter groups:
    param_groups = [
        # 1. Parameters that are subject to weight decay and layer adaptation
        {
            'params': [
                p for name, p in model.named_parameters()
                if not exclude_from_wd_and_adaptation(name)  # Include all parameters NOT excluded
            ],
            'weight_decay': weight_decay,
            'layer_adaptation': True,  # Enable layer adaptation (if supported by the optimizer)
        },
        # 2. Parameters excluded from weight decay and layer adaptation
        {
            'params': [
                p for name, p in model.named_parameters()
                if exclude_from_wd_and_adaptation(name)  # Include all excluded parameters
            ],
            'weight_decay': 0.,         # No weight decay applied
            'layer_adaptation': False,  # No layer adaptation applied
        },
    ]
    return param_groups

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
        model = deit_small_patch16_224(
            img_size=224,
            depth=args.depth,
            localvit=True,
            localvit_act='hs',
            postnorm=False,
            layerscale=False,
            ls_init=1e-5,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            return_after_n_blocks=args.use_n_blocks
        )
    else:
        # Raise an error if an unsupported model is specified
        raise ValueError('Invalid model')

    return model


def build_dataset(args):
    """
    Builds and returns the training and test datasets based on the specified dataset type in args.
    Args:
        args (Namespace): A namespace object containing dataset configuration parameters.
    Returns:
        tuple: (train_loader, test_loader) - Data loaders for training and testing datasets.
    """
    # Define transforms
    transform_train, transform_test = cifar_transforms(args)

    # Load CIFAR-100 dataset
    train_dataset = CIFAR100(root='./data', train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker)

    test_dataset = CIFAR100(root='./data', train=False, transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)


    return train_loader, test_loader


def set_seeders(seed=42):
    """
    Sets all relevant seeds to ensure reproducibility between experiments.
    """
    global GENERATOR
    random.seed(seed)
    np.random.seed(seed)

    GENERATOR = torch.Generator()
    GENERATOR.manual_seed(seed)

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
    """
    To make accessing the confusion matrices easier, this function returns a numpy array of the confusion matrice files indexed by epoch, iteration, and layer.
    :param matrix_files: a list of file paths to the confusion matrices
    :return: a numpy array of the confusion matrices indexed by epoch, iteration, and layer
    """

    unique_epochs = set()
    unique_iterations = set()
    unique_layers = set()

    matrix_dict = {}
    for f in matrix_files:
        epoch_match = re.search(r'epoch_([^_]+)', f)
        if epoch_match is not None:
            epoch = epoch_match.group(1)
        else:
            epoch = 'random'
        iteration = re.search(r'iteration_([^_-]+)', f).group(1)
        layer = re.search(r'layer([^_]+)', f)
        if layer is None:
            layer = re.search(r'block_([^_]+)', f)
        if layer is None:
            raise ValueError(f'Could not find layer in file name {f}')
        layer = layer.group(1)
        use_test = 'testset' in f
        unique_epochs.add(epoch)
        unique_iterations.add(iteration)
        unique_layers.add(layer)

        matrix_dict[(epoch, iteration, layer, use_test)] = f

    unique_epochs = natsorted(list(unique_epochs))
    unique_iterations = natsorted(list(unique_iterations))
    unique_layers = natsorted(list(unique_layers))
    file_matrix = np.zeros((len(unique_epochs), len(unique_iterations), len(unique_layers), 2), dtype=StringDType)

    for i, epoch in enumerate(unique_epochs):
        for j, iteration in enumerate(unique_iterations):
            for k, layer in enumerate(unique_layers):
                for l, use_test in enumerate([False, True]):
                    file_matrix[i, j, k, l] = matrix_dict[(epoch, iteration, layer, use_test)]

    return file_matrix, unique_epochs, unique_iterations, unique_layers
