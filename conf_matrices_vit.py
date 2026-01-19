import argparse
import os
from glob import glob
from pprint import pprint
from pathlib import Path

import torch
from torch import tensor
import torch.nn as nn
from torchinfo import summary
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import matplotlib.pyplot as plt

from utils import initialize_model, set_seeders
from feature_extraction.hooks import register_hooks
from custom_datasets.activation_dataset import ActivationDataset
from models.CustomLossFunction import CustomCrossEntropyLoss


#
#   Helper Methods
#
def create_confusion_matrix(args, model, block, trainloader, testloader, eval_trainloader, use_test_set_for_matrix: bool, alpha, weight_decay: float = 0.0, device=torch.device("cpu")):
    """
    Creates a confusion matrix for the given model and layer.

    Args:
        args (Namespace): Configuration parameters, e.g., dataset information, model settings, or other options.
        model (torch.nn.Module): The model to create the confusion matrix for.
        block (int): The block of the model to create the confusion matrix for.
        trainloader (torch.utils.data.DataLoader): The training dataloader.
        testloader (torch.utils.data.DataLoader): The test dataloader.
        eval_trainloader (torch.utils.data.DataLoader): The evaluation dataloader.
        use_test_set_for_matrix (bool): Whether to use the test set for the confusion matrix.
        alpha (float): The alpha value for the custom loss function.
        weight_decay (float, optional): Weight decay (L2 regularization) coefficient for the optimizer.
        device (torch.device): The device to use for the model.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The confusion matrix for the training set.
            - torch.Tensor or None: The confusion matrix for the test set (if `use_test_set_for_matrix` is True).
            - list[float]: The training losses for each epoch.
            - list[float]: The validation losses for each epoch.
            - list[float]: The training accuracies for each epoch.
            - list[float]: The validation accuracies for each epoch.
    """
    print(f"Initialize the linear classifier.")
    classifier = nn.Linear(in_features=model.embed_dim, out_features=args.num_classes).to(device)
    # Train the linear classifier on the training set
    epochs = 200 # adapt training time for classifier based on layer

    lr = 0.01

    # Set up the optimizer and loss function
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=3)
    criterion = CustomCrossEntropyLoss(alpha).to(device)

    losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_weights = classifier.state_dict() # Track best weights
    best_loss = float('inf')
    best_lc_epoch = 0 # Track epoch with best loss
    patience = 15  # number of epochs to wait without improvement
    epochs_since_improvement = 0

    for e in range(epochs):
        print(f"Epoch {e}")
        enumerated_trainloader = enumerate(trainloader)
        loop = tqdm(enumerated_trainloader, total=len(trainloader), desc=f"Training Linear Classifier on {block}")
        epoch_loss = 0
        train_correct = 0
        train_pred_correct = 0
        train_total = 0
        classifier.train()
        for i, data in loop:
            images, labels, pred_labels = data
            images = images.to(device)
            labels = labels.to(device)
            pred_labels = pred_labels.to(device)

            # Forward pass
            logits = classifier(images)
            loss = criterion(logits, labels, pred_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Compute training accuracy
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_pred_correct += (preds == pred_labels).sum().item()
            train_total += labels.size(0)

        # Calculate training accuracies
        true_train_accuracy = train_correct / train_total
        pred_train_accuracy = train_pred_correct / train_total
        train_accuracy = criterion.lam * true_train_accuracy + (1 - criterion.lam) * pred_train_accuracy # Training accuracy
        train_accuracies.append(train_accuracy)

        # keep track of eval loss
        eval_loss = 0
        true_correct = 0  # Track correct predictions
        pred_correct = 0 # Track correct predictions for pred_labels
        total = 0    # Track total samples

        loop = tqdm(enumerate(eval_trainloader), total=len(eval_trainloader), desc=f"Evaluating Linear Classifier on {block} with training set.")

        # Validation Loop
        classifier.eval()
        for i, data in loop:
            images, labels, pred_labels = data
            images = images.to(device)
            labels = labels.to(device)
            pred_labels = pred_labels.to(device)

            # Forward pass
            logits = classifier(images)
            loss = criterion(logits, labels, pred_labels)

            eval_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Compute validation accuracies
            preds = torch.argmax(logits, dim=1)
            true_correct += (preds == labels).sum().item()
            pred_correct += (preds == pred_labels).sum().item()

            total += labels.size(0)

        avg_loss = epoch_loss / len(trainloader)
        avg_val_loss = eval_loss / len(eval_trainloader)

        # Calculate accuracies
        true_accuracy = true_correct / total
        pred_accuracy = pred_correct / total


        # Weighted accuracy based on lambda
        val_accuracy = criterion.lam * true_accuracy + (1 - criterion.lam) * pred_accuracy

        val_accuracies.append(val_accuracy)
        losses.append(avg_loss)
        val_losses.append(avg_val_loss)


        scheduler.step(avg_val_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lc_epoch = e
            best_weights = classifier.state_dict()
            epochs_since_improvement = 0  # reset counter
        else:
            epochs_since_improvement += 1

        # Check for early stopping (comment if early stopping should be disabled)
        if epochs_since_improvement >= patience:
            break

    print(f"Best lc training epoch was {best_lc_epoch}. Loading best weights.")
    classifier.load_state_dict(best_weights)

    print("Trained Linear Classifier")

    C_train = None

    # Compute confusion matrix for the training set
    classifier.eval()
    with torch.no_grad():
        loop = tqdm(enumerate(eval_trainloader), total=len(eval_trainloader), desc=f"Computing conf matrix for {block} with training set.")
        for i, data in loop:
            images, labels, pred_labels = data
            images = images.to(device)
            labels = labels.to(device)
            logits = classifier(images)

            if C_train is None:
                C_train = torch.zeros(args.num_classes, args.num_classes)

            for j in range(logits.shape[0]):
                C_train[labels[j], torch.argmax(logits[j])] += 1

    # Compute confusion matrix for the test set if requested
    C_test = None
    if use_test_set_for_matrix:
        with torch.no_grad():
            loop = tqdm(enumerate(testloader), total=len(testloader), desc=f"Building confusion matrix for {block} with test set.")
            for i, data in loop:
                images, labels, pred_labels = data
                images = images.to(device)
                labels = labels.to(device)
                logits = classifier(images)
                if C_test is None:
                    C_test = torch.zeros(args.num_classes, args.num_classes)
                for j in range(logits.shape[0]):
                    C_test[labels[j], torch.argmax(logits[j])] += 1

    # normalize the confusion matrices
    C_train = C_train / C_train.sum(1, keepdim=True)
    if use_test_set_for_matrix:
        C_test = C_test / C_test.sum(1, keepdim=True)

    return C_train, C_test, losses, val_losses, train_accuracies, val_accuracies


def extract_features(model, dataloader, device, blocks):
    """
    Extract features from the model for the specified blocks using the provided dataloader.

    Args:
        model (torch.nn.Module): The model from which to extract features.
        dataloader (DataLoader): DataLoader for the training dataset.
        device (torch.device): Device to perform computations.
        blocks (list[int]): List of block indices to extract features from.

    Returns:
        A dictionary containing the features for each block, the true labels of the samples, and the models predictions.
    """
    activation_dict, modules = register_hooks(model, only=[f"blocks.{block}" for block in blocks])
    features = { block: torch.empty(0, model.embed_dim) for block in blocks }
    labels = torch.empty(0)
    predictions = torch.empty(0)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, label = data
            images = images.to(device)
            preds = model(images)
            predicted_class = preds.argmax(1)
            labels = torch.cat([labels, label])
            predictions = torch.cat([predictions, predicted_class.to('cpu')])
            for block in blocks:
                features[block] = torch.cat([features[block], torch.flatten(activation_dict[f"blocks.{block}"][:, 0].to('cpu'), 1)]) # extect class token only

    return features, labels, predictions


def extract_and_save_features(model, trainloader, testloader, eval_trainloader, device, layers, output_dir, rundir, alpha, checkpoint_name):
    """
    Extract features from the model for the training, test, and validation datasets and save them to disk.

    Args:
        model (torch.nn.Module): The model from which to extract features.
        trainloader (DataLoader): DataLoader for the training dataset.
        testloader (DataLoader): DataLoader for the test dataset.
        eval_trainloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations.
        activation_dict (dict): Dictionary to store activations for each layer.
        layers (list): List of layer names to extract features from.
        output_dir (str): Base output directory.
        rundir (str): Specific run directory appended to the output_dir.
        alpha (float): Alpha value for the custom loss function.
        checkpoint_name (str): Name of the checkpoint (used for file naming).

    Returns:
        None: Saves extracted features, labels, and predictions to disk.
    """
    train_features, train_labels, train_preds = extract_features(model, trainloader, device, layers)
    # Save extracted features, labels, and predictions to disk temporarily
    for layer, features in train_features.items():
        torch.save(features,
                   f"{output_dir}/alpha_{alpha}/{checkpoint_name}_train_features_{layer}.pt")
    train_features.clear()
    torch.save(train_labels,
               f"{output_dir}/alpha_{alpha}/{checkpoint_name}_train_labels.pt")
    torch.save(train_preds,
               f"{output_dir}/alpha_{alpha}/{checkpoint_name}_train_preds.pt")

    test_features, test_labels, test_preds = extract_features(model, testloader, device, layers)
    # Save extracted features, labels, and predictions to disk temporarily
    for layer, features in test_features.items():
        torch.save(features,
                   f"{output_dir}/alpha_{alpha}/{checkpoint_name}_test_features_{layer}.pt")
    test_features.clear()
    torch.save(test_labels,
               f"{output_dir}/alpha_{alpha}/{checkpoint_name}_test_labels.pt")
    torch.save(test_preds,
               f"{output_dir}/alpha_{alpha}/{checkpoint_name}_test_preds.pt")

    val_features, val_labels, val_preds = extract_features(model, eval_trainloader, device, layers)
    # Save extracted features, labels, and predictions to disk temporarily
    for layer, features in val_features.items():
        torch.save(features,
                   f"{output_dir}/alpha_{alpha}/{checkpoint_name}_val_features_{layer}.pt")
    val_features.clear()
    torch.save(val_labels,
               f"{output_dir}/alpha_{alpha}/{checkpoint_name}_val_labels.pt")
    torch.save(val_preds,
               f"{output_dir}/alpha_{alpha}/{checkpoint_name}_val_preds.pt")


def plot_metrics(losses, val_losses, train_accuracy, val_accuracy, checkpoint_name, layer, iteration, output_dir, rundir, alpha):
    """
    Plot and save loss and accuracy curves.

    Args:
        losses (list): List of training loss values (one per epoch).
        val_losses (list): List of validation loss values (one per epoch).
        train_accuracy (list): List of training accuracies (one per epoch).
        val_accuracy (list): List of validation accuracies (one per epoch).
        checkpoint_name (str): Name of the checkpoint (used for file naming).
        layer (str): The layer being evaluated.
        iteration (int): Iteration index for the current run.
        output_dir (str): Base output directory.
        rundir (str): Specific run directory appended to the output_dir.
    """
    # Create a figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Subplot 1: Plot training and validation loss curves ---
    axes[0].plot(losses, label="Training Loss", color="blue")
    axes[0].plot(val_losses, label="Validation Loss", color="orange")
    axes[0].set_title(f"Loss Curves - {checkpoint_name} {layer} iteration {iteration}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # --- Subplot 2: Plot training and validation accuracy curves ---
    axes[1].plot(train_accuracy, label="Training Accuracy", color="green")
    axes[1].plot(val_accuracy, label="Validation Accuracy", color="red")
    axes[1].set_title(f"Accuracy Curves - {checkpoint_name} {layer} iteration {iteration}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout, save the figure, and display it
    plt.tight_layout()
    fig.savefig(f"{output_dir}/alpha_{alpha}/train_curves/{checkpoint_name}_iteration_{iteration}_layer_{layer}_metrics.png", dpi=300)


def load_activation_data(output_dir, rundir, alpha, checkpoint_name, layer, batch_size):
    """
    Load activation data from disk and return the extracted features, labels, and predictions in dataloaders

    Args:
        output_dir (str): Base output directory.
        rundir (str): Specific run directory appended to the output_dir.
        alpha (float): Alpha value for the custom loss function.
        checkpoint_name (str): Name of the checkpoint (used for file naming).
        layer (str): layer name to extract features from.

    Returns:
        tuple: (act_trainloader, act_testloader, act_valloader)
            - act_trainloader (DataLoader): DataLoader for the training dataset.
            - act_testloader (DataLoader): DataLoader for the test dataset.
            - act_valloader (DataLoader): DataLoader for the validation dataset.
    """
    train_features = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_train_features_{layer}.pt")
    train_labels = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_train_labels.pt")
    train_preds = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_train_preds.pt")

    test_features = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_test_features_{layer}.pt")
    test_labels = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_test_labels.pt")
    test_preds = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_test_preds.pt")

    val_features = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_val_features_{layer}.pt")
    val_labels = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_val_labels.pt")
    val_preds = torch.load(f"{output_dir}/alpha_{alpha}/{checkpoint_name}_val_preds.pt")

    act_trainset = ActivationDataset(train_features, train_labels, train_preds)
    act_testset = ActivationDataset(test_features, test_labels, test_preds)
    act_valset = ActivationDataset(val_features, val_labels, val_preds)

    act_trainloader = torch.utils.data.DataLoader(act_trainset, batch_size=batch_size, shuffle=False)
    act_testloader = torch.utils.data.DataLoader(act_testset, batch_size=batch_size, shuffle=False)
    act_valloader = torch.utils.data.DataLoader(act_valset, batch_size=batch_size, shuffle=False)

    return act_trainloader, act_testloader, act_valloader


def main():
    # Initialization 
    set_seeders()
    GENERATOR = torch.Generator()
    GENERATOR.manual_seed(42)

    parser = argparse.ArgumentParser(description="Generate confusion matrices for arbitrary model using a linear classifier.")
    parser.add_argument("-m", "--model", help="Model to be used (efficient-vit).", type=str)
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory to model checkpoints. If this is set matrices are generated for all checkpoints in the dir.')
    parser.add_argument('--print_modules_only', action='store_true', help='Print all modules in the model and exit')
    parser.add_argument('--output_dir', type=str, help='Directory to save confusion matrices to.', default='.')
    parser.add_argument('--epoch_step', type=int, help='Number of epochs between confusion matrices', default=1)
    parser.add_argument('--start_index', type=int, help='The index for the checkpoints discovered to start the matrix generation for.', default=0)
    parser.add_argument('--end_index', type=int, help='The index for the checkpoints discovered to end the matrix generation for.', default=None)
    parser.add_argument('--use_n_blocks', type=int, default=None, help='Number of blocks after which to cut of the Transformer.')
    parser.add_argument('--label_type', type=str, default='true', help='Type of label to use for the confusion matrix. Can be "true" or "pred".')
    parser.add_argument('--alpha', type=float, help='The parameter determining the cross entropy mix between true and predicted labels.', default=1.0)
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset (default: 100 for CIFAR-100)')
    parser.add_argument('--depth', type=int, default=12, help='Depth parameter for VisionTransformer')
    parser.add_argument('--dataset', type=str, default='cifar', help='Dataset to use (cifar, tinyimagenet).')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size for training')
    parser.add_argument('--weight_decay', type=float, nargs='?', const=1e-4, default=0.0, help='Weight decay (L2 regularization) coefficient for the optimizer.')
    args = parser.parse_args()
    pprint(args)
    #
    #   Constants
    #
    n_iterations = 1
    #
    #   Datasetes
    #
    if args.dataset in ['cifar', 'cifar100', 'CIFAR', 'CIFAR100']:
        base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=3), # as done by the EfficientViT repo
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        # Datasets for Evaluation (no data augmentation)
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=base_transform)
        train_trainset = torch.utils.data.Subset(trainset, range(40_000))
        eval_trainset = torch.utils.data.Subset(trainset, range(40_000, 50_000))

        trainloader = torch.utils.data.DataLoader(train_trainset, batch_size=args.batch_size, shuffle=True)
        eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=False)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=base_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    elif args.dataset in ['tinyimgnet', 'tinyimagenet']:
        base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=3),
        transforms.Normalize(mean=tensor([0.485, 0.456, 0.406]), std=tensor([0.229, 0.224, 0.225]))
        ])

        trainset = torchvision.datasets.ImageFolder('./tiny-imagenet-200/train', transform=base_transform)
        indices = torch.randperm(len(trainset), generator=GENERATOR).tolist()
        trainset = torch.utils.data.Subset(trainset, indices)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        
        train_trainset = torch.utils.data.Subset(train_loader.dataset, range(int(0.8 * len(train_loader.dataset))))
        eval_trainset = torch.utils.data.Subset(train_loader.dataset, range(int(0.8 * len(train_loader.dataset)), len(train_loader.dataset)))

        trainloader = torch.utils.data.DataLoader(train_trainset, batch_size=args.batch_size, shuffle=False)
        eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=False)
        testset = torchvision.datasets.ImageFolder('./tiny-imagenet-200/val', transform=base_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    rundir = ''
    checkpoint_paths = []
    if args.checkpoint_path:
        checkpoint_paths = [args.checkpoint_path]
        checkpoint_path = Path(args.checkpoint_path)
    elif args.checkpoint_dir:
        checkpoint_paths = natsorted(glob(os.path.join(args.checkpoint_dir, '*.pth')))
        checkpoint_dir = Path(args.checkpoint_dir)
        rundir = checkpoint_dir.name

    print(f"Found {len(checkpoint_paths)} checkpoints.")
    print(f"Start index {args.start_index}, End index {args.end_index}, Step {args.epoch_step}")
    checkpoint_paths = checkpoint_paths[args.start_index:args.end_index:args.epoch_step]
    print(f"Using {len(checkpoint_paths)} checkpoints:")
    pprint(checkpoint_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(f"{args.output_dir}/alpha_{args.alpha}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/alpha_{args.alpha}/train_curves", exist_ok=True)
    #
    #   Generate Confusion Matrices
    #
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        model = initialize_model(args)
        model.eval()
        # register hooks to find the layers
        model.to(device)
        if args.print_modules_only:
            # Print model summary
            activation_dict, modules = register_hooks(model)
            summary(model, input_size=(args.batch_size, 3, args.image_size, args.image_size))
            print("Modules:")
            pprint(modules)
            return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])

        checkpoint_name = checkpoint_path.stem

        # Call extract and save features
        blocks = []
        match args.depth:
            case 12:
                blocks = [0, 2, 5, 8, 11]
            case 8:
                blocks = [0, 1, 3, 5, 7]
            case 4:
                blocks = [0, 1, 2, 3]

        extract_and_save_features(model, trainloader, testloader, eval_trainloader, device, blocks, args.output_dir, rundir, args.alpha, checkpoint_name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for block in blocks:
            for iteration in range(n_iterations):
                print(f"\n---------------------- Iteration {iteration} ----------------------")

                act_trainloader, act_testloader, act_evalloader = load_activation_data(args.output_dir, rundir, args.alpha, checkpoint_name, block, args.batch_size)

                print(f"\nGenerating confusion matrix for block {block} in iteration {iteration} (checkpoint {checkpoint_name}).")
                # Perform the confusion matrix generation
                C_train, C_test, losses, val_losses, train_accuracy, val_accuracy = create_confusion_matrix(args, model, block, act_trainloader, act_testloader, act_evalloader, True, args.alpha, args.weight_decay, device=device)
                np.save(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_iteration_{iteration}_block_{block}_use_testset.npy", C_test)
                np.save(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_iteration_{iteration}_block_{block}_use_trainset.npy", C_train)

                # Call plot function
                plot_metrics(losses, val_losses, train_accuracy, val_accuracy, checkpoint_name, block, iteration, args.output_dir, rundir, args.alpha)

            # clean disk
            os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_train_features_{block}.pt")
            os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_test_features_{block}.pt")
            os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_val_features_{block}.pt")

        os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_train_preds.pt")
        os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_test_preds.pt")
        os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_val_preds.pt")
        os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_train_labels.pt")
        os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_test_labels.pt")
        os.remove(f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_val_labels.pt")


if __name__ == '__main__':
    main()
