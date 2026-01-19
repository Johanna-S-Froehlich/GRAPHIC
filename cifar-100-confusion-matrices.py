import argparse
import os
from glob import glob
from pprint import pprint
from pathlib import Path

import torch
from torchinfo import summary
import torchvision
import torchvision.transforms as transforms
import numpy as np
from natsort import natsorted
import re
import matplotlib.pyplot as plt

from utils import initialize_model, set_seeders
from models.LinearClassifier import LinearClassifier
from models.CustomLossFunction import CustomCrossEntropyLoss
from feature_extraction.hooks import register_hooks
from feature_extraction.hooks import capture_output_dims


#
#   Helper Methods
#
def create_confusion_matrix(model, ouput_dim, trainloader, testloader, eval_trainloader, alpha, layer: str, use_test_set_for_matrix: bool, weight_decay: float = 0.0, device=torch.device("cpu")):
    """
    Creates a confusion matrix for a given model and layer using a linear classifier.

    This function:
    - Trains a linear classifier on top of a specific layer of the model.
    - Computes confusion matrices for both the training and test sets.
    - Monitors the training process using loss and accuracy metrics.

    Args:
        model (torch.nn.Module): The backbone model from which features are extracted.
        ouput_dim (int): The output dimension of the extracted features.
        trainloader (DataLoader): DataLoader for the training dataset.
        testloader (DataLoader): DataLoader for the validation/test dataset.
        eval_trainloader (DataLoader): DataLoader for evaluating the training set (to compute confusion matrix).
        alpha (float): Weighting factor for the custom loss function (used in `CustomCrossEntropyLoss`).
        layer (str): The name of the layer where features are extracted.
        use_test_set_for_matrix (bool): Whether to compute the confusion matrix on the test dataset.
        weight_decay (float, optional): Weight decay (L2 regularization) coefficient for the optimizer.
        device (torch.device, optional): Device to perform computations (default is CPU).

    Returns:
        tuple: (C_train, C_test, losses, val_losses, train_accuracies, val_accuracies)
            - C_train (Tensor): Normalized confusion matrix for the training set.
            - C_test (Tensor or None): Normalized confusion matrix for the test set (if applicable).
            - losses (list): Training loss values for each epoch.
            - val_losses (list): Validation loss values for each epoch.
            - train_accuracies (list): Training accuracy values for each epoch.
            - val_accuracies (list): Validation accuracy values for each epoch.
    """

    print(f"Initialize the linear classifier.")

    # Step 1: Initialize the linear classifier on top of the selected model layer
    classifier = LinearClassifier(model, layer, ouput_dim, 100).to(device)

    # Step 2: Set learning rate based on the layer depth
    if "1" in layer:
        lr = 0.0001
    elif "2" in layer:
        lr = 0.0002
    elif "3" in layer:
        lr = 0.0006
    else:
        lr = 0.001

    # Step 3: Initialize optimizer (Adam) and learning rate scheduler
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=3)

    # Step 4: Set the custom cross-entropy loss function
    criterion = CustomCrossEntropyLoss(alpha)

    # Step 5: Initialize tracking variables
    losses = []  # Store training losses
    val_losses = []  # Store validation losses
    train_accuracies = []  # Store training accuracies
    val_accuracies = []  # Store validation accuracies

    best_weights = classifier.state_dict()  # Save the best model weights
    best_loss = float('inf')  # Initialize best loss to infinity
    best_lc_epoch = 0  # Track epoch with best loss
    patience = 5  # number of epochs to wait without improvement
    epochs_since_improvement = 0


    epochs = 110  # Number of training epochs
    model.eval()  # Ensure the model is in evaluation mode

    # Step 6: Training loop
    for e in range(epochs):
        epoch_loss = 0
        train_correct = 0
        train_pred_correct = 0
        train_total = 0
        classifier.train()
        classifier.features.eval()
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions from the linear classifier
            logits = classifier(images)

            # Get predictions from the original model (frozen backbone)
            with torch.no_grad():
                model_logits = model(images)  # Forward pass through the original model
                pred_labels = torch.argmax(model_logits, dim=1)  # Extract predicted labels

            # Compute the custom loss using the true and predicted labels
            loss = criterion(logits, labels, pred_labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute training accuracy
            preds = torch.argmax(logits, dim=1)  # Get predicted class from logits

            # Accuracy with true labels
            train_correct += (preds == labels).sum().item()
            # Accuracy with predicted labels (from the frozen model)
            train_pred_correct += (preds == pred_labels).sum().item()

            train_total += labels.size(0)

        # Step 7: Track epoch-level statistics
        avg_loss = epoch_loss / len(trainloader)  # Average training loss

        # Calculate training accuracies
        true_train_accuracy = train_correct / train_total
        pred_train_accuracy = train_pred_correct / train_total
        train_accuracy = criterion.lam * true_train_accuracy + (
                    1 - criterion.lam) * pred_train_accuracy  # Training accuracy

        losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        # Step 8: Validation loop (on the test set)
        classifier.eval()  # Switch the classifier to evaluation mode
        epoch_loss_test = 0
        true_correct = 0
        pred_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                logits = classifier(images)
                preds = torch.argmax(logits, dim=1)  # Get predicted class from logits

                # Get predictions from the original model (frozen backbone)
                model_test_logits = model(images)  # Forward pass through the original model
                pred_labels = torch.argmax(model_test_logits, dim=1)  # Extract predicted labels

                test_loss = criterion(logits, labels, pred_labels)

                epoch_loss_test += test_loss.item()

                # Accuracy with true labels
                true_correct += (preds == labels).sum().item()

                # Accuracy with predicted labels (from the frozen model)
                pred_correct += (preds == pred_labels).sum().item()
                total += labels.size(0)

        avg_val_loss = epoch_loss_test / len(testloader)  # Average validation loss
        # Calculate accuracies
        true_accuracy = true_correct / total
        pred_accuracy = pred_correct / total

        # Weighted accuracy based on lambda
        val_accuracy = criterion.lam * true_accuracy + (1 - criterion.lam) * pred_accuracy

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Adjust learning rate using the scheduler
        scheduler.step(avg_val_loss)

        # Save the best model based on lowest validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_lc_epoch = e
            best_weights = classifier.state_dict()
            epochs_since_improvement = 0  # reset counter
        else:
            epochs_since_improvement += 1


        print(
            f"Epoch {e + 1}/{epochs} | LR: {optimizer.param_groups[0]['lr']} | Train Loss: {avg_loss:.4f} | Test Loss: {avg_val_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train Acc True: {true_train_accuracy:.4f} | Train Acc Pred: {pred_train_accuracy:.4f} | Test Acc: {val_accuracy:.4f} | Test Acc True: {true_accuracy:.4f} | Test Acc Pred: {pred_accuracy:.4f}",
            flush=True)
        
        # Check for early stopping (comment if early stopping should be disabled)
        if epochs_since_improvement >= patience:
            break


    # Step 9: Load the best classifier weights
    print(f"Best lc training epoch was {best_lc_epoch}. Loading best weights.")
    classifier.load_state_dict(best_weights)
    print("Trained Linear Classifier")

    # Step 10: Compute confusion matrix for the training set
    C_train = None
    with torch.no_grad():
        for images, labels in eval_trainloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = classifier(images)

            if C_train is None:
                C_train = torch.zeros(100, 100)

            for j in range(logits.shape[0]):
                C_train[labels[j], torch.argmax(logits[j])] += 1

    # Step 11: Compute confusion matrix for the test set (if required)
    C_test = None
    if use_test_set_for_matrix:
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                logits = classifier(images)

                if C_test is None:
                    C_test = torch.zeros(100, 100)

                for j in range(logits.shape[0]):
                    C_test[labels[j], torch.argmax(logits[j])] += 1

    # Step 12: Normalize confusion matrices
    C_train = C_train / C_train.sum(1, keepdim=True)
    if use_test_set_for_matrix:
        C_test = C_test / C_test.sum(1, keepdim=True)

    # Step 13: Return results
    return C_train, C_test, losses, val_losses, train_accuracies, val_accuracies


def plot_metrics(losses, val_losses, train_accuracy, val_accuracy, checkpoint_name, layer, iteration, output_dir, alpha):
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
    fig.savefig(
        f"{output_dir}/alpha_{alpha}/train_curves/{checkpoint_name}_iteration_{iteration}_layer_{layer}_metrics.png",
        dpi=300)


def main():
    """
    Main function to generate confusion matrices for a model using a linear classifier.
    """
    # Initialization: set seeds for reproducibility
    set_seeders()

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices for arbitrary model using a linear classifier.")
    parser.add_argument("-m", "--model", help="Model to be used (ResNet).", type=str)
    parser.add_argument('--small_conv1', action='store_true', help='Use 3x3 conv1 instead of 7x7 as first layer')
    parser.add_argument("--only", help="Only include these layers in the activation data.", type=str, nargs='*')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory to model checkpoints. If this is set matrices are generated for all checkpoints in the dir.')
    parser.add_argument('--print_modules_only', action='store_true', help='Print all modules in the model and exit')
    parser.add_argument('--output_dir', type=str, help='Directory to save confusion matrices to.', default='.')
    parser.add_argument('--epoch_step', type=int, help='Number of epochs between confusion matrices', default=1)
    parser.add_argument('--start_index', type=int,
                        help='The index for the checkpoints discovered to start the matrix generation for.', default=0)
    parser.add_argument('--end_index', type=int,
                        help='The index for the checkpoints discovered to end the matrix generation for.', default=None)
    parser.add_argument('--alpha', type=float,
                        help='The parameter determining the cross entropy mix between true and predicted labels.',
                        default=1.0)
    parser.add_argument('--iterations', type=int, help='Number of iterations to run the classifier', default=1)
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset (default: 100 for CIFAR-100)')
    parser.add_argument('--weight_decay', type=float, nargs='?', const=1e-4, default=0.0, help='Weight decay (L2 regularization) coefficient for the optimizer.')


    args = parser.parse_args()
    pprint(args)

    # Constants: batch size
    batch_size = 2000

    # Define transformations for training and test sets (normalization, conversion to Tensor)
    transform_train = []
    transform_test = []

    # Add standard transformations (ToTensor and Normalize)
    transform_train.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    transform_test.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Compose the transformations
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    # Load CIFAR-100 dataset and split into training and evaluation subsets
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,transform=transform_train)
    train_trainset = torch.utils.data.Subset(trainset, range(40_000))
    eval_trainset = torch.utils.data.Subset(trainset, range(40_000, 50_000))

    # Create DataLoaders for training, evaluation, and testing
    trainloader = torch.utils.data.DataLoader(train_trainset, batch_size=batch_size, shuffle=True)
    eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=batch_size, shuffle=False)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Determine checkpoint paths based on provided arguments
    checkpoint_paths = []
    if args.checkpoint_dir:
        cp_dir = Path(args.checkpoint_dir)
        # Find all .pth files in the directory
        checkpoint_paths = natsorted(glob(str(cp_dir / '*.pth'), recursive=True))
    else:
        raise ValueError("You must provide a checkpoint_dir argument.")

    print(f"Found {len(checkpoint_paths)} checkpoints.")
    print(f"Start index {args.start_index}, End index {args.end_index}, Step {args.epoch_step}")

    epoch_to_ckpt = {}
    for path in checkpoint_paths:
        match = re.search(r'epoch[_-]?(\d+)', path)
        if match:
            epoch = int(match.group(1))
            epoch_to_ckpt[epoch] = path

    # Define which epochs you want: e.g., 1, 6, 11, ...
    selected_epochs = list(range(args.start_index, args.end_index, args.epoch_step))

    # Now filter checkpoint_paths accordingly
    checkpoint_paths = [epoch_to_ckpt[e] for e in selected_epochs if e in epoch_to_ckpt]

    print(f"Using {len(checkpoint_paths)} checkpoints:")
    pprint(checkpoint_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Main loop: generate confusion matrices for each checkpoint
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        # Initialize the model using a provided function
        model = initialize_model(args)
        model.eval()
        model.to(device)
        # Capture output dimensions from specific layers
        output_dims = capture_output_dims(model, only=args.only, device=device)
        # Register hooks to find the layers (for activation extraction)
        activation_dict, modules = register_hooks(model, only=args.only)
        layers = list(modules.keys())
        model.to(device)
        print(f"Layers to extract: {layers}")
        if args.print_modules_only:
            # If flag is set, print model summary and exit without further processing
            summary(model, input_size=(batch_size, 3, 32, 32))
            print("Modules in model:")
            pprint(modules.keys())
            print("Output dimensions:")
            pprint(output_dims)
            return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        checkpoint_name = checkpoint_path.stem

        # Create output directories for saving confusion matrices
        os.makedirs(f"{args.output_dir}/alpha_{args.alpha}", exist_ok=True)
        os.makedirs(f"{args.output_dir}/alpha_{args.alpha}/train_curves",
                    exist_ok=True)

        # Iterate over number of iterations (typically one) for matrix generation
        for iteration in range(args.iterations):
            print(f"\n---------------------- Iteration {iteration} ----------------------")
            for layer in layers:
                print(
                    f"\nGenerating confusion matrix for layer {layer} in iteration {iteration} (checkpoint {checkpoint_name}).")
                # Call the function to create confusion matrices and capture metrics
                C_train, C_test, losses, val_losses, train_accuracy, val_accuracy = create_confusion_matrix(model,
                                                                                                            output_dims[
                                                                                                                layer],
                                                                                                            trainloader,
                                                                                                            testloader,
                                                                                                            eval_trainloader,
                                                                                                            args.alpha,
                                                                                                            layer, True, args.weight_decay,
                                                                                                            device=device)
                # Save confusion matrices for test and train sets
                np.save(
                    f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_iteration_{iteration}_{layer}_use_testset.npy",
                    C_test)
                np.save(
                    f"{args.output_dir}/alpha_{args.alpha}/{checkpoint_name}_iteration_{iteration}_{layer}_use_trainset.npy",
                    C_train)

                # Call the plotting function to create and save the metric curves
                plot_metrics(losses, val_losses, train_accuracy, val_accuracy, checkpoint_name, layer, iteration,
                             args.output_dir, args.alpha)


if __name__ == '__main__':
    main()
