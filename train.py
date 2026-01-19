import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import os
from utils import initialize_optimizer, initialize_scheduler, initialize_model, set_seeders,  build_dataset


def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()  # Set the model back to training mode
    return correct / total, loss


def none_or_int(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")


def parse_args():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Training ResNet-50 on CIFAR-100')
    parser.add_argument('--model', type=str, default='resnet50', help='Model to use for training')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=71, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--outdir', type=str, default='resnet_checkpoints',
                        help='Output directory for saving model checkpoints')
    parser.add_argument('--runname', type=str, default='resnet_training', help='Name of the run')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained weights')
    parser.add_argument('--small_conv1', action='store_true', help='Use 3x3 conv1 instead of 7x7 as first layer')
    parser.add_argument('--gradclip', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--scheduler', type=str, default='plateau', help='Learning rate scheduler to use')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use (adam)')
    parser.add_argument('--weight_decay', type=float, default=0.4e-04, help='Weight decay for optimizer')
    parser.add_argument('--autoaugment', action='store_true', help='Use AutoAugment for data augmentation')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for dataset')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Directory for dataset')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')
    parser.add_argument('--image_size', type=int, default=32, help='Size of the input images')
    return parser.parse_args()


def train(args, model, train_loader, test_loader, criterion, optimizer, scheduler, device):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=str(os.path.join(args.outdir, args.runname)))

    # Training loop
    model.to(device)
    model.train()

    iters = len(train_loader)
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch {epoch + 1}/{args.epochs}')
        for i, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if args.gradclip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.gradclip)

            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (i + 1))

        # Test the model
        accuracy, loss = test_model(model, test_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Test accuracy: {accuracy}, Test loss: {loss}')
        writer.add_scalar('testing/accuracy', accuracy, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if args.scheduler == 'plateau':
            scheduler.step(loss)

        # Log the loss to TensorBoard
        writer.add_scalar('training/loss', running_loss / len(train_loader), epoch)
        if args.scheduler != 'None':
            writer.add_scalar('training/lr', scheduler.get_last_lr()[0], epoch)

        # Save the model
        if args.outdir and (epoch % 5 == 0):
            torch.save(model.state_dict(), str(os.path.join(args.outdir, f'{args.model}_cifar100_epoch_{epoch + 1}.pth')))

    writer.add_hparams(hparam_dict=dict(vars(args)), metric_dict={'hparam/accuracy': best_accuracy}, run_name='.')
    print('Finished Training')

    # Close TensorBoard writer
    writer.close()


def main():
    set_seeders()
    args = parse_args()

    print(args)

    print('Using device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

    train_loader, test_loader = build_dataset(args)

    # Initialize model
    model = initialize_model(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = initialize_optimizer(args, model)
    # Learning rate scheduler
    scheduler = initialize_scheduler(args, optimizer)

    train(args, model, train_loader, test_loader, criterion, optimizer, scheduler, device=device)


if __name__ == '__main__':
    main()
