import argparse
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    build_dataset,
    initialize_model,
    initialize_optimizer,
    initialize_scheduler,
    set_seeders,
)


def evaluate(model, val_loader, criterion, device):
    """Run model on val_loader, return (accuracy, average loss)."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_sum += loss.item()
            n_batches += 1

    model.train()
    return correct / total, loss_sum / n_batches


def parse_args():
    parser = argparse.ArgumentParser(description='Training ResNet-50 on CIFAR-100')
    parser.add_argument('--model', type=str, default='resnet50', help='Model to use for training')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=71, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--outdir', type=str, default='resnet_checkpoints',
                        help='Output directory for saving model checkpoints')
    parser.add_argument('--runname', type=str, default='resnet_training', help='Name of the run')
    parser.add_argument('--small_conv1', action='store_true', help='Use 3x3 conv1 instead of 7x7 as first layer')
    parser.add_argument('--gradclip', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--scheduler', type=str, default='plateau', help='Learning rate scheduler to use')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use (adam)')
    parser.add_argument('--weight_decay', type=float, default=0.4e-04, help='Weight decay for optimizer')
    parser.add_argument('--autoaugment', action='store_true', help='Use AutoAugment for data augmentation')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')
    return parser.parse_args()


def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    writer = SummaryWriter(log_dir=str(os.path.join(args.outdir, args.runname)))

    model.to(device)
    model.train()

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

        accuracy, val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Validation accuracy: {accuracy}, Validation loss: {val_loss}')
        writer.add_scalar('validation/accuracy', accuracy, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)

        writer.add_scalar('training/loss', running_loss / len(train_loader), epoch)
        if args.scheduler != 'None':
            writer.add_scalar('training/lr', scheduler.get_last_lr()[0], epoch)

        # Save the model every 5 epochs (1, 6, 11, ... in 1-indexed terms).
        if args.outdir and (epoch % 5 == 0):
            torch.save(model.state_dict(), str(os.path.join(args.outdir, f'{args.model}_cifar100_epoch_{epoch + 1}.pth')))

    writer.add_hparams(hparam_dict=dict(vars(args)), metric_dict={'hparam/accuracy': best_accuracy}, run_name='.')
    print('Finished Training')
    writer.close()


def main():
    set_seeders()
    args = parse_args()
    print(args)
    print('Using device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

    train_loader, val_loader = build_dataset(args)
    model = initialize_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer)

    train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device=device)


if __name__ == '__main__':
    main()
