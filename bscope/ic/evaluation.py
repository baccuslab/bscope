import tqdm
import torchvision
import torch.nn as nn
import torch
import numpy as np


def calculate_accuracy(model, val_loader, device='cuda'):
    """
    Evaluate a model on the validation dataset and return top-1 and top-5 accuracy.

    Args:
        model: PyTorch model to evaluate
        val_loader: DataLoader for the validation dataset
        device: Device to run evaluation on ('cuda' or 'cpu')

    Returns:
        top1_acc: Top-1 accuracy as a percentage
        top5_acc: Top-5 accuracy as a percentage
    """
    model.eval()
    model = model.to(device)

    correct_1 = 0
    correct_5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct_1 += (predicted == targets).sum().item()

            # Top-5 accuracy
            _, top5_predicted = outputs.topk(5, 1)
            for i in range(targets.size(0)):
                if targets[i] in top5_predicted[i]:
                    correct_5 += 1

            total += targets.size(0)

    top1_acc = 100 * correct_1 / total
    top5_acc = 100 * correct_5 / total

    return top1_acc, top5_acc


def calculate_class_accuracy(model,
                             val_loader,
                             num_classes=1000,
                             device='cuda'):

    model.eval()
    model = model.to(device)

    # Initialize counters for per-class accuracy
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(targets.size(0)):
                label = targets[i].item()
                total_per_class[label] += 1
                if predicted[i].item() == label:
                    correct_per_class[label] += 1

    # Compute per-class accuracy (handle division by zero)
    class_accuracy = torch.zeros(num_classes, device=device)
    for i in range(num_classes):
        if total_per_class[i] > 0:
            class_accuracy[i] = (correct_per_class[i] /
                                 total_per_class[i]) * 100

    return class_accuracy, total_per_class.int()
