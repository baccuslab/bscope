import tqdm
import torchvision
import torch.nn as nn
import torch
import numpy as np


def calculate_class_accuracy(model,
                             val_loader,
                             num_classes=1000,
                             device='cuda',
                             topk=1):

    model.eval()
    model = model.to(device)

    # Initialize counters for per-class accuracy
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Get top-k predictions
            _, pred = outputs.topk(topk, dim=1, largest=True, sorted=True)

            for i in range(targets.size(0)):
                label = targets[i].item()
                total_per_class[label] += 1
                if label in pred[i]:
                    correct_per_class[label] += 1

    # Compute per-class accuracy (handle division by zero)
    class_accuracy = torch.zeros(num_classes, device=device)
    for i in range(num_classes):
        if total_per_class[i] > 0:
            class_accuracy[i] = (correct_per_class[i] /
                                 total_per_class[i]) * 100

    return class_accuracy.detach().cpu().numpy(), total_per_class.int()
