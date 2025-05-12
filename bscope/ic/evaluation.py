import tqdm
import torchvision
import torch.nn as nn
import torch
import numpy as np


def calculate_class_accuracy(model,
                             val_loader,
                             num_classes=1000,
                             device='cuda:1',
                             target_classes=None,
                             target_topk=5,
                             nontarget_topk=1):

    if isinstance(target_classes, int):
        target_classes = [target_classes]
    elif target_classes is None:
        target_classes = []

    model.eval()
    model = model.to(device)

    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Get top max(target_topk, nontarget_topk) predictions once for efficiency
            max_k = max(target_topk, nontarget_topk)
            _, pred_all = outputs.topk(max_k, dim=1, largest=True, sorted=True)

            for i in range(targets.size(0)):
                label = targets[i].item()
                total_per_class[label] += 1

                # Choose dynamic top-k
                topk = target_topk if label in target_classes else nontarget_topk
                pred = pred_all[i][:topk]

                if label in pred:
                    correct_per_class[label] += 1

    class_accuracy = torch.zeros(num_classes, device=device)
    for i in range(num_classes):
        if total_per_class[i] > 0:
            class_accuracy[i] = (correct_per_class[i] /
                                 total_per_class[i]) * 100

    return class_accuracy.detach().cpu().numpy(), total_per_class.int()
