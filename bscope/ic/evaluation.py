# import tqdm
# import torchvision
# import torch.nn as nn
# import torch
# import numpy as np
# import tqdm
# import torchvision
# import torch.nn as nn
# import torch
# import numpy as np


# def calculate_accuracy(model, val_loader, device='cuda'):
#     """
#     Evaluate a model on the validation dataset and return top-1 and top-5 accuracy.

#     Args:
#         model: PyTorch model to evaluate
#         val_loader: DataLoader for the validation dataset
#         device: Device to run evaluation on ('cuda' or 'cpu')

#     Returns:
#         top1_acc: Top-1 accuracy as a percentage
#         top5_acc: Top-5 accuracy as a percentage
#     """
#     model.eval()
#     model = model.to(device)

#     correct_1 = 0
#     correct_5 = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, targets in tqdm.tqdm(val_loader):
#             inputs, targets = inputs.to(device), targets.to(device)

#             # Forward pass
#             outputs = model(inputs)

#             # Top-1 accuracy
#             _, predicted = outputs.max(1)
#             correct_1 += (predicted == targets).sum().item()

#             # Top-5 accuracy
#             _, top5_predicted = outputs.topk(5, 1)
#             for i in range(targets.size(0)):
#                 if targets[i] in top5_predicted[i]:
#                     correct_5 += 1

#             total += targets.size(0)

#     top1_acc = 100 * correct_1 / total
#     top5_acc = 100 * correct_5 / total

#     return top1_acc, top5_acc



# def calculate_class_accuracy(model,
#                              val_loader,
#                              num_classes=1000,
#                              device='cuda:1',
#                              target_classes=None,
#                              target_topk=5,
#                              nontarget_topk=1):

#     if isinstance(target_classes, int):
#         target_classes = [target_classes]
#     elif target_classes is None:
#         target_classes = []

#     model.eval()
#     model = model.to(device)

#     correct_per_class = torch.zeros(num_classes, device=device)
#     total_per_class = torch.zeros(num_classes, device=device)

#     with torch.no_grad():
#         for inputs, targets in tqdm.tqdm(val_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)

#             # Get top max(target_topk, nontarget_topk) predictions once for efficiency
#             max_k = max(target_topk, nontarget_topk)
#             _, pred_all = outputs.topk(max_k, dim=1, largest=True, sorted=True)

#             for i in range(targets.size(0)):
#                 label = targets[i].item()
#                 total_per_class[label] += 1

#                 # Choose dynamic top-k
#                 topk = target_topk if label in target_classes else nontarget_topk
#                 pred = pred_all[i][:topk]

#                 if label in pred:
#                     correct_per_class[label] += 1

#     class_accuracy = torch.zeros(num_classes, device=device)
#     for i in range(num_classes):
#         if total_per_class[i] > 0:
#             class_accuracy[i] = (correct_per_class[i] /
#                                  total_per_class[i]) * 100

#     return class_accuracy.detach().cpu().numpy(), total_per_class.int()


# def calculate_subsample_accuracy(model,
#                              val_loader,
#                              subclasses,
#                              topk=5,
#                              device='cuda:1'):


#     model.eval()
#     model = model.to(device)

#     correct_per_class = torch.zeros(len(subclasses), device=device)
#     n_subsample = len(val_loader.dataset) // len(subclasses)
#     print('N subsample ', n_subsample)

#     label_mapping = {label: idx for idx, label in enumerate(subclasses)}

#     with torch.no_grad():
#         for inputs, targets in tqdm.tqdm(val_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)

#             # Get top max(target_topk, nontarget_topk) predictions once for efficiency
#             _, pred_all = outputs.topk(topk, dim=1, largest=True, sorted=True)
        
#             for i in range(targets.size(0)):
#                 label = targets[i].item()
#                 pred = pred_all[i][:topk]

#                 if label in pred:
#                     correct_per_class[label_mapping[label]] += 1

#     class_accuracy = torch.zeros(len(subclasses), device=device)
#     num_classes = len(subclasses)
#     for i in range(num_classes):
#         class_accuracy[i] = (correct_per_class[i] /
#                                  n_subsample) * 100

#     return class_accuracy.detach().cpu().numpy()

import tqdm
import torchvision
import torch.nn as nn
import torch
import numpy as np
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


def calculate_subsample_accuracy(model,
                             val_loader,
                             subclasses,
                             device='cuda:1'):


    model.eval()
    model = model.to(device)

    correct_per_class_top_1 = torch.zeros(len(subclasses), device=device)
    correct_per_class_top_5 = torch.zeros(len(subclasses), device=device)

    n_subsample = len(val_loader.dataset) // len(subclasses)
    print('N subsample ', n_subsample)

    label_mapping = {label: idx for idx, label in enumerate(subclasses)}

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Get top max(target_topk, nontarget_topk) predictions once for efficiency
            _, pred_all = outputs.topk(5, dim=1, largest=True, sorted=True)

        
            for i in range(targets.size(0)):
                label = targets[i].item()

                topk = 1
                pred_1 = pred_all[i][:topk]

                topk = 5
                pred_5 = pred_all[i][:topk]

                if label in pred_1:
                    correct_per_class_top_1[label_mapping[label]] += 1
                if label in pred_5:
                    correct_per_class_top_5[label_mapping[label]] += 1

    top1_class_accuracy = torch.zeros(len(subclasses), device=device)
    top5_class_accuracy = torch.zeros(len(subclasses), device=device)

    num_classes = len(subclasses)
    for i in range(num_classes):
        top1_class_accuracy[i] = (correct_per_class_top_1[i] / n_subsample) * 100
        top5_class_accuracy[i] = (correct_per_class_top_5[i] / n_subsample) * 100


    return top1_class_accuracy.cpu().numpy(), top5_class_accuracy.cpu().numpy()