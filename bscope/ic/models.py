from torchvision import models, datasets
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.mobilenet import MobileNet_V3_Small_Weights
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def get_alexnet(imagenet_path='/mnt/data/imagenet', batch_size=64):
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights)

    transforms = weights.transforms()
    val_dataset = datasets.ImageNet(root=imagenet_path,
                                    split='val',
                                    transform=transforms)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    return model, val_dataset, val_dataloader


def get_mobilenet(imagenet_path='/mnt/data/imagenet', batch_size=64):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)

    transforms = weights.transforms()
    val_dataset = datasets.ImageNet(root=imagenet_path,
                                    split='val',
                                    transform=transforms)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    return model, val_dataset, val_dataloader


def get_resnet50(imagenet_path='/mnt/data/imagenet', batch_size=64):
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)

    transforms = weights.transforms()
    val_dataset = datasets.ImageNet(root=imagenet_path,
                                    split='val',
                                    transform=transforms)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    return model, val_dataset, val_dataloader


def get_resnet18(imagenet_path='/mnt/data/imagenet', batch_size=64):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    transforms = weights.transforms()
    val_dataset = datasets.ImageNet(root=imagenet_path,
                                    split='val',
                                    transform=transforms)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    return model, val_dataset, val_dataloader


def get_rgb_dataset(imagenet_path='/mnt/data/imagenet', batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = datasets.ImageNet(root=imagenet_path,
                                    split='val',
                                    transform=transform)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    return val_dataset
