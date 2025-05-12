from torchvision import models, datasets
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.mobilenet import MobileNet_V3_Small_Weights
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights, ResNet101_Weights
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from IPython import embed

def get_model(which_model, return_layers=False, imagenet_path='/mnt/data/imagenet', device='cuda',**kwargs):

    if which_model == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)

    elif which_model == 'resnet101':
        weights = ResNet101_Weights.IMAGENET1K_V1
        model = models.resnet101(weights=weights)

    elif which_model == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)

    elif which_model == 'alexnet':
        weights = AlexNet_Weights.IMAGENET1K_V1
        model = models.alexnet(weights=weights)

    elif which_model == 'mobilenet':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    
    model.to(device)

    transforms = weights.transforms()
    val_dataset = datasets.ImageNet(root=imagenet_path,
                                    split='val',
                                    transform=transforms)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=kwargs.get('batch_size', 128),
                                num_workers=kwargs.get('num_workers', 32),
                                pin_memory=kwargs.get('pin_memory', True),
                                shuffle=(kwargs.get('shuffle', False)))
    if return_layers is False:
        return model, val_dataset, val_dataloader
    else:
        if 'resnet' in which_model:
            model_layers = []
            for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
                for block in layer:
                    model_layers.append(block.bn3)

            print('Found {} layers'.format(len(model_layers)))
            
            return model, val_dataset, val_dataloader, model_layers
