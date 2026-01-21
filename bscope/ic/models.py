from torchvision import models, datasets
import timm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.mobilenet import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights, ResNet101_Weights
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from IPython import embed
from .custom_dataset import CustomImageNetDataset

def get_model(which_model, return_layers=False, imagenet_path='/data/codec/imagenet', device='cuda', subsample=None,subclasses=None,dataloader_only=False,**kwargs):
    
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

    elif which_model == 'mobilenet_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)

    elif which_model == 'mobilenet_large':
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)

    elif which_model=='vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().to(
            device)

        transform = transforms.Compose(
            [transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])])

    
    elif which_model == 'convnext':
        model = timm.create_model('convnext_small.fb_in1k', pretrained=True).eval().to(device)

        transform = transforms.Compose(
            [transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    
    model.to(device)
    model.eval()

    if which_model != 'convnext':
        if which_model != 'mobilenet_lam':
            if which_model != 'vit':
                transform = weights.transforms()

    
    if subsample is not None or subclasses is not None:
        val_dataset = CustomImageNetDataset(root=imagenet_path,
                                                split='val',
                                                transform=transform,
                                                subsample=subsample,
                                                subclasses=subclasses)

    else:
        val_dataset = datasets.ImageNet(root=imagenet_path,
                                        split='val',
                                        transform=transform)



    val_dataloader = DataLoader(val_dataset,
                                batch_size=kwargs.get('batch_size', 128),
                                num_workers=kwargs.get('num_workers', 32),
                                pin_memory=kwargs.get('pin_memory', False),
                                shuffle=(kwargs.get('shuffle', False)))

    if dataloader_only:
        return val_dataloader

    if return_layers is False:
        return model, val_dataset, val_dataloader
    else:
        # Extract the specific layer target from kwargs, default to 'block'
        layer_type = kwargs.get('layer_type', 'block') 
        if 'resnet' in which_model:
            model_layers = []
            for li, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
                for block in layer:
                    model_layers.append(block)

            print('Found {} layers'.format(len(model_layers)))
            
            return model, val_dataset, val_dataloader, model_layers
        elif 'mobilenet' in which_model:
            if 'lam' in which_model:
                model_layers = []
                for block in model.blocks:
                    for layer in block:
                        model_layers.append(layer)
            else:
                model_layers = []
                for block in model.features:
                    model_layers.append(block)

            print('Found {} layers'.format(len(model_layers)))
            
            return model, val_dataset, val_dataloader, model_layers

        elif 'convnext' in which_model:
            model_layers = []
            for stage in model.stages:
                for block in stage.blocks:
                    model_layers.append(block)

            print('Found {} layers'.format(len(model_layers)))
            
            return model, val_dataset, val_dataloader, model_layers
        elif 'vit' in which_model:
            model_layers = []
            # Iterate through the blocks in the timm Vision Transformer
            for block in model.blocks:
                
                if layer_type == 'block':
                    # 1. Token Features
                    # We return the whole block. 
                    # HOOK STRATEGY: Hook the OUTPUT of this module.
                    model_layers.append(block)
                    
                elif layer_type == 'mlp':
                    # 2. MLP Neurons
                    # We return the second linear layer (down_proj).
                    # HOOK STRATEGY: Hook the INPUT of this module (Post-GELU activations).
                    # timm structure: block.mlp.fc2
                    model_layers.append(block.mlp.fc2)
                    
                elif layer_type == 'attention':
                    # 3. Attention Channels (individual dimensions in z vector)
                    # We return the final projection layer (W_O).
                    # HOOK STRATEGY: Hook the INPUT of this module (The 'z' vector).
                    # Each index in head*head_dim is treated as its own channel.
                    # timm structure: block.attn.proj
                    model_layers.append(block.attn.proj)
                    
                elif layer_type == 'attn_heads':
                    # 4. Attention Heads (entire heads as channels)
                    # We return the final projection layer (W_O).
                    # HOOK STRATEGY: Hook the INPUT of this module (The 'z' vector).
                    # Each head is treated as a single large channel.
                    # timm structure: block.attn.proj
                    model_layers.append(block.attn.proj)

            print(f'Found {len(model_layers)} layers of type {layer_type}')
            
            return model, val_dataset, val_dataloader, model_layers

def get_rgb_dataset(imagenet_path='/data/codec/imagenet', batch_size=64):
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

    return val_dataset, val_dataloader

