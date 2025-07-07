import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

#         Get the gradient for a specific layer by index.


class Disruptor:
    """
    Disruptor class to apply disruptions to the model's activations based on a number of different 
    causal mechanisms. The class can be used to apply disruptions to the activations of a specific layer
    in the model during inference.
    """

    def __init__(self, layer, channels, style='destroy', scale=1):
        self.style = style
        self.layer = layer
        self.scale = scale
        self.channels = channels
        self.hook = None
        self.original_hook = None

    def _hook_fn(self, module, input, output):
        if self.style == 'destroy':
            output[:, self.channels] = 0
        elif self.style == 'patch_destroy':
            output[:, :, self.channels] = 0
        elif self.style == 'corrupt':
            output[:, self.channels] += torch.randn_like(
                output[:, self.channels]) * self.scale
        elif self.style == 'modal_offset':

            noise = torch.ones_like(
                output) * self.scale  # num_samples, num_channels, y, x
            mode = make_torch(self.channels).to(output.device)  # num_channels
            modal_noise = torch.einsum('ijkl,j->ijkl', noise.to(output.device),
                                       mode).to(output.device)
            output += modal_noise

        elif self.style == 'similarity_offset':
            z = output.sum((2, 3))
            # z = z-z.mean()
            zz = torch.from_numpy(self.channels).to(output.device)
            # zz = zz-zz.mean()
            similarities = F.cosine_similarity(z, zz.unsqueeze(0), dim=1)

            noise = torch.ones_like(output) * self.scale
            noise = torch.einsum('ijkl,i->ijkl', noise, similarities)
            output += noise

        elif self.style == 'modal_corruption':

            noise = torch.randn_like(
                output) * self.scale  # num_samples, num_channels, y, x
            mode = make_torch(self.channels).to(output.device)  # num_channels

            modal_noise = torch.einsum('ijkl,j->ijkl', noise.to(output.device),
                                       mode).to(output.device)

            output += modal_noise

        elif self.style == 'mode_weighted':
    # Normalize the raw atom to [0,1]
            atom = torch.from_numpy(self.channels).to(output.device)
            normalized_atom = (atom - atom.min()) / (atom.max() - atom.min())
            
            # Create scaling vector (1.0 = no change, 0.0 = completely suppressed)
            scaling_vector = (1.0 - normalized_atom)
            
            # Apply in-place like the others
            output[:, :] *= scaling_vector.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"Style {self.style} not supported")

        return output

    def activate(self):
        # Find the specified layer in the model
        # Register the hook to zero out specific channels
        self.hook = self.layer.register_forward_hook(self._hook_fn)

    def deactivate(self):
        # Remove the hook to restore normal functionality
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

import torch
import torch.nn.functional as F
from functools import partial

class AttentionDisruptor:
    """
    Disruptor that modifies attention weights in timm ViT models.
    Intercepts attention computation and zeros out specific patch-to-patch connections.
    """
    
    def __init__(self, attention_module, patch_indices, style='outgoing'):
        """
        Args:
            attention_module: The attention module from timm ViT (e.g., model.blocks[i].attn)
            patch_indices: List of patch indices to ablate (0-based, excluding CLS token)
            style: 'outgoing', 'incoming', or 'bidirectional'
        """
        self.attention_module = attention_module
        self.patch_indices = patch_indices
        self.style = style
        
        # Store original forward method
        self.original_forward = attention_module.forward
        self.is_active = False
        
        # Convert patch indices to token indices (add 1 for CLS token)
        self.token_indices = [idx + 1 for idx in patch_indices]
        
    def modified_attention_forward(self, x):
        """
        Modified forward pass that intercepts and modifies attention weights.
        Based on timm's Attention.forward() but with weight modification.
        """
        B, N, C = x.shape
        
        # Standard attention computation (from timm)
        qkv = self.attention_module.qkv(x).reshape(B, N, 3, self.attention_module.num_heads, C // self.attention_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Compute attention weights
        attn = (q @ k.transpose(-2, -1)) * self.attention_module.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention weight modifications
        if self.style == 'outgoing':
            # Prevent specified patches from sending information
            for token_idx in self.token_indices:
                if token_idx < N:  # Safety check
                    attn[:, :, token_idx, :] = 0
                    
        elif self.style == 'incoming':
            # Prevent specified patches from receiving information
            for token_idx in self.token_indices:
                if token_idx < N:  # Safety check
                    attn[:, :, :, token_idx] = 0
                    
        elif self.style == 'bidirectional':
            # Prevent both sending and receiving
            for token_idx in self.token_indices:
                if token_idx < N:  # Safety check
                    attn[:, :, token_idx, :] = 0  # outgoing
                    attn[:, :, :, token_idx] = 0  # incoming
        
        # Apply dropout (if exists)
        attn = self.attention_module.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attention_module.proj(x)
        x = self.attention_module.proj_drop(x)
        
        return x
    
    def activate(self):
        """Replace the attention forward method with our modified version."""
        if not self.is_active:
            self.attention_module.forward = self.modified_attention_forward
            self.is_active = True
            print(f"Activated attention disruption: {self.style} for patches {self.patch_indices}")
    
    def deactivate(self):
        """Restore the original attention forward method."""
        if self.is_active:
            self.attention_module.forward = self.original_forward
            self.is_active = False
            print("Deactivated attention disruption")


# Integration with your existing pipeline
def create_attention_disruptors(model, layers_to_ablate, important_patches_per_layer, style='outgoing'):
    """
    Create attention disruptors for multiple layers.
    
    Args:
        model: timm ViT model
        layers_to_ablate: List of layer indices (e.g., [9, 10, 11])
        important_patches_per_layer: Dict mapping layer_idx to list of patch indices
        style: 'outgoing', 'incoming', or 'bidirectional'
    
    Returns:
        List of AttentionDisruptor objects
    """
    disruptors = []
    
    for layer_idx in layers_to_ablate:
        if layer_idx < len(model.blocks):
            attention_module = model.blocks[layer_idx].attn
            patch_indices = important_patches_per_layer.get(layer_idx, [])
            
            if patch_indices:
                disruptor = AttentionDisruptor(
                    attention_module=attention_module,
                    patch_indices=patch_indices,
                    style=style
                )
                disruptors.append(disruptor)
                print(f"Created attention disruptor for layer {layer_idx} with {len(patch_indices)} patches")
    
    return disruptors

