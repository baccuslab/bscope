import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Logger:
    """
    A class that logs activations and gradients for multiple layers in a PyTorch model.
    """

    def __init__(self, layers_list):
        """
        Initialize the logger with a list of layers to track.
        
        Args:
            layers_list (list): List of PyTorch modules to track
        """
        self.layers = layers_list
        self.activations = [[] for i in range(len(layers_list))]
        self.gradients = [[] for i in range(len(layers_list))]
        self.handles = []

        # Register hooks for each layer
        for idx, layer in enumerate(self.layers):
            # Forward hook to capture activations
            handle_fwd = layer.register_forward_hook(
                lambda module, input, output, idx=idx: self._store_activation(
                    idx, output))

            # Backward hook to capture gradients
            handle_bwd = layer.register_backward_hook(
                lambda module, grad_input, grad_output, idx=idx: self.
                _store_gradient(idx, grad_output[0]))

            self.handles.append(handle_fwd)
            self.handles.append(handle_bwd)

    def _store_activation(self, idx, output):
        """Store the activation of a specific layer."""
        if isinstance(output, tuple):
            self.activations[idx] = output[0].cpu().detach().numpy()
        else:
            self.activations[idx] = output.cpu().detach().numpy()

    def _store_gradient(self, idx, grad):
        """Store the gradient of a specific layer."""
        self.gradients[idx] = grad.cpu().detach().numpy()

    def get_activation(self, idx):
        """
        Get the activation for a specific layer by index.
        
        Args:
            idx (int): Index of the layer in the original list
            
        Returns:
            Tensor: The activation tensor
        """
        return self.activations[idx]

    def get_gradient(self, idx):
        """
        Get the gradient for a specific layer by index.
        
        Args:
            idx (int): Index of the layer in the original list
            
        Returns:
            Tensor: The gradient tensor
        """
        return self.gradients[idx]

    def remove_hooks(self):
        """Remove all hooks to prevent memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
