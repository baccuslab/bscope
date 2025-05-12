import tqdm
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import grad
from torch.autograd import gradcheck
def compute_output_jacobian(model, module, input_tensor, output_neurons=None):
    """
    Compute the Jacobian of specified output neurons with respect to
    intermediate layer channel activations (summed across spatial dimensions).

    Args:
        model: The neural network model
        module: The specific intermediate layer
        input_tensor: Input to the model (shape [1, 3, 224, 224])
        output_neurons: List of output neuron indices or single index

    Returns:
        Jacobian with shape [num_output_neurons, num_intermediate_channels]
    """
    model.eval()

    # Convert single output neuron to list
    if output_neurons is None:
        output_neurons = [0]
    elif isinstance(output_neurons, int):
        output_neurons = [output_neurons]

    # Step 1: Get intermediate layer activation for the given input
    activation = None

    def forward_hook(mod, inp, out):
        nonlocal activation
        activation = out
        return out

    handle = module.register_forward_hook(forward_hook)
    with torch.no_grad():
        output = model(input_tensor)
    handle.remove()

    # Get number of channels in intermediate layer
    if len(activation.shape) == 4:  # Conv layer
        num_channels = activation.shape[1]
    else:  # Linear layer
        num_channels = activation.shape[1]

    # Create tensor to hold Jacobian results
    jacobian = torch.zeros((len(output_neurons), num_channels))

    # Step 2: For each channel in the intermediate layer,
    # calculate its effect on each output neuron
    for channel_idx in tqdm.tqdm(range(num_channels)):

        # Create a detached copy of the activation
        perturbed_activation = activation.detach().clone()

        # Small perturbation value
        epsilon = 1e-4

        # Apply perturbation to the current channel
        if len(perturbed_activation.shape) == 4:  # Conv layer [B, C, H, W]
            # Store original channel values before perturbation
            original_channel = perturbed_activation[:, channel_idx, :, :].clone()

            # Add small perturbation
            perturbed_activation[:, channel_idx, :, :] += epsilon
        else:  # Linear layer [B, F]
            # Store original feature value
            original_feature = perturbed_activation[:, channel_idx].clone()

            # Add small perturbation
            perturbed_activation[:, channel_idx] += epsilon

        # Forward pass with the perturbed activation
        def inject_activation(mod, inp, out):
            return perturbed_activation

        # Register hook to inject the perturbed activation
        temp_handle = module.register_forward_hook(inject_activation)

        # Get outputs with the perturbed activation
        with torch.no_grad():
            perturbed_output = model(input_tensor)

        # Remove hook
        temp_handle.remove()

        # Reset the activation to its original value
        if len(perturbed_activation.shape) == 4:
            perturbed_activation[:, channel_idx, :, :] = original_channel
        else:
            perturbed_activation[:, channel_idx] = original_feature

        # Calculate change in output for each output neuron
        for out_idx, out_neuron in enumerate(output_neurons):
            if output.dim() > 1:
                original_value = output[0, out_neuron].item()
                perturbed_value = perturbed_output[0, out_neuron].item()
            else:
                original_value = output[out_neuron].item()
                perturbed_value = perturbed_output[out_neuron].item()

            # Calculate finite difference approximation of the derivative
            derivative = (perturbed_value - original_value) / epsilon

            # Store in Jacobian
            jacobian[out_idx, channel_idx] = derivative

    # If only one output neuron, return a 1D tensor
    if len(output_neurons) == 1:
        return jacobian[0]

    return jacobian
def compute_layer_jacobian(model, module, input_tensor):
    model.eval()
    input_tensor = input_tensor.detach().clone().requires_grad_(True)
    activation = {}
    
    def get_activation(name):
        def hook(mod, inp, out):
            if len(out.shape) == 4:
                activation[name] = out.sum(dim=(2, 3))
            else:
                activation[name] = out
            return None
        return hook
    
    hook_handle = module.register_forward_hook(get_activation('target_module'))
    
    def get_layer_output(x):
        _ = model(x)
        return activation['target_module']
    
    layer_wrt_input = torch.autograd.functional.jacobian(get_layer_output, input_tensor)
    hook_handle.remove()
    
    return layer_wrt_input, activation


