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
