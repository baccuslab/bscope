from .inspector import Inspector
from .utils import ei_split
import numpy as np
from IPython import embed
import torch
import torch.nn as nn

import torch

import numpy as np

def torch_normalize_batch_across_cyx(array):
    # array shape: [B, C, Y, X]
    
    # Get batch size
    b_size = array.shape[0]
    
    # Reshape to [B, C*Y*X] to compute norm across all C, Y, X dimensions
    reshaped = array.reshape(b_size, -1)
    
    # Compute the norm for each batch item
    norms = torch.norm(reshaped, dim=1, keepdim=True)
    
    # Avoid division by zero
    norms = torch.clamp(norms, min=1e-8)
    
    # Normalize each batch item
    normalized_reshaped = reshaped / norms
    
    # Reshape back to original shape
    normalized = normalized_reshaped.reshape(array.shape)
    
    return normalized
def normalize_batch_across_cyx(array):
    # array shape: [B, C, Y, X]
    
    # Get batch size
    b_size = array.shape[0]
    
    # Reshape to [B, C*Y*X] to compute norm across all C, Y, X dimensions
    reshaped = array.reshape(b_size, -1)
    
    # Compute the norm for each batch item
    norms = np.linalg.norm(reshaped, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms = np.clip(norms, 1e-8, None)
    
    # Normalize each batch item
    normalized_reshaped = reshaped / norms
    
    # Reshape back to original shape
    normalized = normalized_reshaped.reshape(array.shape)
    
    return normalized

class Scope:
    """
    last_X is the full
    X is the appropriate reduction
    log_X is the logging of X
    """

    def __init__(self, model, layer_list, hook_input=False, to_numpy=True):
        model.eval()
        self.model = model
        self.layer_list = layer_list
        
        self.to_numpy = to_numpy
        self.inspector = Inspector(layer_list, hook_input=hook_input, to_numpy=to_numpy)
        self.num_layers = len(layer_list)

        self.reduction = None
        self.logging = False

        self.contribution_type = None
        self.contribution_target = None

    def use_smooth_grad(self, sigma=0.2, steps=5):
        self.contribution_type = 'smooth_grad'
        self.sigma = sigma
        self.steps = steps

    def use_int_grad(self, steps=5):
        self.contribution_type = 'int_grad'
        self.steps = steps

    def use_act_grad(self):
        self.contribution_type = 'act_grad'
    
    def use_act_normgrad(self):
        self.contribution_type='act_normgrad'

    def use_normact_normgrad(self):
        self.contribution_type='normact_normgrad'

    def use_jacobians(self):
        self.contribution_type = 'jacobians'

    def wrt_entropy(self, softmax=True):
        self.contribution_target = 'entropy'

        self.softmax=softmax

    def wrt_output_neuron(self, neuron_index=0, softmax=False):
        self.contribution_target = 'output_neuron'
        self.neuron_index = neuron_index
        self.softmax = softmax

    def wrt_topk(self, k=5, softmax=True):
        self.contribution_target = 'topk'
        self.k = k
        self.softmax = softmax

    def wrt_sum(self, softmax=False):
        self.softmax = softmax
        self.contribution_target = 'sum'

    def wrt_surprisal(self, softmax=False):
        self.contribution_target = 'surprisal'
        self.surprisal_mu = None
        self.surprisal_sigma_inv = None
        self.softmax = softmax  # Raw neural outputs

    def wrt_contrastive_top2(self, softmax=True):
        self.contribution_target = 'contrastive_top2'
        self.softmax = softmax

    def set_surprisal_stats(self, mu, sigma_inv):
        self.surprisal_mu = mu
        self.surprisal_sigma_inv = sigma_inv


    def log_start(self, reduction=None, heads=None):
        self.logging = True

        self.log_gradients = [[] for i in range(self.num_layers)]
        self.log_activations = [[] for i in range(self.num_layers)]
        self.log_contributions = [[] for i in range(self.num_layers)]

        self.log_outputs = []

        self.reduction = reduction
        self.heads = heads

    def log_stop(self):
        for i in range(self.num_layers):
            self.log_gradients[i] = np.concatenate(self.log_gradients[i])
            self.log_activations[i] = np.concatenate(self.log_activations[i])
            self.log_contributions[i] = np.concatenate(
                self.log_contributions[i])

        self.logging = False

    def compute_target(self, y):
        # Use a local variable to avoid modifying the input tensor reference
        y_target = y
        if self.softmax:
            y_target = torch.softmax(y, dim=-1)

        if self.contribution_target == 'entropy':
            entropy = -torch.sum(y_target * torch.log(y_target + 1e-9), dim=1)
            return entropy
        
        elif self.contribution_target == 'output_neuron':
            return y_target[:, self.neuron_index]
        
        elif self.contribution_target == 'topk':
            sorted, indices = torch.topk(y_target, self.k, dim=-1)
            return sorted.sum(dim=1)
        
        elif self.contribution_target == 'sum':
            return y_target.sum(dim=1)
        
        elif self.contribution_target == 'surprisal':
            if self.surprisal_mu is None or self.surprisal_sigma_inv is None:
                raise ValueError("Surprisal statistics not set. Call set_surprisal_stats() first.")
        
            # Convert numpy arrays to tensors on the right device
            mu_tensor = torch.from_numpy(self.surprisal_mu).to(y_target.device).float()
            sigma_inv_tensor = torch.from_numpy(self.surprisal_sigma_inv).to(y_target.device).float()
        
            # Compute (y - μ)ᵀ Σ⁻¹ (y - μ)
            centered = y_target - mu_tensor  # [batch_size, n_neurons]
            surprisal = 0.5 * torch.sum((centered @ sigma_inv_tensor) * centered, dim=1)  # [batch_size]
            return surprisal
        elif self.contribution_target == 'contrastive_top2':
            top2_values, top2_indices = torch.topk(y, 2, dim=-1)
            
            # Compute difference: top-1 minus top-2
            contrastive_score = top2_values[:, 0] - top2_values[:, 1]
            
            # Backward through the difference
            return contrastive_score
        else:
            raise ValueError(f"Unknown contribution target: {self.contribution_target}")

    def backward_pass(self, y):
        target = self.compute_target(y)
        target.sum().backward()

    def __call__(self, input_tensor):
        if self.contribution_type is None:
            raise ValueError(
                "Contribution type not set. Use smooth_grad, int_grad, or act_grad."
            )
        if self.contribution_target is None:
            raise ValueError(
                "Contribution target not set. Use to_entropy, to_output_neuron, or to_topk."
            )

        # Prepare the stimulus
        if self.contribution_type == 'int_grad':
            self.stim = interpolate_stim(input_tensor, self.steps)
        elif self.contribution_type == 'smooth_grad':
            self.stim = corrupt_stim(input_tensor, self.sigma, self.steps)
        elif self.contribution_type == 'act_normgrad' or self.contribution_type == 'normact_normgrad' or self.contribution_type == 'jacobians' or self.contribution_type == 'act_grad':
            self.stim = input_tensor.unsqueeze(0)
            # self.stim.requires_grad = True
            self.steps = 0

        else:
            raise ValueError(
                f"Unknown contribution type: {self.contribution_type}")

        self.last_activations = [[] for i in range(self.num_layers)]
        self.last_gradients = [[] for i in range(self.num_layers)]
        self.last_outputs = []

        for step in range(self.steps+1):
            self.model.zero_grad()
            # get device of the model
            device = next(self.model.parameters()).device
            y = self.model(self.stim[step].to(device))

            self.backward_pass(y) # Already contains softmax

            # Operate on a detached copy so we don't modify the autograd graph used for backward()
            y = y.detach()
            if self.softmax:
                y = torch.softmax(y, dim=-1)

            self.last_outputs.append(y.detach().cpu().numpy())

            [
                self.last_activations[i].append(self.inspector.activations[i])
                for i in range(self.num_layers)
            ]
            [
                self.last_gradients[i].append(self.inspector.gradients[i])
                for i in range(self.num_layers)
            ]

        self.last_activations = [
            self.last_activations[i] for i in range(self.num_layers)
        ]

        self.last_gradients = [
            self.last_gradients[i] for i in range(self.num_layers)
        ]

        if self.to_numpy:
            self.last_activations = [np.array(self.last_activations[i]) for i in range(self.num_layers)]
            self.last_gradients = [np.array(self.last_gradients[i]) for i in range(self.num_layers)]
        
        if self.contribution_type == 'act_normgrad':
            contributions = []
            for layer in range(self.num_layers):
                act = self.last_activations[layer][0]

                if self.to_numpy:
                    grad = self.last_gradients[layer][0]
                    norm_grad = normalize_batch_across_cyx(grad)
                    contributions.append(np.array(act * norm_grad))
                else:
                    grad = self.last_gradients[layer][0]
                    norm_grad = torch_normalize_batch_across_cyx(grad)
                    contributions.append(act * norm_grad)
            self.activations = [
                self.last_activations[layer][0]
                for layer in range(self.num_layers)
            ]
            self.gradients = [
                self.last_gradients[layer][0]
                for layer in range(self.num_layers)
            ]
        
        elif self.contribution_type == 'normact_normgrad':
            contributions = []
            for layer in range(self.num_layers):
                act = self.last_activations[layer][0]
                grad = self.last_gradients[layer][0]


                if self.to_numpy:
                    contributions.append(np.array(norm_act* norm_grad))
                    norm_act = normalize_batch_across_cyx(act)
                    norm_grad = normalize_batch_across_cyx(grad)
                else:
                    contributions.append(norm_act* norm_grad)
                    norm_act = torch_normalize_batch_across_cyx(act)
                    norm_grad = torch_normalize_batch_across_cyx(grad)

            self.activations = [
                self.last_activations[layer][0]
                for layer in range(self.num_layers)
            ]
            self.gradients = [
                self.last_gradients[layer][0]
                for layer in range(self.num_layers)
            ]

        elif self.contribution_type == 'act_grad':
            contributions = []
            for layer in range(self.num_layers):
                act = self.last_activations[layer][0]
                grad = self.last_gradients[layer][0]

                if self.to_numpy:
                    contributions.append(np.array(act * grad))
                else:
                    contributions.append(act * grad)

            self.activations = [
                self.last_activations[layer][0]
                for layer in range(self.num_layers)
            ]
            self.gradients = [
                self.last_gradients[layer][0]
                for layer in range(self.num_layers)
            ]
        elif self.contribution_type == 'int_grad':
            contributions = []
            for layer in range(self.num_layers):
                interp_activations = self.last_activations[layer]
                interp_gradients = self.last_gradients[layer]

                if self.to_numpy:
                    contributions.append(
                        np.array(
                            interneuron_integral_approximation(
                                interp_activations, interp_gradients)))
                else:
                    contributions.append(
                        torch_interneuron_integral_approximation(
                            interp_activations, interp_gradients))

            self.activations = [
                self.last_activations[layer][-1]
                for layer in range(self.num_layers)
            ]
            self.gradients = [
                self.last_gradients[layer][-1]
                for layer in range(self.num_layers)
            ]

        elif self.contribution_type == 'smooth_grad':
            contributions = []
            for layer in range(self.num_layers):
                corrupt_activations = self.last_activations[layer]
                corrupt_gradients = self.last_gradients[layer]
                contributions.append(
                    np.mean(corrupt_activations * corrupt_gradients, axis=0))

                self.activations = [
                    np.mean(np.array(self.last_activations[layer]), axis=0)
                    for layer in range(self.num_layers)
                ]
                self.gradients = [
                    np.mean(np.array(self.last_gradients[layer]), axis=0)
                    for layer in range(self.num_layers)
                ]
        elif self.contribution_type == 'grad_cam':
            contributions = []
            for layer in range(self.num_layers):
                act = self.last_activations[layer][0]  # [B, C, H, W]
                grad = self.last_gradients[layer][0]   # [B, C, H, W]
                
                # Global average pool gradients to get weights
                weights = np.mean(grad, axis=(2, 3), keepdims=True)  # [B, C, 1, 1]
                
                # Weighted sum over channels
                cam = np.sum(weights * act, axis=1, keepdims=True)  # [B, 1, H, W]
                
                # ReLU to keep only positive contributions
                cam = np.maximum(cam, 0)
                
                contributions.append(cam)
        else:
            raise ValueError(
                f"Unknown contribution type: {self.contribution_type}")

        self.contributions = contributions

        if self.logging:
            for layer in range(self.num_layers):
                g = np.array(self.gradients[layer])
                a = np.array(self.activations[layer])
                c = np.array(self.contributions[layer])
                
                if self.reduction is not None:
                    if 'ei_split' in self.reduction:
                        g = ei_split(g)
                        a = ei_split(a)
                        c = ei_split(c)
                    if 'spatial_sum' in self.reduction:
                        g = g.sum((2, 3))
                        a = a.sum((2, 3))
                        c = c.sum((2, 3))

                    if 'patch_ei_split' in self.reduction or 'mlp_ei_split' in self.reduction or 'attention_ei_split' in self.reduction:
                        # (batch, tokens, hidden), where hidden is like channels, or
                        # (batch, tokens, mlp_hidden), where mlp_hidden is like channels, or
                        # (batch, tokens, heads*head_dim), where heads*head_dim is like channels
                        g = ei_split(g, dim=-1)
                        a = ei_split(a, dim=-1)
                        c = ei_split(c, dim=-1)

                    if 'patch_sum' in self.reduction or 'mlp_sum' in self.reduction or 'attention_sum' in self.reduction:
                        g = g.sum(1)
                        a = a.sum(1)
                        c = c.sum(1)

                    if 'attn_head_ei_split' in self.reduction:
                        # (batch, tokens, heads*head_dim), where heads are like channels
                        # First reshape to (batch, tokens, heads, head_dim)
                        n_heads = self.heads
                        head_dim = g.shape[-1] // n_heads
                        g = g.reshape(g.shape[0], g.shape[1], n_heads, head_dim)
                        a = a.reshape(a.shape[0], a.shape[1], n_heads, head_dim)
                        c = c.reshape(c.shape[0], c.shape[1], n_heads, head_dim)
                        g = ei_split(g, dim=-2)
                        a = ei_split(a, dim=-2)
                        c = ei_split(c, dim=-2)
                    
                    if 'attn_head_sum' in self.reduction:
                        # Sum over both tokens and head_dim to get (batch, heads)
                        g = g.sum((-1,-3))
                        a = a.sum((-1,-3))
                        c = c.sum((-1,-3))

                self.log_gradients[layer].append(g)
                self.log_activations[layer].append(a)
                self.log_contributions[layer].append(c)

        return y


def corrupt_stim(stim, sigma=0.03, steps=10):
    """
    Corrupt the stimulus by adding Gaussian noise.
    """
    stim = stim.detach()
    corrupted_stim = []
    for step in range(steps):
        noise = torch.randn_like(stim) * sigma
        corrupted_stim.append(stim + noise)
    corrupted_stim = torch.stack(corrupted_stim)
    corrupted_stim.requires_grad = True

    return corrupted_stim


def interpolate_stim(stim, steps=10):
    """
    Interpolate the stimulus to create a series of inputs for integrated gradients.
    """
    stim = stim.detach()
    baseline = torch.zeros_like(stim)

    interp_stim = [
        baseline + (float(i) / steps) * (stim - baseline)
        for i in range(0, steps + 1)
    ]

    interp_stim = torch.stack(interp_stim)
    interp_stim.requires_grad = True
    return interp_stim


def torch_interneuron_integral_approximation(acts, grads):
    """
    Trapezoidal integral approximation for integrated gradients.
    """
    igs = []
    for i, (a, g) in enumerate(zip(acts, grads)):
        if i == 0:
            last_act = a
            continue

        diff_act = a - last_act
        last_act = a

        trapezoidal = grads[i - 1] + grads[i]

        trapezoidal /= 2

        ig = trapezoidal * diff_act
        igs.append(ig)
    
    igs = torch.stack(igs)
    igs = torch.sum(igs, axis=0)

    return igs
def interneuron_integral_approximation(acts, grads):
    """
    Trapezoidal integral approximation for integrated gradients.
    """
    igs = []
    for i, (a, g) in enumerate(zip(acts, grads)):
        if i == 0:
            last_act = a
            continue

        diff_act = a - last_act
        last_act = a

        trapezoidal = grads[i - 1] + grads[i]

        trapezoidal /= 2

        ig = trapezoidal * diff_act
        igs.append(ig)

    igs = np.array(igs)
    igs = np.sum(igs, axis=0)

    return igs
