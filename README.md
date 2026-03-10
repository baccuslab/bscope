# bscope

**bscope** is a PyTorch library for neural network interpretation through gradient-based attribution methods. It helps you understand which neurons and layers contribute most to your model's predictions. It supports CNNs, fully-connected networks, and Vision Transformers (via `timm`).

## Architecture

```
bscope/
|
|-- scope.py            Scope: main entry point and orchestrator
|       |                 - configures attribution method & target
|       |                 - runs forward/backward passes
|       |                 - produces per-layer contributions
|       |
|       v
|-- inspector.py        Inspector: hook registration
|       |                 - attaches forward/backward hooks to layers
|       |                 - captures activations and gradients
|       |
|       v
|-- disruptor.py        Disruptor / AttentionDisruptor: causal perturbation
|                         - modifies activations mid-forward-pass
|                         - ablation studies (destroy, corrupt, modal, etc.)
|                         - ViT attention head disruption
|
|-- jacobian.py         Jacobian: direct derivative computation
|                         - finite-difference and autograd Jacobians
|                         - output-to-layer and layer-to-input
|
|-- sae.py              Sparse Autoencoders: dictionary learning
|                         - NNSTSAE, STSAE, SigThreshSAE, SSSAE
|                         - encoder/decoder with thresholding
|
|-- metrics.py          Metrics: evaluation functions
|                         - norms (l0, l1, l2, lp)
|                         - reconstruction losses (l1, l2, r2)
|                         - sparsity (hoyer, kappa_4)
|                         - dictionary distance (hungarian, cosine)
|                         - distribution (wasserstein, frechet)
|
|-- utils.py            Utils: analysis and selection tools
|                         - significance selection (threshold, kmeans, otsu, etc.)
|                         - ei_split, participation ratio, AUC
|                         - config parsing, plot styling
|
`-- ic/                 Image Classification subpackage
    |-- models.py           Model loading (ResNet, AlexNet, MobileNet, ViT, ConvNeXt)
    |-- evaluation.py       Top-1/Top-5 and per-class accuracy
    |-- custom_dataset.py   ImageNet dataset with subsampling
    |-- mode_summary.py     HDF5-based mode analysis (LayerSummary, ModeSummary)
    |-- semantic_utils.py   Semantic similarity and hierarchy tools
    |-- visualization.py    Attribution visualization (mode maps, CWIRF masks)
    `-- load_contribution_data.py   HDF5 contribution data loading
```

### Core Pipeline

```
Input Tensor
     |
     v
+-----------+     +-------------+
|   Scope   |---->|  Inspector  |  (hooks into layers)
+-----------+     +-------------+
     |                  |
     |   forward pass   |---> captures activations
     |   backward pass  |---> captures gradients
     |                  |
     v                  v
  contributions = activations * gradients  (per layer)
```

## Installation

### Dependencies

Core: `torch`, `torchvision`, `numpy`, `scipy`, `timm`, `matplotlib`, `tqdm`, `scikit-learn`

```bash
pip install -e .
```

## Quick Start

See `quickstart.py` for a complete working example, or follow the tutorial below.

## Tutorial: Understanding Neural Network Contributions

### 1. Create a Simple Model

First, let's create a simple convolutional neural network:

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from bscope import Scope

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model and create random input
model = SimpleCNN()
input_tensor = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST-like images
```

### 2. Set Up the Scope Object

The `Scope` object is the core of bscope. It tracks activations and gradients for specified layers:

```python
# Specify which layers to track
layer_list = [model.conv1, model.conv2, model.fc1]

# Create the Scope object
scope = Scope(model, layer_list)
```

**Parameters:**
- `model`: Your PyTorch model
- `layer_list`: List of layers to track (can be any `nn.Module`)
- `to_numpy`: Whether to convert results to numpy arrays (default: True)

### 3. Configure Attribution Method

Choose how to compute contributions. Integrated gradients is recommended:

```python
# Use integrated gradients with 20 interpolation steps
scope.use_int_grad(steps=20)
```

**Available methods:**
- `use_int_grad(steps=20)`: Integrated gradients (recommended)
- `use_smooth_grad(sigma=0.2, steps=5)`: SmoothGrad
- `use_act_grad()`: Activation × Gradient
- `use_act_normgrad()`: Activation × Normalized Gradient
- `use_normact_normgrad()`: Normalized Activation × Normalized Gradient
- `use_jacobians()`: Jacobian-based attribution

### 4. Configure Attribution Target

Specify what output you want to attribute to:

```python
# Attribute to the top-5 predicted classes
scope.wrt_topk(k=5, softmax=True)
```

**Available targets:**
- `wrt_topk(k=5, softmax=True)`: Top-k logits/probabilities
- `wrt_output_neuron(neuron_index=0)`: Specific output neuron
- `wrt_entropy(softmax=True)`: Prediction entropy
- `wrt_sum()`: Sum of all outputs
- `wrt_surprisal()`: Surprisal (requires setting stats with `set_surprisal_stats()`)
- `wrt_contrastive_top2(softmax=True)`: Top-1 minus top-2 logit difference

### 5. Run the Attribution

Simply call the scope object on your input:

```python
# Compute attributions
output = scope(input_tensor)

# Access the results
contributions = scope.contributions  # List of contribution arrays, one per layer
activations = scope.activations      # List of activation arrays
gradients = scope.gradients          # List of gradient arrays
```

### 6. Visualize Layer Contributions

Plot the contributions for a specific layer:

```python
import numpy as np

# Get contributions for the second convolutional layer (index 1)
conv2_contributions = scope.contributions[1]  # Shape: [batch, channels, height, width]

# Aggregate across spatial dimensions and batch
contrib_per_channel = np.abs(conv2_contributions).sum(axis=(0, 2, 3))

# Plot
plt.figure(figsize=(10, 4))
plt.bar(range(len(contrib_per_channel)), contrib_per_channel)
plt.xlabel('Channel Index')
plt.ylabel('Total Absolute Contribution')
plt.title('Contributions per Channel in conv2 Layer')
plt.tight_layout()
plt.savefig('contributions.png')
plt.show()
```

## Advanced Usage

### Logging Multiple Forward Passes

Track contributions across multiple inputs:

```python
# Start logging with optional reduction
scope.log_start(reduction=['spatial_sum'])  # Sum over spatial dimensions

# Run multiple forward passes
for batch in dataloader:
    scope(batch)

# Stop logging and concatenate results
scope.log_stop()

# Access logged data
all_contributions = scope.log_contributions  # List of concatenated arrays
all_activations = scope.log_activations
all_gradients = scope.log_gradients
```

**Reduction options:**
- `'spatial_sum'`: Sum over spatial dimensions (H, W)
- `'ei_split'`: Split into positive and negative contributions
- `'patch_sum'`: Sum over patches
- `'patch_ei_split'`: Split patches into positive/negative
- `'attn_head_sum'`: Sum over attention heads
- `'attn_head_ei_split'`: Split attention heads into positive/negative
- `'mlp_sum'`: Sum over MLP dimensions
- `'mlp_ei_split'`: Split MLP into positive/negative
- `'attention_sum'`: Sum over attention dimensions
- `'attention_ei_split'`: Split attention into positive/negative

### Different Attribution Methods

#### Integrated Gradients
Integrates gradients along the path from a baseline (zeros) to the input:

```python
scope.use_int_grad(steps=20)  # More steps = more accurate but slower
```

#### SmoothGrad
Averages gradients over noisy versions of the input:

```python
scope.use_smooth_grad(sigma=0.2, steps=10)  # sigma controls noise level
```

#### Activation × Gradient
Simple element-wise product (fastest but less principled):

```python
scope.use_act_grad()
```

### Visualizing Contributions

For convolutional layers, you can visualize spatial contributions:

```python
# Get contributions for first sample, first channel
contrib_map = conv2_contributions[0, 0, :, :]  # Shape: [height, width]

plt.imshow(contrib_map, cmap='RdBu_r')
plt.colorbar()
plt.title('Spatial Contribution Map')
plt.savefig('spatial_contributions.png')
```

For fully-connected layers:

```python
# Get contributions for the fully connected layer
fc_contributions = scope.contributions[2]  # Shape: [batch, neurons]

# Plot average contribution per neuron
avg_contrib = np.abs(fc_contributions).mean(axis=0)

plt.figure(figsize=(12, 4))
plt.bar(range(len(avg_contrib)), avg_contrib)
plt.xlabel('Neuron Index')
plt.ylabel('Average Absolute Contribution')
plt.title('Contributions per Neuron in fc1 Layer')
plt.savefig('fc_contributions.png')
```

### Causal Perturbation with Disruptor

The `Disruptor` class modifies layer activations mid-forward-pass for ablations:

```python
from bscope import Disruptor

# Zero out specific channels in a layer
disruptor = Disruptor(model.conv2, channels=[0, 3, 7], style='destroy')
disruptor.activate()

output = model(input_tensor)  # Forward pass with channels ablated

disruptor.deactivate()
```

**Disruption styles:**
- `'destroy'`: Zero out channels
- `'corrupt'`: Add Gaussian noise
- `'modal_offset'`: Offset based on mode vector
- `'similarity_offset'`: Offset based on cosine similarity
- `'modal_corruption'`: Noise scaled by mode vector
- `'mode_weighted'`: Suppress by normalized mode weights
- `'patch_destroy'`: Zero out patches (ViT)
- `'mlp_destroy'`: Destroy MLP input (ViT)
- `'attn_head_destroy'`: Destroy attention head input (ViT)
- `'attn_destroy'`: Destroy attention input (ViT)

#### AttentionDisruptor (Vision Transformers)

For ViT attention head ablation:

```python
from bscope import AttentionDisruptor, create_attention_disruptors

# Disrupt outgoing connections from specific patches
disruptor = AttentionDisruptor(model.blocks[0].attn, patch_indices=[1, 5], style='outgoing')
disruptor.activate()
output = model(input_tensor)
disruptor.deactivate()
```

**Attention disruption styles:**
- `'outgoing'`: Block outgoing connections from patches
- `'incoming'`: Block incoming connections to patches
- `'bidirectional'`: Block both directions

Use `create_attention_disruptors()` to create disruptors for multiple layers at once.

### Jacobian Computation

Compute direct derivatives between layers:

```python
from bscope import compute_output_jacobian, compute_layer_jacobian

# Jacobian of model output w.r.t. intermediate layer channels (finite differences)
jacobian = compute_output_jacobian(model, model.conv2, input_tensor, output_neurons=[0, 1, 2])

# Jacobian of intermediate layer w.r.t. input (autograd)
layer_jacobian = compute_layer_jacobian(model, model.conv2, input_tensor)
```

### Sparse Autoencoders

Dictionary learning on activations for interpretability:

```python
from bscope import NNSTSAE, load_sae

# Create a non-negative sparse autoencoder
sae = NNSTSAE(input_dim=512, hidden_dim=1024, num_atoms=256, threshold=0.1)

# Forward pass returns (codes, sparse_codes, reconstruction)
codes, z, reconstructed = sae(activations)

# Load a pretrained SAE and process data
codes, z, reconstructed, sae_model = load_sae('path/to/model.pt', data, device='cuda')
```

**SAE architectures:**
- `NNSTSAE`: Non-negative with soft thresholding
- `STSAE`: Standard soft thresholding
- `SigThreshSAE`: Sigmoid-based thresholding
- `SSSAE`: Smooth sigmoid sparsity

### Image Classification Subpackage (`bscope.ic`)

Specialized tools for vision model analysis:

```python
from bscope.ic import get_model, calculate_accuracy, CustomImageNetDataset

# Load a pretrained model with its layers and dataloader
model, layers, val_loader = get_model('resnet50', return_layers=True, imagenet_path='/data/imagenet')

# Evaluate accuracy
top1, top5 = calculate_accuracy(model, val_loader)
```

**Supported models:** ResNet-18/50/101, AlexNet, MobileNet (small/large), ViT, ConvNeXt

**ViT layer types:** `'block'`, `'mlp'`, `'attention'`, `'attn_heads'`

**Additional tools:**
- `calculate_class_accuracy()`: Per-class accuracy with dynamic top-k
- `CustomImageNetDataset`: ImageNet loading with class subsampling
- `ModeSummary` / `LayerSummary`: HDF5-based mode analysis and visualization
- `generate_mode_map()`: Attribution visualization with sign policy controls
- `generate_cwirf_mask()`: Contribution-weighted impulse response field masks
- `load_contribution_data()`: Load HDF5 contribution data

## API Reference

### Scope Class

```python
scope = Scope(model, layer_list, hook_input=False, to_numpy=True)
```

**Parameters:**
- `model`: Your PyTorch model
- `layer_list`: List of layers to track (any `nn.Module`)
- `hook_input`: Hook layer inputs instead of outputs (default: False)
- `to_numpy`: Convert results to numpy arrays (default: True)

**Attribution Methods:**
- `use_int_grad(steps=20)`: Integrated gradients
- `use_smooth_grad(sigma=0.2, steps=5)`: SmoothGrad
- `use_act_grad()`: Activation × Gradient
- `use_act_normgrad()`: Activation × Normalized Gradient
- `use_normact_normgrad()`: Normalized Activation × Normalized Gradient
- `use_jacobians()`: Jacobian-based attribution

**Attribution Targets:**
- `wrt_topk(k=5, softmax=True)`: Top-k outputs
- `wrt_output_neuron(neuron_index=0, softmax=False)`: Specific neuron
- `wrt_entropy(softmax=True)`: Entropy
- `wrt_sum(softmax=False)`: Sum of outputs
- `wrt_surprisal()`: Surprisal metric
- `wrt_contrastive_top2(softmax=True)`: Top-1 minus top-2 difference

**Logging:**
- `log_start(reduction=None, heads=None)`: Start logging with optional reduction
- `log_stop()`: Stop logging and concatenate results
- `set_surprisal_stats(mu, sigma_inv)`: Configure surprisal computation

**Attributes (after calling scope):**
- `contributions`: List of contribution arrays per layer
- `activations`: List of activation arrays per layer
- `gradients`: List of gradient arrays per layer
- `last_activations`: Raw activations from all interpolation steps
- `last_gradients`: Raw gradients from all interpolation steps
- `log_contributions`: Logged contributions (after `log_stop()`)
- `log_activations`: Logged activations (after `log_stop()`)
- `log_gradients`: Logged gradients (after `log_stop()`)

### Inspector Class

```python
inspector = Inspector(layers_list, hook_input=False, to_numpy=True)
```

Low-level hook management. Registers forward and backward hooks on layers to capture activations and gradients. Used internally by `Scope`.

- `get_activation(idx)`: Retrieve activation for layer at index
- `get_gradient(idx)`: Retrieve gradient for layer at index
- `remove_hooks()`: Clean up all hooks

### Disruptor Class

```python
disruptor = Disruptor(layer, channels, style='destroy', scale=1)
```

- `activate(heads=None)`: Register perturbation hooks
- `deactivate()`: Remove hooks

### AttentionDisruptor Class

```python
attn_disruptor = AttentionDisruptor(attention_module, patch_indices, style='outgoing')
```

- `activate()`: Apply attention disruption
- `deactivate()`: Remove disruption

### Jacobian Functions

- `compute_output_jacobian(model, module, input_tensor, output_neurons=None)`: Finite-difference Jacobian of outputs w.r.t. intermediate layer channels
- `compute_layer_jacobian(model, module, input_tensor)`: Autograd Jacobian of intermediate layer w.r.t. input

### SAE Classes

```python
sae = NNSTSAE(input_dim, hidden_dim, num_atoms, threshold)
codes, z, reconstructed = sae(x)
```

All SAE variants share the same forward interface returning `(codes, sparse_codes, reconstruction)`.

- `NNSTSAE`: Non-negative soft thresholding
- `STSAE`: Standard soft thresholding
- `SigThreshSAE`: Sigmoid-based thresholding
- `SSSAE`: Smooth sigmoid sparsity
- `load_sae(path, data, device, bs=1024, eval_mode=True, alive_threshold=0)`: Load and evaluate a trained SAE

### Metrics Module

**Norms:** `l0`, `l0_eps`, `l1`, `l2`, `lp`

**Losses:** `avg_l2_loss`, `avg_l1_loss`, `relative_avg_l2_loss`, `relative_avg_l1_loss`, `r2_score`

**Sparsity:** `hoyer`, `kappa_4`, `l1_l2_ratio`, `dead_codes`

**Dictionary:** `hungarian_loss`, `cosine_hungarian_loss`, `dictionary_collinearity`, `compute_stability`, `coherence_regularization`

**Distribution:** `wasserstein_1d`, `frechet_distance`, `codes_correlation_matrix`

**Signal:** `normalized_mean_square_error`, `cross_entropy_degradation`, `energy_of_codes`, `participation_ratio`

### Utils Module

**Significance selection:**
```python
from bscope import select_significant_indices

indices = select_significant_indices(vector, method='kmeans', param=2)
```
Methods: `'threshold'`, `'percentile'`, `'top_n'`, `'kmeans'`, `'otsu'`, `'std'`

**Other utilities:** `ei_split`, `mtx_corr`, `compute_participation_ratio`, `compute_auc`, `normalized_mean_square_error`, `cross_entropy_degradation`, `img_norm`, `style_plot`, `parse_config`

## Examples

See `quickstart.py` for a complete working example.

## Citation

If you use bscope in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]
