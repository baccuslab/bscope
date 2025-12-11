# bscope

**bscope** is a PyTorch library for neural network interpretation through gradient-based attribution methods. It helps you understand which neurons and layers contribute most to your model's predictions.

## Installation

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

## API Reference

### Scope Class

```python
scope = Scope(model, layer_list, to_numpy=True)
```

**Attribution Methods:**
- `use_int_grad(steps=20)`: Integrated gradients
- `use_smooth_grad(sigma=0.2, steps=5)`: SmoothGrad
- `use_act_grad()`: Activation × Gradient
- `use_act_normgrad()`: Activation × Normalized Gradient
- `use_normact_normgrad()`: Normalized Activation × Normalized Gradient

**Attribution Targets:**
- `wrt_topk(k=5, softmax=True)`: Top-k outputs
- `wrt_output_neuron(neuron_index=0, softmax=False)`: Specific neuron
- `wrt_entropy(softmax=True)`: Entropy
- `wrt_sum(softmax=False)`: Sum of outputs
- `wrt_surprisal()`: Surprisal metric

**Logging:**
- `log_start(reduction=None)`: Start logging
- `log_stop()`: Stop logging and concatenate results

**Attributes (after calling scope):**
- `contributions`: List of contribution arrays per layer
- `activations`: List of activation arrays per layer
- `gradients`: List of gradient arrays per layer
- `last_activations`: Raw activations from all interpolation steps
- `last_gradients`: Raw gradients from all interpolation steps

## Examples

See `quickstart.py` for a complete working example.

## Citation

If you use bscope in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]
