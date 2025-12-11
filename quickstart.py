import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from bscope import Scope

class SimpleCNN(nn.Module):
    """A simple CNN for demonstration purposes."""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First conv block
        x = self.relu1(self.conv1(x))

        # Second conv block with pooling
        x = self.relu2(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x



print("Initializing model...")
model = SimpleCNN()
model.eval()  # Set to evaluation mode

# Create a batch of random inputs (simulating MNIST-like images)
batch_size = 4
input_tensor = torch.randn(batch_size, 1, 28, 28)
print(f"Input shape: {input_tensor.shape}")


# Create Scope object to track layers and compute contributions 
# We'll track both convolutional layers and the first fully-connected layer

layer_list = [
    model.conv1,  # First convolutional layer
    model.conv2,  # Second convolutional layer
    model.fc1,    # First fully-connected layer
]

# Create the Scope object
# to_numpy=True means results will be converted to numpy arrays (default)
scope = Scope(model, layer_list, to_numpy=True)

# Use integrated gradients with 20 interpolation steps
# Other more efficient methods are available as well (actgrad, act_normgrad)
scope.use_int_grad(steps=20)

# Attribute to the top-5 predicted classes
# Other targets available
scope.wrt_topk(k=5, softmax=False)

# Run the scope - this computes activations, gradients, and contributions as
# attributes of the scope object
output = scope(input_tensor)

# Print shapes of contributions for each layer
for i, contrib in enumerate(scope.contributions):
    print(scope.contributions[i].shape)
    print(scope.activations[i].shape)
    print(scope.gradients[i].shape)

