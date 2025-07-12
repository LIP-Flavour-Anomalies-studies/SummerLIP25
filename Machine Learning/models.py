"""
Defines the neural network.
Teresa 12/07/2025
"""
import torch.nn as nn

class ClassificationModel(nn.Module):
    """
    Simple feedforward network with:
    - A single linear layer from input to output
    - A sigmoid activation for binary classification
    """
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)