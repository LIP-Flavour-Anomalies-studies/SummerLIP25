"""
Defines the neural network.
Teresa 12/07/2025
"""
import torch.nn as nn

class ClassificationModel(nn.Module):
    """
    A simple feedforward neural network for binary classification.

    Architecture:
    - Input layer of size `input_size`
    - First hidden layer with 32 neurons and ReLU activation
    - Second hidden layer with 16 neurons and ReLU activation
    - Output layer with 1 neuron and sigmoid activation to produce a probability in [0, 1]
    - Dropout applied (p=0.1) to reduce overfitting
    """
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        # 1st hidden layer
        self.fc1 = nn.Linear(input_size, 32) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        # 2nd hidden layer
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        # Output layer
        self.out = nn.Linear(16, 1)           
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.out(x))
        return x 