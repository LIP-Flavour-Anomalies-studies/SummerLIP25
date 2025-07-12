"""
Contains loss functions.
Teresa 12/07/2025
"""
import torch
import torch.nn as nn

class BalancedLoss(nn.Module):
    """
    Implements balanced binary cross-entropy loss using class weights.
    """
    def __init(self, alpha=None):
        super(BalancedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Calculate the standard binary cross-entropy loss without reduction
        CE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            CE_loss *= alpha_t
        
        # return the mean of the balanced cross-entropy loss
        return torch.mean(CE_loss)
    
class FocalLoss(nn.Module):
    """
    Implements Focal Loss for class imbalance mitigation.
    """
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

        # probability of correct classification
        pt = torch.exp(-CE_loss)

        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            loss = alpha_t * (1 - pt) ** self.gamma * CE_loss
        else:
            loss = (1 - pt) ** self.gamma * CE_loss
            
        return torch.mean(loss)