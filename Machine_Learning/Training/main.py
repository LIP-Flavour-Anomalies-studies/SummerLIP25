"""
Main script to run everything.
Teresa 12/07/2025
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from prepdata_v0 import prepdata, ClassificationDataset
from models import ClassificationModel
from losses import BalancedLoss, FocalLoss
from train import train_model
from early_stopping import EarlyStopping

sns.set_style("darkgrid")

def main():
    try:
        dir = "/user/u/u25teresaesc/Internship/Signal_vs_Background/ROOT_files"
        file_signal = "signal.root"
        file_back = "background.root"

        x, y = prepdata(dir, file_signal, file_back)
        dataset = ClassificationDataset(x, y)

        # Stratified splitting to keep class proportions
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x, y, test_size=0.25, stratify=y, random_state=42
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.3333, stratify=y_train_val, random_state=42
        )
        # Now approx 50% train, 25% val, 25% test

        # Create datasets
        train_set = ClassificationDataset(x_train, y_train)
        val_set = ClassificationDataset(x_val, y_val)
        test_set = ClassificationDataset(x_test, y_test)

        # Verify lengths
        print("Training set length:", len(train_set))
        print("Testing set length:", len(test_set))
        print("Validation set length:", len(val_set))
        print()

        # Create DataLoader for training and testing
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

        # Initialize the model
        input_size = x.shape[1]
        B_model = ClassificationModel(input_size)
        F_model = ClassificationModel(input_size)

        # Calculate class weights
        class_weights = torch.tensor([1 / np.sum(y == 0), 1 / np.sum( y == 1)], dtype=torch.float32)
        class_weights /= class_weights.sum()

        # Define loss function and optimizer
        B_criterion = BalancedLoss(alpha=class_weights)
        F_criterion = FocalLoss(alpha=class_weights)
        B_optimizer = optim.Adam(B_model.parameters(), lr=0.001)
        F_optimizer = optim.Adam(F_model.parameters(), lr=0.001)

        # Early stopping
        B_early_stopping = EarlyStopping(patience=100, delta=1e-6)
        F_early_stopping = EarlyStopping(patience=100, delta=1e-6)

        # Train models
        print("\nTraining model with Balanced Loss...")
        train_model(B_model, B_early_stopping, train_loader, val_loader, B_criterion, B_optimizer, flag=0)

        print("\nTraining model with Focal Loss...")
        train_model(F_model, F_early_stopping, train_loader, val_loader, F_criterion, F_optimizer, flag=1)

        # Save models
        checkpoint_dir = "/user/u/u25teresaesc/Internship/Machine_Learning/Evaluation"
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save({"model_state_dict": B_model.state_dict(),
                    "optimizer_state_dict": B_optimizer.state_dict(),
                    "dataset": dataset,
                    "test_set": test_loader.dataset}, os.path.join(checkpoint_dir, "B_model_checkpoint_v1.pth"))
        
        torch.save({"model_state_dict": F_model.state_dict(),
                    "optimizer_state_dict": F_optimizer.state_dict(),
                    "dataset": dataset,
                    "test_set": test_loader.dataset}, os.path.join(checkpoint_dir, "F_model_checkpoint_v1.pth"))
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
