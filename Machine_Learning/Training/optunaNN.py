"""
Module to study hyperparameter optimisation.
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from prepdata_v0 import prepdata, ClassificationDataset
from losses import BalancedLoss
from early_stopping import EarlyStopping
from train import train_model

sns.set_style("darkgrid")


class DynamicClassificationModel(nn.Module):
    def __init__(self, input_size, n_layers, neurons, dropout_rate):
        
        super(DynamicClassificationModel, self).__init__()
        layers = []
        in_features = input_size 
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, neurons[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = neurons[i]
        
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective(trial, train_loader, val_loader, input_size, class_weights, models, version):

    # --- Hyperparameters to tune ---
    n_layers = trial.suggest_int("n_layers", 1, 5)
    # Sample neurons per layer individually
    neurons = [trial.suggest_int(f"neurons_l{i}", 8, 128, log=True) for i in range(n_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    model = DynamicClassificationModel(input_size, n_layers, neurons, dropout_rate)

    criterion = BalancedLoss(alpha=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=50, delta=1e-6)

    tl_vector, vl_vector, best_epoch = train_model(model, early_stopping, train_loader, val_loader, criterion, 
                                                    optimizer, num_epochs=400, flag=0, return_losses=True)

    val_loss = -early_stopping.best_score

    # Load best weights before saving
    early_stopping.load_best_model(model)

    # Compare current trial to best trial in study (if any)
    if len(trial.study.trials) == 1 or val_loss < trial.study.best_trial.value:
        # Save the best model and info
        models.clear()
        models.extend([model.state_dict(), trial.params, val_loss,
                        tl_vector, vl_vector, best_epoch])

    return val_loss



def main():

    # paths and version 
    dir = "Signal_vs_Background/ROOT_files"
    file_signal = "signal.root"
    file_back = "background.root"
    # Choose training version
    version = 2

    # load data
    x, y = prepdata(dir, file_signal, file_back, version)
    dataset = ClassificationDataset(x, y)

    # approx 50% train, 25% val, 25% test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.3333, stratify=y_train_val, random_state=42
    )

    train_set = ClassificationDataset(x_train, y_train)
    val_set = ClassificationDataset(x_val, y_val)
    test_set = ClassificationDataset(x_test, y_test)

    # Default, will be overridden in objective but dataloaders need batch size initially
    batch_size = 64

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size*2, shuffle=False)

    input_size = x.shape[1]
    class_weights = torch.tensor([1 / np.sum(y == 0), 1 / np.sum(y == 1)], dtype=torch.float32)
    class_weights /= class_weights.sum()

    models = []

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, input_size, class_weights, models, version), n_trials=50)

    print("Best trial params:", study.best_params)
    print("Best validation loss:", study.best_value)

    # Save the best model checkpoint after optimization
    checkpoint_dir = "Machine_Learning/Evaluation/checkpoints_optim"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_v{version}.pth")

    torch.save({
        "model_state_dict": models[0],
        "trial_params": models[1],
        "val_loss": models[2],
        "train_loss_curve": models[3],
        "val_loss_curve": models[4],
        "best_epoch": models[5],
        "dataset": dataset, 
        "test_set": test_set
    }, checkpoint_path) 

    # Plot loss curve for best trial
    tl_vector, vl_vector, best_epoch = models[3], models[4], models[5]
    indices = range(1, len(tl_vector) + 1)
    plt.figure()
    plt.plot(indices, tl_vector, label="Training", color="navy", markersize=1)
    plt.plot(indices, vl_vector, label="Validation", color="orange", markersize=1)
    plt.scatter(best_epoch + 1, vl_vector[best_epoch], color="black", label="Early Stop", s=64)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    min_loss = min(min(tl_vector), min(vl_vector))
    max_loss = max(min(tl_vector), min(vl_vector))
    plt.ylim(min_loss * 0.95, max_loss * 1.2)
    plt.savefig(f"best_loss_curve_v{version}.pdf")
    plt.close()


if __name__ == "__main__":
    main()
