"""
Feature selection with Optuna
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

from sklearn.model_selection import train_test_split
from prepdata_v0 import prepdata, ClassificationDataset
from losses import BalancedLoss
from early_stopping import EarlyStopping
from train import train_model

# Add the directory containing variables
sys.path.append(os.path.abspath("Machine_Learning"))
from variable_versions import load_variables

# --- Model Definition --- #

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

# --- Optuna objective function --- #
def objective(trial, x_train, y_train, x_val, y_val, class_weights, models, feature_names):

    input_size = x_train.shape[1]

    # --- Suggest a subset of features ---
    # binary mask for each feature (0 = drop, 1 = keep)
    mask = [trial.suggest_int(f"feat_{i}", 0, 1) for i in range(input_size)]
    selected_indices = [i for i, m in enumerate(mask) if m == 1]

    # ensure at least N features are kept (say between 8 and 16)
    if not (8 <= len(selected_indices) <= 16):
        raise optuna.exceptions.TrialPruned()
    
    # filter datasets by feature mask
    x_train_sel = x_train[:, selected_indices]
    x_val_sel   = x_val[:, selected_indices]

    train_set = ClassificationDataset(x_train_sel, y_train)
    val_set   = ClassificationDataset(x_val_sel, y_val)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=128, shuffle=False)

    # ---- Fixed, simple NN hyperparameters for feature search (from first model) ----
    n_layers = 2
    neurons = [32, 16]
    dropout_rate = 0.1
    learning_rate = 1e-3
    weight_decay = 1e-4

    model = DynamicClassificationModel(len(selected_indices), n_layers, neurons, dropout_rate)

    criterion = BalancedLoss(alpha=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=50, delta=1e-6)

    tl_vector, vl_vector, best_epoch = train_model(model, early_stopping, train_loader, val_loader, criterion, 
                                                    optimizer, num_epochs=500, flag=0, return_losses=True)

    val_loss = -early_stopping.best_score

    # Load best weights before saving
    early_stopping.load_best_model(model)

    # Compare current trial to best trial in study (if any)
    if len(trial.study.trials) == 1 or val_loss < trial.study.best_trial.value:
        selected_features = [feature_names[i] for i in selected_indices]
        # Save the best model and info
        models.clear()
        models.extend([model.state_dict(), trial.params, val_loss,
                        tl_vector, vl_vector, best_epoch, selected_features])

    return val_loss



def main():

    # paths and version 
    dir = "Signal_vs_Background/ROOT_files"
    file_signal = "signal.root"
    file_back = "background.root"
    # Choose training version
    version = 11

    # load data
    x, y = prepdata(dir, file_signal, file_back, version)
    dataset = ClassificationDataset(x, y)

    # load feature names (align with X)
    feature_names = load_variables(version)

    # approx 50% train, 25% val, 25% test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.3333, stratify=y_train_val, random_state=42
    )

    class_weights = torch.tensor([1 / np.sum(y == 0), 1 / np.sum(y == 1)], dtype=torch.float32)
    class_weights /= class_weights.sum()

    models = []

    # Run Optuna for feature search 
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val,
                                           class_weights, models, feature_names), n_trials=100)

    print("Best validation loss:", study.best_value)
    print("Best feature subset:", models[6])

    # --- Guard in case all trials were pruned ---
    if not models:
        print("No valid feature subset found. Try relaxing constraints (min/max features).")
        return

    # Save the best model checkpoint after optimization
    checkpoint_dir = "Machine_Learning/Evaluation/checkpoints_optim"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_features_v{version}.pth")

    torch.save({
        "model_state_dict": models[0],
        "trial_params": models[1],
        "val_loss": models[2],
        "train_loss_curve": models[3],
        "val_loss_curve": models[4],
        "best_epoch": models[5],
        "selected_features": models[6],
        "X_test": x_test.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }, checkpoint_path) 


if __name__ == "__main__":
    main()