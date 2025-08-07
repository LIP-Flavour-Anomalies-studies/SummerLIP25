import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.serialization
import shap
from captum.attr import IntegratedGradients
import torch.nn as nn

# Add path to import variable_versions.py from one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add path to import models.py and train.py from Training folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Training')))

from variable_versions import load_variables
from models import ClassificationModel
from prepdata_v0 import ClassificationDataset
from prepdata_v0 import prepdata
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class LegacyModel(nn.Module):
    """
    Simpler version of the model used in older checkpoints.
    Architecture:
    - One fully connected layer (fc) with sigmoid activation
	(used so I wouldn't have to change models.py)
    """
    def __init__(self, input_size):
        super(LegacyModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))


@torch.no_grad()
def compute_permutation_importance(model, val_loader, variable_list, device):
    """
    Computes permutation feature importance based on F1 Score.

    For each feature:
        - Randomly shuffles its values across the validation set
        - Measures the drop in F1 Score
        - The bigger the drop, the more important the feature is

    Returns a list of tuples (feature_name, importance), sorted by importance.
    """
    model.eval()
    X_all = []
    y_all = []

    # Collect the full validation data
    for inputs, labels in val_loader:
        X_all.append(inputs)
        y_all.append(labels)
    X_all = torch.cat(X_all).to(device)
    y_all = torch.cat(y_all).to(device)

    # Predictions and F1 score with original data
    base_outputs = model(X_all).squeeze()
    base_preds = (base_outputs > 0.5).int()
    base_score = f1_score(y_all.cpu(), base_preds.cpu())

    importances = []

    # For each feature, shuffle and recompute the F1 score
    for i, var in enumerate(variable_list):
        X_shuffled = X_all.clone()
        idx_perm = torch.randperm(X_shuffled.size(0))
        X_shuffled[:, i] = X_shuffled[idx_perm, i]

        outputs = model(X_shuffled).squeeze()
        preds = (outputs > 0.5).int()
        score = f1_score(y_all.cpu(), preds.cpu())

        delta = base_score - score  # drop in performance
        importances.append((var, delta))

    # Sort features by importance (largest drop first)
    return sorted(importances, key=lambda x: x[1], reverse=True)

def compute_shap_importance(model, val_loader, variable_list, device):
    """
    Computes SHAP (SHapley Additive exPlanations) feature importances.

    This method:
        - Uses KernelExplainer to estimate Shapley values for each feature
        - Measures the average absolute SHAP value per feature
        - Higher SHAP value = more impact on model prediction

    Returns a list of (feature_name, importance), sorted descending.
    """
    model.eval()

    # Prepare a sample of validation data
    X_all = []
    for inputs, _ in val_loader:
        X_all.append(inputs)
    X_all = torch.cat(X_all).to(device)

    def model_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        return model(x_tensor).detach().cpu().numpy()

    background = X_all[:100].cpu().numpy()  # small subset as background
    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(X_all[:200].cpu().numpy())  # subset to reduce cost

    # Compute mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    results = list(zip(variable_list, [float(val) for val in mean_abs_shap]))

    return sorted(results, key=lambda x: x[1], reverse=True)

def run_feature_importance(version, loss_type='binary', method='permutation'):
    """
    Loads model and validation data for a given version and loss function,
    computes feature importance using the selected method, and saves the results.

    Args:
        version (int): Variable version to use (as defined in variable_versions.json)
        loss_type (str): Either 'binary' or 'focal', to choose model checkpoint
        method (str): 'permutation', 'shap'
    """
    # Go to project root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(root_dir)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load variables to use based on version
    variable_list = load_variables(version, config_path="Machine_Learning/variable_versions.json")

    # Select checkpoint suffix based on loss type
    suffix = "B" if loss_type == "binary" else "F"
    checkpoint_path = os.path.join("Machine_Learning", "Evaluation", "checkpoints", f"{suffix}_model_checkpoint_v{version}.pth")

    # Load model and weights
    # Use older or newer model architecture based on version:
    # Versions 0, 2, 4 used an older model (1-layer), others use the current 2-layer model
    if version in [ 2, 3, 4]:
        model = LegacyModel(input_size=len(variable_list)).to(device)
    else:
        model = ClassificationModel(input_size=len(variable_list)).to(device)

    with torch.serialization.safe_globals([ClassificationDataset]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ROOT file paths
    dir_path = "Signal_vs_Background/ROOT_files"
    root_mc = "signal.root"
    root_data = "background.root"

    # Load data for specified version
    X, y = prepdata(dir_path, root_mc, root_data, version)

    # Split into validation set (same logic as main.py)
    from sklearn.model_selection import train_test_split
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.3333, stratify=y_train_val, random_state=42)

    # Wrap validation data in Dataset and DataLoader
    val_data = ClassificationDataset(x_val, y_val)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)

    # Choose importance method
    if method == 'permutation':
        results = compute_permutation_importance(model, val_loader, variable_list, device)
    elif method == 'shap':
        results = compute_shap_importance(model, val_loader, variable_list, device)
    else:
        raise ValueError(f"Unknown method: {method}")
    # Define output directory using version + method + loss
    outdir = f"Machine_Learning/Evaluation/version{version}_{method}_{loss_type}/feature_importance"
    os.makedirs(outdir, exist_ok=True)

# Save results as JSON
    with open(os.path.join(outdir, f"{method}_importance.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save plot
    vars_, scores = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.barh(vars_, scores)
    plt.xlabel("Importance Score")
    plt.title(f"{method.replace('_', ' ').title()} (v{version}, {loss_type})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{method}_importance.pdf"))
    plt.close()


if __name__ == "__main__":
    for version in [0,1, 2, 3, 4, 5]:
        for method in ['permutation', 'shap']:
            print(f"\nRunning version={version} | loss_type=binary | method={method}")
            run_feature_importance(version=version, loss_type='binary', method=method)

"""
Interpretation of Feature Importance Methods:

1. Permutation Importance:
   - Shuffles each feature and sees how performance drops.
   - A higher drop = more important feature.
   - Negative values mean model may perform better without that feature.

2. SHAP (SHapley Values):
   - Based on game theory; distributes model output among input features.
   - Measures how much each feature contributes to the prediction.
   - Average absolute SHAP value shows feature impact.

Note: These methods offer complementary insights. SHAP is more interpretable, and permutation is intuitive but model-agnostic.
"""
