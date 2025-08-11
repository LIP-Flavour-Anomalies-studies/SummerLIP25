"""
Module that calculates cumulative SHAP importance.
"""
import os
import sys
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add path to import variable_versions.py from one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add path to import models.py and train.py from Training folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Training')))

from variable_versions import load_variables
from models import ClassificationModel
from prepdata_v0 import ClassificationDataset, prepdata

sns.set_style("darkgrid")

# Config 
version = 10
loss_type = "binary"
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cumulative_threshold = 0.90  # keep top features covering 90% importance

# Load Variables 
variable_list = load_variables(version)
input_size = len(variable_list)

# Load Model
suffix = "B" if loss_type == "binary" else "F"
checkpoint_path = os.path.join(
    "Machine_Learning", "Evaluation", "checkpoints",
    f"{suffix}_model_checkpoint_v{version}.pth"
)

model = ClassificationModel(input_size=input_size).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load Data
dir_path = "Signal_vs_Background/ROOT_files"
root_mc = "signal.root"
root_data = "background.root"
X, y = prepdata(dir_path, root_mc, root_data, version)

# Split Data 
_, x_val, _, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

val_dataset = ClassificationDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Prep data for SHAP 
X_all = []
for inputs, _ in val_loader:
    X_all.append(inputs)
X_all = torch.cat(X_all).to(device)

# Define SHAP model wrapper
def model_fn(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    preds = model(x_tensor).detach().cpu().numpy()
    return preds.reshape(-1)  # ensure 1D output


# Compute SHAP values
background = X_all[:100].cpu().numpy()
explainer = shap.KernelExplainer(model_fn, background)
shap_values = explainer.shap_values(X_all[:200].cpu().numpy())

# Calculate mean absolute SHAP
mean_abs_shap = np.abs(shap_values).mean(axis=0)
results = list(zip(variable_list, mean_abs_shap))

# Sort and normalise 
# sort list by second element of each tuple, highest to lowest 
results.sort(key=lambda x: x[1], reverse=True)
total_importance = sum(score for _, score in results)
normalised = [(var, score / total_importance) for var, score in results]
sorted_features = [var for var, _ in normalised]

# Cumulative importance
cumulative_vals = []
running_total = 0
for var, frac in normalised:
    running_total += frac
    cumulative_vals.append(running_total)

# Select features under threshold
selected_features = [
    var for var, cum_val in zip(sorted_features, cumulative_vals)
    if cum_val <= cumulative_threshold
]

# Output
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Cumulative_SHAP'))
os.makedirs(out_dir, exist_ok=True)

# Plot 
plt.figure(figsize=(14, 6))
plt.plot(sorted_features, cumulative_vals, 'o')
plt.xticks(rotation=90)
plt.axhline(y=cumulative_threshold, color="r", linestyle="--", label=f"{cumulative_threshold*100}% threshold")
plt.axvline(x=(sorted_features.index(selected_features[-1]) + 0.5), color="g", linestyle="--", label=f"{len(selected_features)} features at {cumulative_threshold*100}%")
plt.ylabel("Cumulative SHAP Importance")
plt.xlabel("Features")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"v{version}_shap_cumulative_importance.png"))

# Output selected features
print(f"Selected top {len(selected_features)} features covering {cumulative_threshold*100}% importance:")
print(selected_features)

# Retrain model with selected_features list to check performance