"""
One-shot feature selection: SHAP importance combined with correlation-based clustering.
- Correlation matrices (Signal/Background) are pre-computed and loaded from CSV.
- Features are grouped when correlation >= threshold (using the maximum correlation between Signal and Background).
- Within each group, a single representative feature is chosen (the one with highest |mean SHAP|).
- On the set of representatives, cumulative SHAP importance is applied until >= 95% is covered.
- This script only performs feature selection and reporting. No retraining is performed here.
"""

import os
import sys
import json
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ---- Project paths
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))  # Machine_Learning
TRAIN_DIR = os.path.join(BASE, 'Training')
EVAL_DIR = os.path.join(BASE, 'Evaluation')
CORR_DIR = os.path.join(EVAL_DIR, 'Correlation_Matrices')
CKPT_DIR = os.path.join(EVAL_DIR, 'checkpoints')

# To allow imports from project modules
sys.path.append(BASE)
sys.path.append(TRAIN_DIR)

from variable_versions import load_variables
from models import ClassificationModel
from prepdata_v0 import ClassificationDataset, prepdata

# ---------------- Configuration ----------------
version = 11
loss_type = "binary"        # use checkpoint 'B'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corr_threshold = 0.80       # threshold for correlation-based grouping
cumulative_threshold = 0.95 # coverage threshold for cumulative importance
batch_size = 1024           # batch size for validation loader

# SHAP parameters (KernelExplainer)
n_background = 50           # number of background samples
n_explain = 200             # number of samples to explain
nsamples_kernel = 100       # approximation parameter for KernelExplainer

# ------------- Load correlation matrices -------------
sig_csv = os.path.join(CORR_DIR, f"Signal_CorrelationMatrix_v{version}.csv")
bkg_csv = os.path.join(CORR_DIR, f"Background_CorrelationMatrix_v{version}.csv")

if not (os.path.isfile(sig_csv) and os.path.isfile(bkg_csv)):
    raise FileNotFoundError(f"Correlation CSV files for v{version} are missing. Please generate them with correlation_matrix.py.")

corr_sig = pd.read_csv(sig_csv, index_col=0)
corr_bkg = pd.read_csv(bkg_csv, index_col=0)

# ------------- Load variable list (consistent order) -------------
variable_list = load_variables(version, config_path=os.path.join(BASE, 'variable_versions.json'))

# Reorder correlation matrices to match the official variable list
corr_sig = corr_sig.reindex(index=variable_list, columns=variable_list)
corr_bkg = corr_bkg.reindex(index=variable_list, columns=variable_list)

# Combined correlation matrix: maximum absolute correlation between Signal and Background
abs_max_corr = pd.DataFrame(
    np.maximum(np.abs(corr_sig.values), np.abs(corr_bkg.values)),
    index=variable_list, columns=variable_list
)

# ------------- Correlation-based clustering (connected components) -------------
visited = set()
groups = []  # list of lists with variable names

for v in variable_list:
    if v in visited:
        continue
    stack = [v]
    comp = []
    visited.add(v)
    while stack:
        cur = stack.pop()
        comp.append(cur)
        # neighbors: all variables with |rho(cur, j)| >= threshold
        neighbors = abs_max_corr.loc[cur]
        neighbors = neighbors[(neighbors.index != cur) & (neighbors >= corr_threshold)].index.tolist()
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    groups.append(sorted(comp))

# ------------- Load model and data (no retraining) -------------
input_size = len(variable_list)
suffix = "B" if loss_type == "binary" else "F"
ckpt_path = os.path.join(CKPT_DIR, f"{suffix}_model_checkpoint_v{version}.pth")

model = ClassificationModel(input_size=input_size).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state)
model.eval()

# Load ROOT data into tensors
dir_path = os.path.abspath(os.path.join(BASE, '..', 'Signal_vs_Background', 'ROOT_files'))
X, y = prepdata(dir_path, "signal.root", "background.root", version)

# Validation split
_, X_val, _, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
val_ds = ClassificationDataset(X_val, y_val)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Merge all validation data into a single tensor (for SHAP)
X_all = []
for xb, _ in val_loader:
    X_all.append(xb)
X_all = torch.cat(X_all, dim=0).to(device)

# Limit background and explained samples for SHAP (to reduce computational cost)
bg = X_all[: min(n_background, X_all.shape[0])].detach().cpu().numpy()
X_exp = X_all[: min(n_explain, X_all.shape[0])].detach().cpu().numpy()

# Wrapper for the model (1D output)
def model_fn(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(x_tensor)
    preds = preds.squeeze(-1).detach().cpu().numpy()
    return preds

# SHAP KernelExplainer (model-agnostic, but slower; reduced sample sizes)
explainer = shap.KernelExplainer(model_fn, bg)
shap_values = explainer.shap_values(X_exp, nsamples=nsamples_kernel)

# For binary classification with 1D output, shap_values has shape (n_samples, n_features)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feat_importance = pd.Series(mean_abs_shap, index=variable_list).sort_values(ascending=False)

# ------------- Choose one representative per group -------------
representatives = []
for grp in groups:
    if len(grp) == 1:
        representatives.append(grp[0])
    else:
        # choose the feature with highest |mean SHAP| within the group
        best = max(grp, key=lambda v: feat_importance.get(v, 0.0))
        representatives.append(best)

# Importance restricted to representatives
rep_importance = feat_importance.loc[representatives].sort_values(ascending=False)

# ------------- Cumulative importance until threshold -------------
total = rep_importance.sum()
fractions = rep_importance / (total if total > 0 else 1.0)
cumulative = fractions.cumsum()

# minimum k such that cumulative >= cumulative_threshold
k = np.searchsorted(cumulative.values, cumulative_threshold, side="left")
k = int(np.clip(k, 0, len(representatives)-1))
selected_reps = list(cumulative.index[:k+1])

# ------------- Save outputs -------------
OUT_DIR = os.path.join(EVAL_DIR, f'Cumulative_SHAP_Grouped_v{version}')
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "groups.json"), "w") as f:
    json.dump({"groups": groups, "representatives": selected_reps}, f, indent=2)

rep_importance.to_csv(os.path.join(OUT_DIR, "representatives_importance.csv"))
fractions.to_csv(os.path.join(OUT_DIR, "representatives_importance_fractions.csv"))
cumulative.to_csv(os.path.join(OUT_DIR, "representatives_cumulative.csv"))

# Plot cumulative importance (representatives only)
plt.figure(figsize=(14, 6))
plt.plot(range(len(cumulative)), cumulative.values, marker='o', linestyle='-')
plt.xticks(range(len(cumulative)), cumulative.index, rotation=90)
plt.axhline(y=cumulative_threshold, linestyle='--', label=f"Threshold {int(cumulative_threshold*100)}%")
plt.axvline(x=k + 0.5, linestyle='--', label=f"{k+1} reps >= {int(cumulative_threshold*100)}%")
plt.ylabel("Cumulative SHAP importance (representatives)")
plt.xlabel("Features (group representatives)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cumulative_representatives.png"), dpi=200)
plt.close()

# Log final results
print(f"[OK] Groups formed (|rho|>={corr_threshold}), number of groups: {len(groups)}")
print(f"[OK] Selected representatives covering >= {int(cumulative_threshold*100)}%: {k+1}")
print(selected_reps)

