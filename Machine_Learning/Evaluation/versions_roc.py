"""
Plots roc curves for different versions overlayed.
"""
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sns.set_style("darkgrid")

sys.path.append(os.path.abspath("Machine_Learning/Evaluation"))
from evaluation import calculate_roc_auc_thr
from evaluation import load_model


roc_data = {}

for loss_type in ["binary"]:

    for version in [0, 1, 2, 3, 4, 5]:

        model, test_loader = load_model(loss_type, version)

        model.eval()
        targets = []
        probabilities = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                probabilities.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        targets = torch.tensor(targets)
        probabilities = torch.tensor(probabilities)

        fpr, tpr, auc, best_thr, best_point = calculate_roc_auc_thr(targets, probabilities, loss_type)

        roc_data[version] = (fpr, tpr, auc, best_point)

    plt.figure()
    for version, (fpr, tpr, auc, best_point) in roc_data.items():
        label = f"v{version} {loss_type} (AUC = {auc:.3f})"
        plt.plot(fpr, tpr, label=label)
        plt.scatter(best_point[0], best_point[1], label=f"v{version} Best Threshold", marker='o')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"{loss_type}_versions_roc.png")
    plt.close()
