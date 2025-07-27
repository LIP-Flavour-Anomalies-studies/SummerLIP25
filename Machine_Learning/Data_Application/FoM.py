"""
Module to determine best threshold based on FoM.
Teresa 27/07/2025
"""

import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


sys.path.append(os.path.abspath("Machine_Learning/Training"))
from models import ClassificationModel

sns.set_style("darkgrid")

def load_model(loss_type, version):

    if loss_type == "focal":
        checkpoint_path = f"Machine_Learning/Evaluation/checkpoints/F_model_checkpoint_v{version}.pth"
    elif loss_type == "binary":
        checkpoint_path = f"Machine_Learning/Evaluation/checkpoints/B_model_checkpoint_v{version}.pth"
    else:
        raise ValueError("Invalid loss type. Use 'focal' or 'binary'.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    full_dataset = checkpoint["dataset"]
    test_dataset = checkpoint["test_set"]
    test_dataset.y = test_dataset.y.long()

    input_size = full_dataset.X.shape[1]
    model = ClassificationModel(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return model, test_loader


def get_weights(loss_type, version):
    with open(f"Machine_Learning/Data_Application/weights/weights_{loss_type}.json", "r") as f:
        return json.load(f)[str(version)]


def calculate_fom(probabilities, targets, version, loss_type=None, version_dir=None):
    """
    Calculate the ROC curve points and find the best threshold maximizing the figure of merit (FOM).

    Args:
        probabilities (torch.Tensor): Predicted probabilities for the positive class.
        targets (torch.Tensor): Ground truth labels (0 or 1).

    Returns:
        tuple: (fpr_list, tpr_list, auc, best_threshold, best_point)
            - fpr_list (np.ndarray): False Positive Rates at each threshold.
            - tpr_list (np.ndarray): True Positive Rates at each threshold.
            - auc (float): Area Under the ROC Curve.
            - best_threshold (float): Threshold maximizing the FOM.
            - best_point (tuple): (FPR, TPR) corresponding to best_threshold.
    """
    # Load weights
    weights = get_weights(loss_type, version)
    w_S = weights["w_S"]
    w_B = weights["w_B"]

    thresholds = np.linspace(0.0, 1.0, 500)
    best_thr = 0.5
    best_fom = -np.inf
    best_point = None
    tpr_list = []
    fpr_list = []
    fom_values = []

    total_signal = ((targets == 1) * w_S).sum().item()
    total_background = ((targets == 0) * w_B).sum().item()

    if total_signal == 0 or total_background == 0:
        raise ValueError("Targets must contain both signal (1) and background (0) examples.")

    for thr in thresholds:
        predicted = (probabilities >= thr).float()
        tp = (((predicted == 1) & (targets == 1)).float()).sum().item() * w_S
        fp = (((predicted == 1) & (targets == 0)).float()).sum().item() * w_B

        fom = tp / np.sqrt(tp + fp) if (tp + fp) > 0 else 0
        tpr = tp / total_signal
        fpr = fp / total_background

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        fom_values.append(fom)

        if fom > best_fom:
            best_fom = fom
            best_thr = thr
            best_point = (fpr, tpr)

        # Plot Youden's J vs Threshold
    if loss_type is not None and version_dir is not None:
        plt.figure()
        plt.plot(thresholds, fom_values, label="FoM")
        plt.axvline(x=best_thr, color='black', linestyle='--', label=f"Best Threshold = {best_thr:.4f}")
        plt.xlabel("Threshold")
        plt.ylabel("FoM values")
        plt.legend()
        save_path = os.path.join(version_dir, f"{loss_type[0].upper()}_FoM_vs_thresholds.pdf")
        plt.savefig(save_path)
        plt.close()

    auc = roc_auc_score(targets.numpy(), probabilities.numpy())

    return np.array(fpr_list), np.array(tpr_list), auc, best_thr, best_point


def update_threshold_file_2(loss_type, version, best_thr, save_dir):

    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    filename = os.path.join(save_dir, f"FoM_{loss_type}_thresholds.json")
    # Load existing thresholds if the file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {}

    # Initialize group if not present
    if loss_type not in thresholds:
        thresholds[loss_type] = {}

    # Update or add the version's threshold
    thresholds[loss_type][str(version)] = round(best_thr.item(), 3)

    # Write back to file
    with open(filename, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"[{loss_type}] Threshold for version {version} saved to {filename}")


def evaluate_model_2(model, test_loader, loss_type, version=None, version_dir=None):
    """
    Evaluate the classification model on a test set.

    Args:
        model (torch.nn.Module): Trained PyTorch classification model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        tuple: (probabilities, targets, best_threshold)
            - probabilities (torch.Tensor): Predicted probabilities for the positive class.
            - targets (torch.Tensor): Ground truth labels.
            - best_threshold (float): Threshold maximizing the figure of merit.
    """
    model.eval()
    predictions = []
    targets = []
    probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    targets = torch.tensor(targets)
    probabilities = torch.tensor(probabilities)

    fpr, tpr, auc, best_thr, best_point = calculate_fom(probabilities, targets, version, loss_type, version_dir)
    predictions = (probabilities >= best_thr).float()

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.scatter(best_point[0], best_point[1], color="black", label="Best Threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_path = os.path.join(version_dir, f"{loss_type[0].upper()}_roc_curve.pdf")
    plt.savefig(save_path)
    plt.close()

    if version is not None:
        update_threshold_file_2(loss_type, version, best_thr, save_dir="Machine_Learning/Data_Application/thresholds")

    return probabilities, targets, best_thr, fpr, tpr, auc, best_point

def plot_histogram(model, data_loader, labels, best_thr, loss_type, version_dir):

    model.eval()
    prob = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs).squeeze()
            prob.extend(outputs.cpu().numpy())

    prob = np.array(prob)
    targets = np.array(labels)

    plt.figure(figsize=(8, 6))

    signal_predict = prob[targets == 1]
    background_predict = prob[targets == 0]

    plt.hist(signal_predict, bins=40, density=True, alpha=0.9, label="Signal", color="blue", range=(0.0, 1.0))
    plt.hist(background_predict, bins=40, density=True, alpha=0.5, label="Background", color="red", hatch="//", edgecolor="black", range=(0.0, 1.0))

    plt.axvline(x=best_thr, color="black", lw=2, linestyle="--", label=f"Threshold = {best_thr:.2f}")
    plt.xlabel("Predicted Probability", fontsize=14, labelpad=15)
    plt.ylabel("Normalized Density", fontsize=14, labelpad=15)
    plt.legend()
    save_path = os.path.join(version_dir, f"{loss_type[0].upper()}_prob_distribution.pdf")
    plt.savefig(save_path)
    plt.close()


def plot_combined_roc(roc_data, version_dir):
    plt.figure()
    for loss_type, (fpr, tpr, auc, best_point) in roc_data.items():
        label = f"{loss_type.capitalize()} Loss (AUC = {auc:.3f})"
        plt.plot(fpr, tpr, label=label)
        plt.scatter(best_point[0], best_point[1], label=f"{loss_type.capitalize()} Best Threshold", marker='o')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_path = os.path.join(version_dir, f"combined_roc_curve.pdf")
    plt.savefig(save_path)
    plt.close()
    print("Combined ROC curve saved to combined_roc_curve.pdf") 


def main():
    roc_data = {}
    for loss_type in ["focal", "binary"]:
        try:
            # Choose version
            version = 1
            # Save evaluations files
            version_dir = None
            if version is not None:
                version_dir = os.path.join("Machine_Learning/Data_Application", f"v{version}")
                os.makedirs(version_dir, exist_ok=True)
            
            model, test_loader = load_model(loss_type, version)

            # Evaluate on the test set
            probabilities, targets, best_thr, fpr, tpr, auc, best_point = evaluate_model_2(model, test_loader, loss_type, version, version_dir)
            roc_data[loss_type] = (fpr, tpr, auc, best_point)

            # Plot the histograms of predicted probabilities
            plot_histogram(model, test_loader, targets, best_thr, loss_type, version_dir)
            
        except Exception as e:
            print(f"[{loss_type.upper()}] An error occurred: {e}")
    
    # Plot merged ROC after both evaluations
    plot_combined_roc(roc_data, version_dir)


if __name__ == '__main__':
    main()





