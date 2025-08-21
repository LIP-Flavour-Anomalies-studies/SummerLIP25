"""
Module to evaluate ML training for both focal loss and binary cross-entropy.
23/07/2025
"""
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json

from torch.utils.data import DataLoader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from sklearn.metrics import roc_auc_score

sns.set_style("darkgrid")

# Add the directory containing NeuralNetwork.py to the Python path
sys.path.append(os.path.abspath("Machine_Learning/Training"))
from models import ClassificationModel

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


def calculate_metrics(predictions, targets):

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    conf_matrix = np.array([[tp, fp], [fn, tn]])

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / (total)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, conf_matrix


def save_metrics(conf_matrix, accuracy, precision, recall, f1, best_thr, loss_type, version_dir):

    pdf_filename = os.path.join(version_dir, f"{loss_type[0].upper()}_metrics.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 50, f"{loss_type.capitalize()} Loss - Evaluation Metrics")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, f"Accuracy: {accuracy:.4f}")
    c.drawString(100, height - 120, f"Precision: {precision:.4f}")
    c.drawString(100, height - 140, f"Recall: {recall:.4f}")
    c.drawString(100, height - 160, f"F-score: {f1:.4f}")
    c.drawString(100, height - 180, f"Best Threshold: {best_thr:.4f}")

    c.drawString(100, height - 220, "Confusion Matrix:")
    matrix_top = height - 280
    matrix_left = 220
    cell_width = 100
    cell_height = 40
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)

    for i in range(2):
        for j in range(2):
            x = matrix_left + j * cell_width
            y = matrix_top - i * cell_height
            c.rect(x, y, cell_width, cell_height)
            c.drawCentredString(x + cell_width / 2, y + cell_height / 2 - 6, str(conf_matrix[i, j]))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(matrix_left + cell_width / 2 - 20, matrix_top + 50, "True: 1")
    c.drawString(matrix_left + 3 * cell_width / 2 - 20, matrix_top + 50, "True: 0")
    c.drawString(matrix_left - 60, matrix_top - cell_height / 2 + 35, "Pred: 1")
    c.drawString(matrix_left - 60, matrix_top - 3 * cell_height / 2 + 35, "Pred: 0")

    c.save()
    print(f"[{loss_type.upper()}] Evaluation metrics saved to {pdf_filename}")


def calculate_roc_auc_thr(targets, probabilities, loss_type=None, version_dir=None):

    thresholds = sorted(set(probabilities), reverse=True)
    tpr = []  # True positive rate
    fpr = []  # False positive rate
    J_values = []
    best_thr = 0.5
    best_J = -1
    best_point = None

    for thr in thresholds:
        predicted = (probabilities >= thr).float()
        _, _, _, _, conf_matrix = calculate_metrics(predicted, targets)
        
        tp = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        tn = conf_matrix[1, 1]
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Calculate Youden's J statistic
        J = sensitivity + specificity - 1
        
        tpr.append(sensitivity)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        J_values.append(J)
        
        if J > best_J:
            best_J = J
            best_thr = thr
            best_point = (fpr[-1], tpr[-1])

    # Plot Youden's J vs Threshold
    if loss_type is not None and version_dir is not None:
        plt.figure()
        plt.plot(thresholds, J_values, label="Youden's J", color='darkorange')
        plt.axvline(x=best_thr, color='black', linestyle='--', label=f"Best Threshold = {best_thr:.4f}")
        plt.xlabel("Threshold")
        plt.ylabel("Youden's J Statistic")
        plt.legend()
        save_path = os.path.join(version_dir, f"{loss_type[0].upper()}_youden_J_vs_threshold.pdf")
        plt.savefig(save_path)
        plt.close()

    # Final ROC/AUC computation
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    auc = roc_auc_score(targets.numpy(), probabilities.numpy()) 

    return fpr, tpr, auc, best_thr, best_point


def update_threshold_file(loss_type, version, best_thr, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    filename = os.path.join(save_dir, f"{loss_type}_thresholds.json")
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


def evaluate_model(model, test_loader, loss_type, version=None, version_dir=None):
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

    fpr, tpr, auc, best_thr, best_point = calculate_roc_auc_thr(targets, probabilities, loss_type, version_dir)
    predictions = (probabilities >= best_thr).float()

    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(predictions, targets)

    print(f"\n[{loss_type.upper()}] Accuracy: {accuracy:.4f}")
    print(f"[{loss_type.upper()}] Precision: {precision:.4f}")
    print(f"[{loss_type.upper()}] Recall: {recall:.4f}")
    print(f"[{loss_type.upper()}] F-score: {f1:.4f}")
    print(f"[{loss_type.upper()}] Best threshold: {best_thr:.4f}")
    print(f"\n[{loss_type.upper()}] Confusion Matrix:")
    print(conf_matrix)

    save_metrics(conf_matrix, accuracy, precision, recall, f1, best_thr, loss_type, version_dir)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.scatter(best_point[0], best_point[1], color="black", label="Best Threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_path = os.path.join(version_dir, f"{loss_type[0].upper()}_roc_curve.pdf")
    plt.savefig(save_path)
    plt.close()

    print(f"[{loss_type.upper()}] Number of signal events (label=1):", (targets == 1).sum().item())
    print(f"[{loss_type.upper()}] Number of background events (label=0):", (targets == 0).sum().item())

    if version is not None:
        update_threshold_file(loss_type, version, best_thr, save_dir="Machine_Learning/Data_Application/thresholds")

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

    plt.hist(signal_predict, bins=40, density=True, alpha=0.9, label="Signal (MC)", color="blue", range=(0.0, 1.0))
    plt.hist(background_predict, bins=40, density=True, alpha=0.5, label="Background (Data)", color="red", hatch="//", edgecolor="black", range=(0.0, 1.0))

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
    for loss_type in ["binary"]:
        try:
            # Choose version
            version = 14
            # Save evaluations files
            version_dir = None
            if version is not None:
                version_dir = os.path.join("Machine_Learning/Evaluation", f"v{version}")
                os.makedirs(version_dir, exist_ok=True)
            
            model, test_loader = load_model(loss_type, version)

             # Evaluate on the test set
            probabilities, targets, best_thr, fpr, tpr, auc, best_point = evaluate_model(model, test_loader, loss_type, version, version_dir)
            roc_data[loss_type] = (fpr, tpr, auc, best_point)

            # Plot the histograms of predicted probabilities
            plot_histogram(model, test_loader, targets, best_thr, loss_type, version_dir)
        except Exception as e:
            print(f"[{loss_type.upper()}] An error occurred: {e}")

    # Plot merged ROC after both evaluations
    #plot_combined_roc(roc_data, version_dir)


if __name__ == '__main__':
    main()
