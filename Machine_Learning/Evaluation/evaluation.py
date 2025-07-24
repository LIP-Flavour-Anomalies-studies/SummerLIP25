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

from torch.utils.data import DataLoader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from sklearn.metrics import roc_auc_score

sns.set_style("darkgrid")

# Add the directory containing NeuralNetwork.py to the Python path
sys.path.append(os.path.abspath("/user/u/u25teresaesc/Internship/Machine_Learning/Training"))
from models import ClassificationModel

def load_model(loss_type):

    if loss_type == "focal":
        checkpoint_path = "/user/u/u25teresaesc/Internship/Machine_Learning/Evaluation/checkpoints/F_model_checkpoint_v1.pth"
    elif loss_type == "binary":
        checkpoint_path = "/user/u/u25teresaesc/Internship/Machine_Learning/Evaluation/checkpoints/B_model_checkpoint_v1.pth"
    else:
        raise ValueError("Invalid loss type. Use 'focal' or 'binary'.")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    full_dataset = checkpoint["dataset"]
    test_dataset = checkpoint["test_set"]
    test_dataset.y = test_dataset.y.long()

    print(f"[{loss_type.upper()}] Test set class distribution:", torch.unique(test_dataset.y, return_counts=True))

    input_size = full_dataset.X.shape[1]
    model = ClassificationModel(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return model, test_loader


def calculate_metrics(predictions, targets):

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1, np.array([[tp, fp], [fn, tn]])


def save_metrics(conf_matrix, accuracy, precision, recall, f1, best_thr, loss_type):

    pdf_filename = f"{loss_type[0].upper()}_metrics.pdf"
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


def calculate_fom_roc(probabilities, targets):
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
    thresholds = np.linspace(0.0, 1.0, 500)
    best_thr = 0.5
    best_fom = -np.inf
    best_point = None
    tpr_list = []
    fpr_list = []

    total_signal = (targets == 1).sum().item()
    total_background = (targets == 0).sum().item()

    if total_signal == 0 or total_background == 0:
        raise ValueError("Targets must contain both signal (1) and background (0) examples.")

    for thr in thresholds:
        predicted = (probabilities >= thr).float()
        tp = ((predicted == 1) & (targets == 1)).sum().item()
        fp = ((predicted == 1) & (targets == 0)).sum().item()

        fom = tp / np.sqrt(tp + fp) if (tp + fp) > 0 else 0
        tpr = tp / total_signal
        fpr = fp / total_background

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        if fom > best_fom:
            best_fom = fom
            best_thr = thr
            best_point = (fpr, tpr)

    auc = roc_auc_score(targets.numpy(), probabilities.numpy())

    return np.array(fpr_list), np.array(tpr_list), auc, best_thr, best_point


def evaluate_model(model, test_loader, loss_type):
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

    fpr, tpr, auc, best_thr, best_point = calculate_fom_roc(probabilities, targets)
    predictions = (probabilities >= best_thr).float()

    accuracy, precision, recall, f1, conf_matrix = calculate_metrics(predictions, targets)

    print(f"\n[{loss_type.upper()}] Accuracy: {accuracy:.4f}")
    print(f"[{loss_type.upper()}] Precision: {precision:.4f}")
    print(f"[{loss_type.upper()}] Recall: {recall:.4f}")
    print(f"[{loss_type.upper()}] F-score: {f1:.4f}")
    print(f"[{loss_type.upper()}] Best threshold: {best_thr:.4f}")
    print(f"\n[{loss_type.upper()}] Confusion Matrix:")
    print(conf_matrix)

    save_metrics(conf_matrix, accuracy, precision, recall, f1, best_thr, loss_type)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.scatter(best_point[0], best_point[1], color="black", label="Best Threshold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"{loss_type[0].upper()}_roc_curve.pdf")
    plt.close()

    print(f"[{loss_type.upper()}] Number of signal events (label=1):", (targets == 1).sum().item())
    print(f"[{loss_type.upper()}] Number of background events (label=0):", (targets == 0).sum().item())

    return probabilities, targets, best_thr


def plot_histogram(model, data_loader, labels, best_thr, loss_type):
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
    plt.savefig(f"{loss_type[0].upper()}_prob_distribution.pdf")
    plt.close()


def main():
    for loss_type in ["focal", "binary"]:
        try:
            model, test_loader = load_model(loss_type)
            probabilities, targets, best_thr = evaluate_model(model, test_loader, loss_type)
            plot_histogram(model, test_loader, targets, best_thr, loss_type)
        except Exception as e:
            print(f"[{loss_type.upper()}] An error occurred: {e}")


if __name__ == '__main__':
    main()
