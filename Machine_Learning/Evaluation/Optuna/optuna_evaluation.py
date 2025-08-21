"""
Module to evaluate ML training for optuna optimised model.
14/08/2025
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from fpdf import FPDF
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sns.set_style("darkgrid")

# Add the directory containing variables
sys.path.append(os.path.abspath("Machine_Learning"))
from variable_versions import load_variables

# Add the directory containing NeuralNetwork.py to the Python path
sys.path.append(os.path.abspath("Machine_Learning/Training"))
from optunaNN import DynamicClassificationModel
from models import ClassificationModel
from prepdata_v0 import prepdata, ClassificationDataset

def load_model_save_params(version, out_dir="."):

    # load checkpoint
    checkpoint_path = f"Machine_Learning/Evaluation/checkpoints_optim/best_model_v{version}.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    full_dataset = checkpoint["dataset"]
    test_dataset = checkpoint["test_set"]
    test_dataset.y = test_dataset.y.long()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    hyperparams = checkpoint["trial_params"]
    neurons = [hyperparams[f"neurons_l{i}"] for i in range(hyperparams["n_layers"])]
    input_size = full_dataset.X.shape[1]

    model = DynamicClassificationModel(
        input_size,
        hyperparams["n_layers"],
        neurons,
        hyperparams["dropout_rate"]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Save hyperparameters to PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"\n Balanced model:", ln=True)
    pdf.cell(200, 10, txt=f'Learning rate -> {hyperparams["learning_rate"]}', ln=True)
    pdf.cell(200, 10, txt=f'Number of layers -> {hyperparams["n_layers"]}', ln=True)
    pdf.cell(200, 10, txt=f'Dropout rate -> {hyperparams["dropout_rate"]}', ln=True)
    pdf.cell(200, 10, txt=f'Weight decay -> {hyperparams["weight_decay"]}', ln=True)
    pdf.cell(200, 10, txt=f'Batch size -> {hyperparams["batch_size"]}', ln=True)
    pdf.cell(200, 10, txt=f'Number of layers -> {hyperparams["n_layers"]}', ln=True)
    for i, n in enumerate(neurons):
        pdf.cell(200, 10, txt=f'Neurons in layer {i} -> {n}', ln=True)

    pdf_path = os.path.join(out_dir, "Hyperparameters.pdf")
    pdf.output(pdf_path)
    print(f"Hyperparameters PDF saved to {pdf_path}")

    
    return model, test_loader

def load_model2(original_version):
    """ for evaluation of the optuna feature selection training."""

    # load checkpoint
    checkpoint_path = f"Machine_Learning/Evaluation/checkpoints_optim/best_features_v{original_version}.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    x_test = checkpoint["X_test"]
    y_test = checkpoint["y_test"]
    selected_features = checkpoint["selected_features"]

    # Get indices of selected features
    FEATURE_NAMES = load_variables(original_version)  # or your original variable list
    selected_indices = [i for i, f in enumerate(FEATURE_NAMES) if f in selected_features]

    # Filter datasets to only selected features
    x_test  = x_test[:, selected_indices]

    # Update input_size
    input_size = len(selected_indices)

    test_set = ClassificationDataset(x_test, y_test)

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = DynamicClassificationModel(
        input_size, 2, [32, 16], 0.1)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, test_loader




# --- Get targets and probabilities ---
def get_targets_probabilities(model, test_loader):

    targets, probabilities = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return np.array(targets), np.array(probabilities)


# --- Plot probability histogram ---
def plot_histogram(targets, probabilities, out_dir="."):

    plt.figure(figsize=(8, 6))

    # Signal predictions
    signal_predict = probabilities[targets == 1]
    plt.hist(signal_predict, bins=40, density=True, alpha=0.9, label="Signal (MC)", color="blue", range=(0.0, 1.0))

    # Background predictions
    background_predict = probabilities[targets == 0]
    plt.hist(background_predict, bins=40, density=True, alpha=0.5, label="Background (Data)", color="red", hatch="//", edgecolor="black", range=(0.0, 1.0))
    
    #plt.axvline(x=best_thresh, color='grey', lw=2, label=f'Threshold = {best_thresh:.2f}')
    plt.xlabel("Predicted Probability", fontsize=14, labelpad=15)
    plt.ylabel("Normalized Density", fontsize=14, labelpad=15) 
    plt.legend()
    save_path = os.path.join(out_dir, "prob_distribution.pdf")
    plt.savefig(save_path)  # Save the plot as a PDF file
    plt.close()

# --- Plot ROC curve ---
def plot_roc_curve(targets, probabilities, out_dir="."):

    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    save_path = os.path.join(out_dir, "roc_curve.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

# --- Save metrics to PDF ---
def save_metrics_pdf(targets, probabilities, out_dir=".",  best_thr=0.5):
    
    # For binary classification, using 0.5 threshold
    pred_labels = (probabilities >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(targets, pred_labels)
    prec = precision_score(targets, pred_labels)
    rec = recall_score(targets, pred_labels)
    f1 = f1_score(targets, pred_labels)
    conf_matrix = confusion_matrix(targets, pred_labels)
    
    pdf_filename = os.path.join(out_dir, f"Metrics.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 50, f"Evaluation Metrics")

    # Metrics text
    c.setFont("Helvetica", 12)
    y_pos = height - 100
    for metric_name, value in [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1-score", f1), ("Best Threshold", best_thr)]:
        c.drawString(100, y_pos, f"{metric_name}: {value:.4f}")
        y_pos -= 20

    # Confusion Matrix Table
    c.drawString(100, y_pos - 20, "Confusion Matrix:")
    matrix_top = y_pos - 80
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

    # Labels
    c.setFont("Helvetica-Bold", 12)
    c.drawString(matrix_left + cell_width / 2 - 20, matrix_top + 50, "True: 1")
    c.drawString(matrix_left + 3 * cell_width / 2 - 20, matrix_top + 50, "True: 0")
    c.drawString(matrix_left - 60, matrix_top - cell_height / 2 + 35, "Pred: 1")
    c.drawString(matrix_left - 60, matrix_top - 3 * cell_height / 2 + 35, "Pred: 0")

    c.save()
    print(f"Metrics PDF saved to {pdf_filename}")


def plot_loss_curve(version, out_dir="."):

    # load checkpoint
    #checkpoint_path = f"Machine_Learning/Evaluation/checkpoints_optim/best_model_v{version}.pth"
    checkpoint_path = f"Machine_Learning/Evaluation/checkpoints_optim/best_features_v11.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    train_loss_curve = checkpoint["train_loss_curve"]
    val_loss_curve = checkpoint["val_loss_curve"]
    best_epoch = checkpoint["best_epoch"]

    indices = range(1, len(train_loss_curve) + 1)
    plt.figure()
    plt.plot(indices, train_loss_curve, label="Training", color="navy", markersize=1)
    plt.plot(indices, val_loss_curve, label="Validation", color="orange", markersize=1)
    plt.scatter(best_epoch + 1, val_loss_curve[best_epoch], color="black", label="Early Stop", s=64)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"loss_curve_v{version}.pdf"))
    plt.close()


def plot_roc_comparison(version, out_dir="."):

    # get checkpoint for model before optimisation
    path = f"Machine_Learning/Evaluation/checkpoints/B_model_checkpoint_v{version}.pth"
    checkpoint = torch.load(path, weights_only=False)

    full_dataset = checkpoint["dataset"]
    test_dataset = checkpoint["test_set"]
    test_dataset.y = test_dataset.y.long()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = full_dataset.X.shape[1]
    model = ClassificationModel(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # get roc data before optimisation
    targets, probabilities = get_targets_probabilities(model, test_loader)
    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    # get model after optimisation
    path_optim = f"Machine_Learning/Evaluation/checkpoints_optim/best_model_v{version}.pth"
    checkpoint_optim = torch.load(path_optim, weights_only=False)

    full_dataset_optim = checkpoint_optim["dataset"]
    test_dataset_optim = checkpoint_optim["test_set"]
    test_dataset_optim.y = test_dataset_optim.y.long()

    test_loader_optim = DataLoader(test_dataset_optim, batch_size=32, shuffle=False)

    hyperparams = checkpoint_optim["trial_params"]
    neurons = [hyperparams[f"neurons_l{i}"] for i in range(hyperparams["n_layers"])]
    input_size_optim = full_dataset_optim.X.shape[1]

    model_optim = DynamicClassificationModel(
        input_size_optim,
        hyperparams["n_layers"],
        neurons,
        hyperparams["dropout_rate"]
    )
    model_optim.load_state_dict(checkpoint_optim['model_state_dict'])
    model_optim.eval()

    # get roc data after optimisation
    targets_optim, probabilities_optim = get_targets_probabilities(model_optim, test_loader_optim)
    fpr_optim, tpr_optim, _ = roc_curve(targets_optim, probabilities_optim)
    roc_auc_optim = auc(fpr_optim, tpr_optim)

    # Plot ROC comparison
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot(fpr_optim, tpr_optim, color='darkorange', lw=2,
             label=f'Optim ROC curve (AUC = {roc_auc_optim:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    save_path = os.path.join(out_dir, "roc_comparison.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve comparison saved to {save_path}")
    





if __name__ == "__main__":

    version = 14
    #original_version = 11
    output_dir = os.path.join("Machine_Learning/Evaluation/Optuna", f"v{version}")
    os.makedirs(output_dir, exist_ok=True)

    model, test_loader = load_model_save_params(version, output_dir)
    #model, test_loader = load_model2(original_version)
    
    targets, probabilities = get_targets_probabilities(model, test_loader)
    plot_histogram(targets, probabilities, output_dir)
    plot_roc_curve(targets, probabilities, output_dir)
    save_metrics_pdf(targets, probabilities, output_dir)
    plot_loss_curve(version, output_dir)
    plot_roc_comparison(version, output_dir)
    