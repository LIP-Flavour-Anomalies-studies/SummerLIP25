"""

Teresa 12/07/2025
"""
import torch
import matplotlib.pyplot as plt

def regul(val_loader, model, criterion, epoch, num_epochs, early_stopping):
    model.eval()
    val_loss = 0.0

    # Compute validation loss for 1 epoch
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs).squeeze()
            loss = criterion(val_outputs, val_targets)
            val_loss += loss.item() * val_inputs.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Check if validation loss has reached its minimum
    early_stopping(val_loss, model)

    return val_loss

def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, num_epochs=1000, flag=0):
    """
    Train the model and plot training vs validation loss.
    """
    stop = 0
    tl_vector, vl_vector = [], []
    idx = num_epochs - 1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze() # Adjust outputs to match the shape of targets
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss = regul(val_loader, model, criterion, epoch, num_epochs, early_stopping)

        # Save training and validation loss in vectors
        tl_vector.append(train_loss)
        vl_vector.append(val_loss)

        # Save best epoch number
        if early_stopping.early_stop and stop == 0:
            idx = epoch - early_stopping.patience
            print(f"Early stopping at epoch {idx}\n Lowest loss: {-early_stopping.best_score}")
            stop = 1

    # Load the best model
    early_stopping.load_best_model(model)

    indices = range(1, num_epochs + 1) 

    # Plot training and validation loss 
    plt.figure()
    plt.plot(indices, tl_vector, label="Training", color="navy", markersize=1)
    plt.plot(indices, vl_vector, label="Validation", color="orange", markersize=1)
    plt.scatter(idx + 1, vl_vector[idx], color="black", label="Early Stop", s=64)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.ylim(0, max(max(tl_vector), max(vl_vector)) / (1.5 if flag == 0 else 4))
    plt.savefig("B_loss_v0.png" if flag == 0 else "F_loss_v0.png")
    plt.close()