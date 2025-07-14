# Machine Learning Module for ROOT Data Classification

## Overview

This folder contains a PyTorch-based machine learning pipeline designed for classification tasks using ROOT files. It supports reading multiple ROOT trees from the same file, preprocessing data, defining neural network models, custom loss functions, early stopping, and training loops.

Code was heavily based on the code made by Diogo Pereira and Gonçalo Marujo during the Flavour Anomalies LIP Internship Program 2024.

---

## Directory Structure

- **main.py**  
  Entry point of the training pipeline. Loads data, prepares datasets, initializes models, losses, optimizers, and trains the models with early stopping.

- **models.py**  
  Defines the neural network architecture used for binary classification.

- **losses.py**  
  Contains custom loss functions: Balanced Cross-Entropy Loss and Focal Loss, useful for imbalanced datasets.

- **early_stopping.py**  
  Implements an EarlyStopping utility that monitors validation loss and stops training when no improvement is detected.

- **train.py**  
  Contains training and validation loops, loss tracking, plotting of loss curves, and supports early stopping.

- **prepdata_v0.py**  
  Data loading and preprocessing utilities that handle ROOT file input and prepare PyTorch datasets.

---

## How to Use

1. Place ROOT files with multiple trees inside the appropriate data folder.
2. Modify paths in `main.py` to point to ROOT files.
3. Run `main.py` to start training:
