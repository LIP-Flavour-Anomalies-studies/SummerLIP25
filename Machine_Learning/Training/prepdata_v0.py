"""
Data loading code to prep for ML.
Teresa 12/07/2025
"""
import sys
import os
import uproot
import numpy as np
import torch
from torch.utils.data import Dataset

# Add the directory containing variables
sys.path.append(os.path.abspath("Machine_Learning"))
from variable_versions import load_variables

def prepdata(dir_path, root_mc, root_data, version):
    """
    Load ROOT file, extract features and labels from two trees (signal and background),
    and prepare numpy arrays for training.

    Parameters:
        dir_path : str
            Directory path containing ROOT file.
        root_mc : str
            Filename of the ROOT file containing signal from MC.
        root_signal : str
            Filename of the ROOT file containing background from data.

    Returns:
        X : numpy.ndarray
            Combined features from signal and background samples.
        y : numpy.ndarray
            Corresponding labels (1 for signal, 0 for background).
    """

    # Open ROOT file and tree
    file_signal = uproot.open(f"{dir_path}/{root_mc}")
    file_back = uproot.open(f"{dir_path}/{root_data}")


    # Acess signal and background trees
    tree_signal = file_signal["Tsignal"]
    tree_background = file_back["Tback"]

    # List of variables to extract (features)
    variables = load_variables(version)
    
    # Load features as numpy arrays from ROOT trees
    signal_arrays = tree_signal.arrays(variables, library="np")
    background_arrays = tree_background.arrays(variables, library="np")

    # Stack features column-wise (shape: [n_events, n_features])
    X_signal = np.column_stack([signal_arrays[var] for var in variables])
    X_background = np.column_stack([background_arrays[var] for var in variables])

    # Create y (labels): 1 for signal, 0 for background
    y_signal = np.ones(X_signal.shape[0])
    y_background = np.zeros(X_background.shape[0])

    # Combine for machine learning
    X = np.concatenate([X_signal, X_background])
    y = np.concatenate([y_signal, y_background])

    return X.astype(np.float32), y.astype(np.float32)


class ClassificationDataset(Dataset):
    """
    PyTorch dataset wrapping features and labels.
    """
    def __init__(self, X, y):
        # Convert to torch tensors
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        