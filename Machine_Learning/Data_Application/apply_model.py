"""
Module to apply trained model to full dataset.
Teresa 25/07/2025
"""
import sys
import os
import uproot
import numpy as np
import torch
import json
from torch.utils.data import Dataset

# Add the directory containing variables
sys.path.append(os.path.abspath("Machine_Learning"))
from variable_versions import load_variables

sys.path.append(os.path.abspath("Machine_Learning/Training"))
from Training.models import ClassificationModel

def prepdata_for_application(file_path, tree_name="Tdata", version=None):
    """
    Load features from a ROOT tree for applying trained model.

    Parameters:
            Path to the ROOT file with full dataset.
        tree_name : str
            Name of the TTree inside the ROOT file.

    Returns:
        X : torch.Tensor
            Feature matrix ready for model inference.
    """
    file = uproot.open(file_path)
    tree = file[tree_name]

    # List of variables used during training

    if version is None:
        raise ValueError('Trained model version not chosen')

    variables = load_variables(version)
    arrays = tree.arrays(variables, library="np")
    X = np.column_stack([arrays[var] for var in variables])
    X = torch.from_numpy(X.astype(np.float32))

    return X

def load_threshold(loss_type, version, fom_flag=0):

    # use youden thresholds
    if fom_flag == 0:
        if loss_type == "focal":
            thr_path = "Machine_Learning/Data_Application/thresholds/focal_thresholds.json"
        elif loss_type == "binary":
            thr_path = "Machine_Learning/Data_Application/thresholds/binary_thresholds.json"
        else:
            raise ValueError("Invalid loss type. Use 'focal' or 'binary'.")
    elif fom_flag == 1:
        if loss_type == "focal":
            thr_path = "Machine_Learning/Data_Application/thresholds/FoM_focal_thresholds.json"
        elif loss_type == "binary":
            thr_path = "Machine_Learning/Data_Application/thresholds/FoM_binary_thresholds.json"
        else:
            raise ValueError("Invalid loss type. Use 'focal' or 'binary'.")

    with open(thr_path, 'r') as f:
        data = json.load(f)

    version_str = str(version)
    thr = data.get(loss_type, {}).get(version_str)
    if thr is None:
        raise ValueError(f"Threshold not found for loss_type '{loss_type}' and version '{version}'")

    return thr


def model_application(file_path, loss_type, version, fom_flag=0):
    """
    Apply trained model on full dataset.

    Parameters:
        file_path : str
            Path to ROOT file with data.
        loss_type : str
            'focal' or 'binary'.
        version : str or int
            Model version.

    Returns:
        np.ndarray: Predicted labels (0 or 1).
        np.ndarray: Model output probabilities.
    """

    if loss_type == "focal":
        checkpoint_path = f"Machine_Learning/Evaluation/checkpoints/F_model_checkpoint_v{version}.pth"
    elif loss_type == "binary":
        checkpoint_path = f"Machine_Learning/Evaluation/checkpoints/B_model_checkpoint_v{version}.pth"
    else:
        raise ValueError("Invalid loss type. Use 'focal' or 'binary'.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    X = prepdata_for_application(file_path, version=version)
    input_size = len(X[0])
    model = ClassificationModel(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.no_grad():
        outputs = model(X)
        probabilities = outputs.squeeze().numpy()
    
    # Apply threshold
    best_thr = load_threshold(loss_type, version, fom_flag)
    labels = (probabilities >= best_thr).astype(int)

    return labels, probabilities

def save_signal_events(input_file, output_file, loss_type, versions, fom_flag=0, tree_name='Tdata'):
    """
    Applies trained model to dataset, keeps only signal events (label==1),
    and writes them to a new ROOT file.

    Parameters:
        input_file : str
            Path to input ROOT file.
        output_file : str
            Path to output ROOT file (signal events only).
        tree_name : str
            Name of the TTree in the ROOT file.
        loss_type : str
            Loss function used during training ('focal' or 'binary').
        version : str or int
            Model version to load.
    """
    
    # Open original ROOT file and read all branches into numpy arrays
    file = uproot.open(input_file)
    tree = file[tree_name]
    arrays = tree.arrays("bTMass",library="np")


    filtered_arrays = {}
    for version in versions:
        # Load data and apply model
        labels, probabilities = model_application(input_file, loss_type, version, fom_flag)
        # Filter events where predicted label == 1
        mask = (labels == 1)
        filtered_arrays[f"bTMass_v{version}"] = arrays["bTMass"][mask]


    # Save filtered bTMass arrays into one ROOT file 
    new_file = uproot.recreate(output_file)

    for tree, bTMass_array in filtered_arrays.items():
        new_file.mktree(tree, {"bTMass": bTMass_array.dtype})
        new_file[tree].extend({"bTMass": bTMass_array})

def main():
    loss_type = "binary" # or "focal"

    input_data = "Machine_Learning/Data_Application/ROOT/data_selected.root"
    output_data = f"Machine_Learning/Data_Application/ROOT/bTMass_{loss_type}.root"
    fom_output_data = f"Machine_Learning/Data_Application/ROOT/FoM_bTMass_{loss_type}.root"
    input_mc = "Machine_Learning/Data_Application/ROOT/mc_selected.root"
    output_mc = f"Machine_Learning/Data_Application/ROOT/bTMass_mc_{loss_type}.root"
    fom_output_mc = f"Machine_Learning/Data_Application/ROOT/FoM_bTMass_mc_{loss_type}.root"
    
    versions = [1] #[1, 2, 3]

    save_signal_events(input_data, output_data, loss_type, versions)
    save_signal_events(input_mc, output_mc, loss_type, versions)

    #save_signal_events(input_data, fom_output_data, loss_type, versions, fom_flag=1)
    #save_signal_events(input_mc, fom_output_mc, loss_type, versions, fom_flag=1)

if __name__ == '__main__':
    main()
