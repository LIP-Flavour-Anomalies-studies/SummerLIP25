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
    # use fom thresholds
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

def save_outputs(input_file, loss_type, versions, fom_flag=0, tree_name='Tdata'):
    """
    Applies trained model to dataset and saves the ML output scores and thresholds
    as new branches to the existing ROOT file.

    Parameters:
        input_file : str
            Path to input ROOT file.
        loss_type : str
            Loss function used during training ('focal' or 'binary').
        versions : list of int
            Model versions to apply.
        fom_flag : int
            Whether to use FoM-optimized thresholds.
        tree_name : str
            Name of the TTree in the ROOT file.
    """
    
    # Open original ROOT file and read all branches into numpy arrays
    file = uproot.open(input_file)
    tree = file[tree_name]
    arrays = tree.arrays(library="np")
    num_entries = len(arrays["bTMass"])


    new_branches = {}
    for version in versions:
        # Load data and apply model
        labels, probabilities = model_application(input_file, loss_type, version, fom_flag)
        thr = load_threshold(loss_type, version, fom_flag)

        # Save model output and threshold as new branches
        new_branches[f"{loss_type[0].upper()}_score_v{version}"] = probabilities
        new_branches[f"{loss_type[0].upper()}_thr_v{version}"] = np.full_like(probabilities, thr, dtype=np.float32)

    # Merge new and old branches
    all_branches = arrays.copy()
    all_branches.update(new_branches)

    # Overwrite or create new file with updated tree
    if fom_flag == 0:
        output_file = input_file.replace(".root", f"_mlJ_output.root")
    elif fom_flag == 1:
        output_file = input_file.replace(".root", f"_mlFoM_output.root")
    
    new_file = uproot.recreate(output_file)
    branch_types = {key: val.dtype for key, val in all_branches.items()}
    new_file.mktree(tree_name, branch_types)
    new_file[tree_name].extend(all_branches)


def main():
    loss_type = "binary" # or "focal"

    input_data = "Machine_Learning/Data_Application/ROOT/data_selected.root"
    input_mc = "Machine_Learning/Data_Application/ROOT/mc_selected.root"
    
    versions = [0, 1, 2, 3, 4, 5] 

    save_outputs(input_data, loss_type, versions)
    save_outputs(input_mc, loss_type, versions)

    # Optional: also add FoM-thresholded branches
    save_outputs(input_data, loss_type, versions, fom_flag=1)
    save_outputs(input_mc, loss_type, versions, fom_flag=1)

if __name__ == '__main__':
    main()
