"""
Module to compute weights to scale FoM.
Teresa 27/07/2025
"""
import uproot
import numpy as np
import json

def get_arrays(path, versions):
    """
    Load bTMass arrays from a ROOT file for given versions.
    """

    file = uproot.open(path)
    arrays = []

    for version in versions:
        tree = file[f"bTMass_v{version}"]
        array = tree.arrays("bTMass", library="np")["bTMass"]
        arrays.append(array)

    return arrays


def compute_weights(data_path, mc_path, versions):
    
    # sidebands
    s_left = 5.15
    s_right = 5.4
    # Assume 1% of Np is signal
    frac = 2

    data_arrays = get_arrays(data_path, versions)
    mc_arrays = get_arrays(mc_path, versions)

    weights = {}

    for i, version in enumerate(versions):
        data_array = data_arrays[i]

        mc_array = mc_arrays[i]
        S_mc = len(mc_array)

        N_l = np.sum(data_array < s_left)
        N_h = np.sum(data_array > s_right)
        N_p = np.sum((data_array <= s_right) & (data_array >= s_left))

        # Background fraction in the peak (to scale MC bkg)
        w_B = N_p / (N_h + N_l)
        
        # Signal weight 
        w_S = frac * N_p / S_mc

        weights[version] = {
            "w_B": w_B,
            "w_S": w_S,
        }

    return weights

def save_weights(weights, save_dir):

    with open(save_dir, "w") as f:
        json.dump(weights, f, indent=4)
        

def main():

    versions = [1, 2, 3]
    loss_type = "binary"

    data_path = f"Machine_Learning/Data_Application/ROOT/bTMass_{loss_type}.root"
    mc_path = f"Machine_Learning/Data_Application/ROOT/bTMass_mc_{loss_type}.root"
    

    weights = compute_weights(data_path, mc_path, versions)
    save_weights(weights, f"Machine_Learning/Data_Application/weights/weights_{loss_type}.json")
    
if __name__ == '__main__':
    main()




