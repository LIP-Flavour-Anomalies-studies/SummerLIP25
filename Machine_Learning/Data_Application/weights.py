"""
Module to compute weights to scale FoM.
Teresa 27/07/2025
"""
import sys
import os
import ROOT
from ROOT import RooRealVar, RooDataSet, RooArgSet, RooExponential, RooFit, TCanvas, TLine
import uproot
import numpy as np
import json

# Add the directory containing function to load thresholds
sys.path.append(os.path.abspath("Machine_Learning/Data_Application"))
from apply_model import load_threshold

def get_signal_arrays(path, versions, loss_type, tree_name="Tdata"):
    """
    Load bTMass values after ML score > threshold cut for each version.
    """

    file = uproot.open(path)
    tree = file[tree_name]

    signal_arrays = []

    for version in versions:
        score_branch = f"{loss_type[0].upper()}_score_v{version}"

        arrays = tree.arrays(["bTMass", score_branch], library="np")

        bTMass = arrays["bTMass"]
        score = arrays[score_branch]
        thr = load_threshold(loss_type, version)
        
        # Apply cut
        mask = score >= thr
        signal_arrays.append(bTMass[mask])

    return signal_arrays

def estimate_Np(mass_values, peak_range=(5.15, 5.4), fit_range=(5.0, 5.6)):

    s_left, s_right = peak_range
    mmin, mmax = fit_range

    # Define RooRealVar
    mass = RooRealVar("mass", "bTMass", mmin, mmax, "GeV/c^{2}")

    # Define sideband ranges
    mass.setRange("left_sb", mmin, s_left)
    mass.setRange("right_sb", s_right, mmax)
    mass.setRange("sidebands", mmin, s_left)
    mass.setRange("sidebands", s_right, mmax)
    mass.setRange("peak", s_left, s_right)
    mass.setRange("fit", mmin, mmax)

    # Prepare RooDataSet with only sideband data
    ds = RooDataSet("ds", "sideband data", RooArgSet(mass))
    for val in mass_values:
        # Only add points in sidebands for fitting
        if (mmin <= val <= s_left) or (s_right <= val <= mmax):
            mass.setVal(val)
            ds.add(RooArgSet(mass))

    # Background model (exponential)
    Lambda = RooRealVar("lambda", "lambda", -1.0, -10.0, 0.0)
    background = RooExponential("background", "Background", mass, Lambda)

    # Number of background events (floating)
    n_back = RooRealVar("n_back", "n_back", ds.sumEntries(), 0, 1.5 * ds.sumEntries())

    # Extended PDF
    extended_bkg = ROOT.RooExtendPdf("extended_bkg", "extended background pdf", background, n_back)

    # Fit only in sidebands range
    fit_result = extended_bkg.fitTo(ds, RooFit.Range("sidebands"), RooFit.Save(), RooFit.PrintLevel(-1))

    # Estimate background in peak region by integration
    integral_peak = background.createIntegral(RooArgSet(mass), RooFit.Range("peak")).getVal()
    integral_sidebands = background.createIntegral(RooArgSet(mass), RooFit.Range("sidebands")).getVal()

    Np_est = n_back.getVal() * (integral_peak / integral_sidebands)

    return Np_est

def compute_weights(data_path, mc_path, versions, loss_type):
    
    # sidebands
    s_left = 5.15
    s_right = 5.4
    # Assume signal is twice of Np
    frac = 2

    data_arrays = get_signal_arrays(data_path, versions, loss_type)
    mc_arrays = get_signal_arrays(mc_path, versions, loss_type)

    weights = {}

    for i, version in enumerate(versions):
        data_array = data_arrays[i]

        mc_array = mc_arrays[i]
        S_mc = len(mc_array)

        N_l = np.sum(data_array < s_left)
        N_h = np.sum(data_array > s_right)

        # Estimate background in peak using RooFit on sidebands
        N_p = estimate_Np(data_array, peak_range=(s_left, s_right))
    
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

    versions = [0, 1, 2, 3, 4, 5]
    loss_type = "binary"

    data_path = f"Machine_Learning/Data_Application/ROOT/data_selected_ml_output.root"
    mc_path = f"Machine_Learning/Data_Application/ROOT/mc_selected_ml_output.root"
    

    weights = compute_weights(data_path, mc_path, versions, loss_type)
    save_weights(weights, f"Machine_Learning/Data_Application/weights/weights_{loss_type}.json")
    
if __name__ == '__main__':
    main()




