"""
Module to compute ROC curves of cut based analysis.
Teresa 17/07/2025
"""
import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import seaborn as sns
sns.set_style("darkgrid")

# Paths
dir_path = "/user/u/u25teresaesc/Internship/Signal_vs_Background/ROOT_files"
root_signal = "signal.root"
root_back = "background.root"

# Input root files and trees
f_signal = uproot.open(f"{dir_path}/{root_signal}")
f_back = uproot.open(f"{dir_path}/{root_back}")

Tsignal = f_signal["Tsignal"]
Tback = f_back["Tback"]

# Variables to loop over 
variables = ["bVtxCL", "bPt", "kstTMass", "kstPt", "mumuMass", "mumuPt", 
        "kstTrkmPt",  "kstTrkmDCABS", "kstTrkpPt", "kstTrkpDCABS",
        "mumPt", "mupPt", "bCosAlphaBS", "bLBS", "bDCABS",
        "muLeadingPt", "muTrailingPt", "bLBSs", "bDCABSs",
        "kstTrkmDCABSs", "kstTrkpDCABSs"]

# Load features as numpy arrays for trees
signal = Tsignal.arrays(variables, library="np")
background = Tback.arrays(variables, library="np")

# number of threshold points for ROC
n_thresh = 100

# Store data for graphs 
roc_data = {}

for var in variables:
    s_vals = signal[var]
    b_vals = background[var]

    # find common threshold range
    values = np.concatenate([s_vals, b_vals])
    thresholds = np.linspace(np.min(values), np.max(values), n_thresh)

    s_eff = []
    b_eff = []

    for t in thresholds:

        s_pass = s_vals > t
        b_pass = b_vals > t 

        # fraction of events that passed the cut
        s_eff.append(np.sum(s_pass) / len(s_vals))
        b_eff.append((np.sum(b_pass) / len(b_vals)))

    # sort points for AUC calculation
    s_eff = np.array(s_eff)
    b_eff = np.array(b_eff)
    sorted_idx = np.argsort(b_eff)
    auc_val = auc(b_eff[sorted_idx], s_eff[sorted_idx])

    roc_data[var] = {
        "s_eff": s_eff,
        "b_eff": b_eff,
        "auc": auc_val
    }

# Plot ROC curves 
plt.figure(figsize=(10, 8))
for var, data in roc_data.items():
    plt.plot(data["b_eff"], data["s_eff"], label=f"{var} (AUC={data['auc']:.3f})")

plt.ylabel("Signal Efficiency (TPR)")
plt.xlabel("Background Efficiency (FPR)")
plt.title("ROC Curves for Discriminating Variables")
plt.legend()
plt.savefig("roc_curves.png")
plt.show()


