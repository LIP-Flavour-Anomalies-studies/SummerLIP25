import uproot
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


mc_path = "NewData/ROOT_files/signal.root"
data_path = "NewData/ROOT_files/background.root"

out_dir = "NewData/Correlation_Matrices"
os.makedirs(out_dir, exist_ok=True)

# ---------------- ROOT file & tree loading ----------------
try:
    file_mc = uproot.open(mc_path)
    file_data = uproot.open(data_path)
except Exception as e:
    raise FileNotFoundError(f"Could not open ROOT file: {e}")

if "Tsignal" not in file_mc:
    raise RuntimeError("TTree 'Tsignal' not found in MC file.")
if "Tback" not in file_data:
    raise RuntimeError("TTree 'Tback' not found in data file.")

mcTree = file_mc["Tsignal"]
dataTree = file_data["Tback"]

# ---------------- Helpers ----------------
def build_dataframe(tree, variables):
    """
    Extracts branches listed in `variables` from a ROOT TTree into a pandas DataFrame.
    Missing branches are reported and filled with NaN.
    """
    available = set(tree.keys())
    missing = [var for var in variables if var not in available]
    if missing:
        print(f"[WARN] Missing branches in tree: {sorted(missing)}")
    # Only load available branches
    load_vars = [var for var in variables if var in available]
    df = tree.arrays(load_vars, library="pd")
    # Add missing columns as NaN
    for var in missing:
        df[var] = float('nan')
    # Ensure column order matches variables
    df = df.reindex(columns=variables)
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def save_correlation_artifacts(df, data_type):
    """
    Computes the Pearson correlation matrix, saves a PNG heatmap and a CSV.
    Filenames follow the convention expected by downstream scripts.
    """
    corr = df.corr(method='pearson')

    # Save CSV (this is what cumulative_shap_groups.py expects)
    csv_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix.csv")
    #csv_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix_final.csv")
    corr.to_csv(csv_path)

    # Save PNG (optional visualization)
    plt.figure(figsize=(27, 25))
    #plt.figure(figsize=(16, 14))
    # annot=False to avoid huge labels; change to True if you really want per-cell text
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"{data_type} Correlation Matrix")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix.png")
    #png_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix_final.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[OK] Saved {data_type} CSV → {csv_path}")
    print(f"[OK] Saved {data_type} PNG → {png_path}")

# ---------------- Versions to generate ----------------
print(f"Generating correlation matrices ...")

# Load the variable list for this version
variables = ["bVtxCL", "kstTMass", "kstPt", "kstTrkmPt", "kstTrkmDCABS", "kstTrkpPt", "kstTrkpDCABS", 
"mumPt", "mupPt", "bCosAlphaBS", "bLBS", "bDCABS", "muLeadingPt", "muTrailingPt", "bLBSs", "bDCABSs", 
"kstTrkmDCABSs", "kstTrkpDCABSs", "kstTrkpPtR", "kstTrkmPtR",  "muTrailingPtR", "muLeadingPtR", "mumuPtR", 
"kstPtR","mumIsoPtR_dr04", "mupIsoPtR_dr04", "kstTrkmIsoPtR_dr04", "kstTrkpIsoPtR_dr04", "IsoPtR_dr04_sum"]

#variables = ['bLBSs', 'kstPt', 'IsoPtR_dr04_sum', 'kstTrkpDCABSs', 'kstTrkmDCABSs', 'bVtxCL', 'mumPt', 
#        'muLeadingPt', 'bDCABSs', 'mupPt', 'mupIsoPtR_dr04', 'kstTrkpDCABS', 'mumuPtR']

# Build DataFrames from the trees using the specified variable set
df_signal = build_dataframe(mcTree, variables)
df_bkg = build_dataframe(dataTree, variables)

# Ensure column order matches `variables` exactly (good for downstream alignment)
df_signal = df_signal.reindex(columns=variables)
df_bkg = df_bkg.reindex(columns=variables)

# Save correlation CSVs + PNGs with the expected naming scheme
save_correlation_artifacts(df_signal, "Signal")
save_correlation_artifacts(df_bkg, "Background")

print(f"Done. Correlation matrices generated")