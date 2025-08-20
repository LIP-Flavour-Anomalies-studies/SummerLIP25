import ROOT
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Project path setup ----------------
HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))                 # Machine_Learning
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, '..'))         # SummerLIP25
ROOT_DIR = os.path.join(PROJECT_ROOT, 'Signal_vs_Background', 'ROOT_files')

# Ensure project modules are importable
sys.path.append(BASE)

from variable_versions import load_variables

# ---------------- I/O configuration ----------------
root_mc = "signal.root"
root_data = "background.root"

out_dir = os.path.join(HERE, 'Correlation_Matrices')
os.makedirs(out_dir, exist_ok=True)

# ---------------- ROOT file & tree loading ----------------
mc_path = os.path.join(ROOT_DIR, root_mc)
data_path = os.path.join(ROOT_DIR, root_data)

file_mc = ROOT.TFile.Open(mc_path)
file_data = ROOT.TFile.Open(data_path)

if not file_mc or file_mc.IsZombie():
    raise FileNotFoundError(f"Could not open MC file at: {mc_path}")
if not file_data or file_data.IsZombie():
    raise FileNotFoundError(f"Could not open data file at: {data_path}")

mcTree = file_mc.Get("Tsignal")
dataTree = file_data.Get("Tback")
if mcTree is None:
    raise RuntimeError("TTree 'Tsignal' not found in MC file.")
if dataTree is None:
    raise RuntimeError("TTree 'Tback' not found in data file.")

# ---------------- Helpers ----------------
def build_dataframe(tree, variables):
    """
    Extracts branches listed in `variables` from a ROOT TTree into a pandas DataFrame.
    Missing branches are reported and skipped.
    """
    data = {var: [] for var in variables}
    missing = set()
    n_events = tree.GetEntries()
    for event in tree:
        for var in variables:
            try:
                data[var].append(getattr(event, var))
            except AttributeError:
                missing.add(var)
                # Keep structure; append NaN so shapes remain consistent
                data[var].append(float('nan'))
    if missing:
        print(f"[WARN] Missing branches in tree: {sorted(missing)}")
    df = pd.DataFrame(data)
    # Keep only numeric columns for correlation
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def save_correlation_artifacts(df, data_type, version):
    """
    Computes the Pearson correlation matrix, saves a PNG heatmap and a CSV.
    Filenames follow the convention expected by downstream scripts.
    """
    corr = df.corr(method='pearson')

    # Save CSV (this is what cumulative_shap_groups.py expects)
    csv_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix_v{version}.csv")
    corr.to_csv(csv_path)

    # Save PNG (optional visualization)
    plt.figure(figsize=(27, 25))
    # annot=False to avoid huge labels; change to True if you really want per-cell text
    sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"{data_type} Correlation Matrix (v{version})")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix_v{version}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[OK] Saved {data_type} CSV → {csv_path}")
    print(f"[OK] Saved {data_type} PNG → {png_path}")

# ---------------- Versions to generate ----------------
# You can list multiple versions if needed (e.g., [6, 7, 11]); here we target v11.
VERSIONS = [11]

for version in VERSIONS:
    print(f"Generating correlation matrices for v{version}...")

    # Load the variable list for this version
    variables = load_variables(version, config_path=os.path.join(BASE, 'variable_versions.json'))

    # Build DataFrames from the trees using the specified variable set
    df_signal = build_dataframe(mcTree, variables)
    df_bkg = build_dataframe(dataTree, variables)

    # Ensure column order matches `variables` exactly (good for downstream alignment)
    df_signal = df_signal.reindex(columns=variables)
    df_bkg = df_bkg.reindex(columns=variables)

    # Save correlation CSVs + PNGs with the expected naming scheme
    save_correlation_artifacts(df_signal, "Signal", version)
    save_correlation_artifacts(df_bkg, "Background", version)

print(f"Done. Correlation matrices generated for versions: {VERSIONS}")

