import ROOT
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Adiciona o diretório principal ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from variable_versions import load_variables

# Caminhos dos ficheiros ROOT
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Signal_vs_Background', 'ROOT_files'))
root_mc = "signal.root"
root_data = "background.root"

# Output
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Correlation_Matrices'))
os.makedirs(out_dir, exist_ok=True)


# Abrir os ficheiros ROOT
file_mc = ROOT.TFile.Open(os.path.join(dir_path, root_mc))
file_data = ROOT.TFile.Open(os.path.join(dir_path, root_data))
mcTree = file_mc.Get("Tsignal")
dataTree = file_data.Get("Tback")

# Função para extrair dados de um TTree
def build_dataframe(tree, data_type, variables):
    data = {var: [] for var in variables}
    for event in tree:
        for var in variables:
            try:
                data[var].append(getattr(event, var))
            except AttributeError:
                print(f"Variável {var} não encontrada no TTree.")
    return pd.DataFrame(data)

# Função para criar e guardar o gráfico
def make_correlation_plot(df, data_type, version):
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"{data_type} Correlation Matrix (v{version})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{data_type}_CorrelationMatrix_v{version}.png"))
    plt.close()

# Loop para todas as versões
for version in [0, 1, 2, 3, 4, 5]:
    print(f"A gerar matrizes de correlação para v{version}...")

    variables = load_variables(version, config_path=os.path.join(os.path.dirname(__file__), '..', 'variable_versions.json'))

    df_signal = build_dataframe(mcTree, "Signal", variables)
    df_bkg = build_dataframe(dataTree, "Background", variables)

    make_correlation_plot(df_signal, "Signal", version)
    make_correlation_plot(df_bkg, "Background", version)

print(" Matrizes de correlação geradas para todas as versões.")

