# SummerLIP25

Flavour Anomalies Internship project focusing on preparing data from Monte Carlo (MC) and real sources, training machine learning models to distinguish signal from background, and evaluating model performance.

---

## Project Structure & Description


| File/Folder             | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `InitialAnalysis/`                                      | Contains early data exploration. |
| `bMass.cc`, `bMass_data.cc`, `bMass_mc.cc`, `ntuple.cc` | C++ source files for ROOT-based mass fitting and data preparation. |
| `Machine_Learning/`                                     | Core ML pipeline: <br>• `Training/` – training scripts and configs <br>• `Evaluation/` – scripts to evaluate trained models <br>•`Data_Application/` - scripts to apply trained models |
| `variable_versions.json`                                | File containing map to different training versions and corresponding input features|
| `variable_versions.py`                                  | Loads variables used for training according to version                      |
| `Data_Application/`                                     | Post-evaluation results. |
| `ROOTvar.cc`                                            | Prepares ROOT files to apply trained models to full datasets.|
| `apply_model.py`                                        | Applies model to full dataset and saves ML output as new branch in ROOT file.|
| `weights.py`                                            | Computes weights to scale FoM, and saves them in json file.|
| `FoM.py`                                                | Computes and saves best threshold based on FoM maximisation.|
| `Plot.cc`                                               | Plots variables histograms pre and post threshold cut.|
| `Evaluation/`                                           | Evaluation results, plots, and post-training analyses. |
| `checkpoints/`                                          | Saved model weights from training (e.g., best/last checkpoints). |
| `B_evaluation.py`, `F_evaluation.py`, `evaluation.py`   | Evaluate models with different loss functions. |
| `feature_importance.py`                                 | Calculates feature importance using two different methods (permutation and SHAP).|
| `correlation_matrix.py`                                 | Creates correlation matrices for different input features.|
| `Training/`                                             | Training-related files and loss curve outputs. |
| `early_stopping.py`                                     | Early stopping implementation based on validation loss. |
| `losses.py`                                             | Custom loss functions (Balanced Cross-Entropy and Focal Loss). |
| `main.py`                                               | Main pipeline script for end-to-end training. |
| `models.py`                                             | Defines the neural network architectures. |
| `prepdata_v0.py`                                        | Preprocessing script: reads ROOT files, prepares PyTorch datasets. |
| `train.py`                                              | Model training launcher. |
| `Signal_vs_Background/`                                 | Signal/background separation: <br>• Signal from MC, background from data <br>• Plots histograms <br>• Saves output for ML training. |
| `ROC.py`                                                | Generates ROC curves for evaluation. |
| `comparison.cc`                                         | Compares signal vs. background variables (ROOT + histograms). |
| `prepdata.cc`                                           | Saves transformed signal/background samples to new ROOT files. |
| `.gitignore`                                            | Specifies files ignored by Git. |
| `requirements.txt`                                      | List of Python packages needed to run the project. |
| `README.md`                                             | Project overview, setup guide, and contribution instructions (this file). |


---

## Getting Started

### 1. Set Up SSH key (different than the one used to connect to LIP servers)

If you haven't already set up a GitHub SSH key from within the LIP servers:
#### Generate an SSH key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
accept default file location and no need for password. 

#### Copy public key:
```bash
cat ~/.ssh/id_ed25519.pub
```
#### Add public key to your Github account
GitHub → Settings → SSH and GPG keys → New SSH key
Paste the LIP server’s public key you just copied

### 2. Clone the Repository
```bash
git clone git@github.com:LIP-Flavour-Anomalies-studies/SummerLIP25.git
cd SummerLIP25
```

### 3. Create and Activate a Virtual Environment 
```bash
# Create virtual environment
python -m venv myenv
# Activate it (Linux/macOS)
source myenv/bin/activate
```

### 4. Install Required Packages
```bash
pip install -r requirements.txt
```
---

## How to Use the Project

1. `module load root` 
2. Activate environment
3. Run `root Signal_vs_Background/prepdata.cc` 
    - Creates new signal and background ROOT files with selected variables.
4. Run `python Machine_Learning/Training/main.py`
    - Trains model using preprocessed data.
    - Check which variables are being used for training by choosing version number.
5. Run `python Machine_Learning/Evaluation/evaluation.py`
    - Evaluates trained models with different loss functions.
    - Computes best threshold based on Youden J's statistic (as a working point).
    - Choose which version to be evaluated in `main()`.
6. Run `root Machine_Learning/Data_Application/ROOTvar.cc`
    - Prepares files to apply trained models on full datasets.
7. Run `python Machine_Learning/Data_Application/apply_model.py`
    - Applies model to full dataset
    - Saves ML output as new branch in the ROOT file.
8. Run `python Machine_Learning/Data_Application/weights.py`
    - Computes scaling weights to use in FoM.
9. Run `python Machine_Learning/Data_Application/FoM.py`
    - Determines new best threshold based on FoM maximisation and saves it in json file.

--- 

## Contribution

#### Working on a New Branch
```bash
# Create and switch to a new branch
git checkout -b your-branch-name

# Work on files, then stage and commit
git add changed_file.py
git commit -m "Describe your changes"

# Push the new branch to GitHub
git push origin your-branch-name
```

#### Sync with Main Regularly
```bash
# Switch to main branch
git checkout main
# Pull the latest updates from main (important to avoid merge conflicts)
git pull origin main 
# Go back to your branch
git checkout your-branch-name
# Merge latest changes from main into your branch
git merge main
```

#### Merging a Branch Back to Main

1. Push your branch:
```bash
git push origin my-feature
```
2. Go to GitHub and click "Compare & pull request".
3. Create the pull request (PR), assign reviewers if needed.
4. Click Merge pull request.
