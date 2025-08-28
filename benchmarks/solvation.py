import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau
from unipka._internal.solvation import get_solvation_energy
from rdkit import Chem
import ray

exp_col = "expt"
pred_col = "pred"


def process_dataset():
    """Process a single dataset and return results DataFrame with metrics"""
    print(f"\n--- Processing FreeSolv ---")
    
    

    df = pd.read_csv("https://raw.githubusercontent.com/MobleyLab/FreeSolv/refs/heads/master/database.txt", sep=";",skiprows=2)
    df = df[[' SMILES', ' experimental value (kcal/mol)']]
    df.columns = ['smiles', exp_col]
    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    df['mol'] = mols
    df = df.dropna(subset=['mol'])

    ray_func = ray.remote(get_solvation_energy)
    df[pred_col] = ray.get([ray_func.remote(mol) for mol in mols])
    
    # Clean data
    old_len = len(df)
    df.dropna(subset=[pred_col], inplace=True)
    new_len = len(df)

    print(f"Failed to generate pKa for {old_len-new_len} molecules in FreeSolv")

    if df.empty:
        print(f"No data to evaluate for FreeSolv after dropping rows with missing values.")
        return None, {}



    return df

def main():    
    
    df = process_dataset()

    # --- Evaluation ---
    mae = mean_absolute_error(df[exp_col], df[pred_col])
    rmse = np.sqrt(mean_squared_error(df[exp_col], df[pred_col]))
    r2 = r2_score(df[exp_col], df[pred_col])
    tau, _ = kendalltau(df[exp_col], df[pred_col])

    print(f"R²: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Kendall’s tau: {tau:.2f}")

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(
        data=df, x=pred_col, y=exp_col, alpha=0.5, edgecolor=None
    )

    # Add diagonal (ideal prediction line)
    lims = [
        min(df[exp_col].min(), df[pred_col].min()),
        max(df[exp_col].max(), df[pred_col].max())
    ]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Add metrics text (top right corner)
    metrics_text = (
        f"$R^2$: {r2:.2f}\n"
        f"$\\tau$: {tau:.2f}\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}"
    )
    ax.text(
        0.95, 0.05, metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    ax.set_xlabel("Predicted hydration energy")
    ax.set_ylabel("Experimental hydration energy")
    plt.tight_layout()

    # Save figure
    plt.savefig("benchmarks/solvation_results.png", dpi=300)
    plt.close()


if __name__=="__main__":
    main()