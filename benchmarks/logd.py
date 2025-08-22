import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau
from tqdm import tqdm
import unipka
from rdkit import Chem
from rdkit.Chem import Draw


from unipka.unipka import EnumerationError


def main():
    calc = unipka.UnipKa()

    exp_col = "logD(7.4)"
    pred_col = "predicted_logD(7.4)"


    df = pd.read_csv(
        "https://raw.githubusercontent.com/nanxstats/logd74/refs/heads/master/logd74.tsv",
        sep="\t"
    ).rename(columns={"SMILES":"smiles", "logD7.4": exp_col}).sample(100)

    df['mol'] = [Chem.MolFromSmiles(smi) for smi in df.smiles]
    df = df.dropna(subset=['mol'])

    # --- Predictions ---
    logds = []
    for smi in tqdm(df.smiles, desc="Predicting logD"):
        try:
            logd = calc.get_logd(smi, pH=7.4) 
        except EnumerationError as e:
            print(repr(e))
            logd = None

        logds.append(logd)
    df[pred_col] = logds
    
    # Remove failed
    bad_df = df[df[pred_col].isnull()]

    # Generate the grid image
    img = Draw.MolsToGridImage(
        bad_df.mol.tolist(),
        molsPerRow=10,
        subImgSize=(300, 300),
        useSVG=False  # Use PNG for file output
    )
    
    # Save to file
    img.save("benchmarks/logd_failed_molecules.png")


    old_len = len(df)
    df.dropna(subset=[exp_col, pred_col], inplace=True)
    new_len = len(df)

    print(f"Failed to generate logD for {old_len-new_len} molecules")

    if df.empty:
        print("No data to evaluate after dropping rows with missing logD values.")
        return

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

    ax.set_xlabel("Predicted logD7.4")
    ax.set_ylabel("Experimental logD7.4")
    plt.tight_layout()

    # Save figure
    plt.savefig("benchmarks/logd_results.png", dpi=300)
    plt.close()

if __name__=="__main__":
    main()

