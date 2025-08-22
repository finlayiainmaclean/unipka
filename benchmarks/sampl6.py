import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau
from tqdm import tqdm
import unipka
from rdkit import Chem

from unipka.unipka import EnumerationError


def process_dataset(dataset_name, calc):
    """Process a single dataset and return results DataFrame with metrics"""
    print(f"\n--- Processing {dataset_name.upper()} ---")
    
    exp_col = "pKa"
    pred_col = "predicted_pKa"

    # Load dataset
    url = f"https://raw.githubusercontent.com/dptech-corp/Uni-pKa/refs/heads/main/dataset/{dataset_name}.tsv"
    df = pd.read_csv(url, sep="\t").rename(columns={"SMILES":"smiles", "TARGET": exp_col})

    # Make predictions
    predictions = []
    for macrostates_AB in tqdm(df.smiles.tolist(), desc=f"Predicting {dataset_name}"):
        try:
            macrostate_A, macrostate_B = macrostates_AB.split(">>")
            macrostate_A = macrostate_A.split(",")
            macrostate_B = macrostate_B.split(",")

            pka = calc.get_macro_pka_from_macrostates(macrostate_A=macrostate_A, macrostate_B=macrostate_B)
            predictions.append(pka)
        except Exception as e:
            print(f"Error processing {macrostates_AB}: {repr(e)}")
            predictions.append(None)

    df[pred_col] = predictions
    
    # Clean data
    old_len = len(df)
    df.dropna(subset=[exp_col, pred_col], inplace=True)
    new_len = len(df)

    print(f"Failed to generate pKa for {old_len-new_len} molecules in {dataset_name}")

    if df.empty:
        print(f"No data to evaluate for {dataset_name} after dropping rows with missing pKa values.")
        return None, {}

    # Calculate metrics
    mae = mean_absolute_error(df[exp_col], df[pred_col])
    rmse = np.sqrt(mean_squared_error(df[exp_col], df[pred_col]))
    r2 = r2_score(df[exp_col], df[pred_col])
    tau, _ = kendalltau(df[exp_col], df[pred_col])

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'tau': tau
    }

    print(f"R²: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Kendall's tau: {tau:.2f}")

    return df, metrics


def plot_results(datasets_results, figsize=(18, 6)):
    """Create 1x3 subplot with results from all datasets"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, (dataset_name, (df, metrics)) in enumerate(datasets_results.items()):
        if df is None:
            axes[idx].text(0.5, 0.5, f"No data for {dataset_name.upper()}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(dataset_name.upper())
            continue
            
        ax = axes[idx]
        exp_col = "pKa"
        pred_col = "predicted_pKa"
        
        # Scatter plot
        sns.scatterplot(
            data=df, x=pred_col, y=exp_col, alpha=0.6, edgecolor=None, ax=ax
        )

        # Add diagonal (ideal prediction line)
        lims = [
            min(df[exp_col].min(), df[pred_col].min()),
            max(df[exp_col].max(), df[pred_col].max())
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Add metrics text
        metrics_text = (
            f"$R^2$: {metrics['r2']:.2f}\n"
            f"$\\tau$: {metrics['tau']:.2f}\n"
            f"MAE: {metrics['mae']:.2f}\n"
            f"RMSE: {metrics['rmse']:.2f}"
        )
        ax.text(
            0.95, 0.05, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        ax.set_xlabel("Predicted pKa")
        ax.set_ylabel("Experimental pKa")
        ax.set_title(dataset_name.upper())
        
        # Make plots square
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def main():
    calc = unipka.UnipKa()
    
    datasets = ["sampl6", "sampl7", "sampl8"]
    results = {}
    
    # Process all datasets
    for dataset_name in datasets:
        try:
            df, metrics = process_dataset(dataset_name, calc)
            results[dataset_name] = (df, metrics)
        except Exception as e:
            print(f"Error processing {dataset_name}: {repr(e)}")
            results[dataset_name] = (None, {})
    
    # Create combined plot
    fig = plot_results(results)
    
    # Save figure
    plt.savefig("benchmarks/sampl_comparison.png", dpi=300, bbox_inches='tight')
    
    # Print summary
    print("\n--- SUMMARY ---")
    for dataset_name, (df, metrics) in results.items():
        if df is not None:
            print(f"{dataset_name.upper()}: n={len(df)}, R²={metrics['r2']:.2f}, MAE={metrics['mae']:.2f}")
        else:
            print(f"{dataset_name.upper()}: Failed to process")


if __name__=="__main__":
    main()