import argparse
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau
from tqdm import tqdm

import unipka
from unipka._internal.template import TRANSLATE_PH, log_sum_exp


def _canonical_smi(s: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=True)


def macro_pka_from_ensemble_free_energy(
    ensemble_free_energy: dict,
    acid_smiles: list[str],
    base_smiles: list[str],
) -> float:
    """
    Match ``get_macro_pka_from_macrostates`` when energies are only known for microstates
    returned by :meth:`unipka.UnipKa._predict_ensemble_free_energy`.
    """
    acid_set = {_canonical_smi(s) for s in acid_smiles}
    base_set = {_canonical_smi(s) for s in base_smiles}
    g_a: list[float] = []
    g_b: list[float] = []
    for _q, rows in ensemble_free_energy.items():
        for smi, _mol, g in rows:
            if smi in acid_set:
                g_a.append(g)
            if smi in base_set:
                g_b.append(g)
    if not g_a or not g_b:
        raise ValueError("listed acid/base macrostates not all present in ensemble predictions")
    return log_sum_exp(g_a) - log_sum_exp(g_b) + TRANSLATE_PH


def process_dataset(dataset_name, calc):
    """Process a single dataset and return results DataFrame with metrics"""
    print(f"\n--- Processing {dataset_name.upper()} ---")
    
    exp_col = "pKa"
    pred_col = "predicted_pKa"

    # Load dataset
    url = f"https://raw.githubusercontent.com/dptech-corp/Uni-pKa/refs/heads/main/dataset/{dataset_name}.tsv"
    df = pd.read_csv(url, sep="\t").rename(columns={"SMILES":"smiles", "TARGET": exp_col})

    # Enumerate + predict microstates via _predict_ensemble_free_energy (prune window from calc),
    # then macro pKa from listed acid/base pools (same log-sum-exp as get_macro_pka_from_macrostates).
    predictions = []
    wall_s = []
    n_microstates = []
    for macrostates_AB in tqdm(df.smiles.tolist(), desc=f"Predicting {dataset_name}"):
        try:
            macrostate_A, macrostate_B = macrostates_AB.split(">>")
            macrostate_A = macrostate_A.split(",")
            macrostate_B = macrostate_B.split(",")
            mol_q = Chem.MolFromSmiles(macrostate_A[0])
            t0 = time.perf_counter()
            _ensemble, efe = calc._predict_ensemble_free_energy(mol_q)
            wall_s.append(time.perf_counter() - t0)
            n_microstates.append(sum(len(v) for v in _ensemble.values()))
            pka = macro_pka_from_ensemble_free_energy(efe, macrostate_A, macrostate_B)
            predictions.append(pka)
        except Exception as e:
            print(f"Error processing {macrostates_AB}: {repr(e)}")
            predictions.append(None)
            wall_s.append(float("nan"))
            n_microstates.append(0)

    df[pred_col] = predictions
    df["ensemble_wall_s"] = wall_s
    df["n_ensemble_microstates"] = n_microstates
    
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
    wt = df["ensemble_wall_s"].dropna()
    if len(wt):
        print(
            f"Ensemble path: mean wall time {wt.mean():.3f} s / row, "
            f"mean microstates {df['n_ensemble_microstates'].mean():.1f}"
        )

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
    parser = argparse.ArgumentParser(
        description="SAMPL pKa benchmark using _predict_ensemble_free_energy (pruned ensemble expansion)."
    )
    parser.add_argument(
        "--prune-window",
        type=float,
        default=25.0,
        help="Passed to UnipKa(ensemble_energy_prune_window=...); microstates farther than this "
        "above the current minimum predicted ΔG are dropped during expansion.",
    )
    args = parser.parse_args()

    calc = unipka.UnipKa(ensemble_energy_prune_window=args.prune_window)
    print(f"Using ensemble_energy_prune_window={args.prune_window}")

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
    plot_results(results)
    
    # Save figure
    plt.savefig("benchmarks/sampl_results.png", dpi=300, bbox_inches='tight')
    
    # Print summary
    print("\n--- SUMMARY ---")
    for dataset_name, (df, metrics) in results.items():
        if df is not None:
            print(f"{dataset_name.upper()}: n={len(df)}, R²={metrics['r2']:.2f}, MAE={metrics['mae']:.2f}")
        else:
            print(f"{dataset_name.upper()}: Failed to process")


if __name__=="__main__":
    main()