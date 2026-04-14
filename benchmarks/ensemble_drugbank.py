"""
Benchmark `get_ensemble` on a slice of DrugBank slim (smallest small molecules by atom count).

Writes a two-panel figure: wall-time distribution (histogram + KDE, modern replacement for
``sns.distplot``) and total microstates vs wall time with Pearson/Spearman correlation.

Run from repo root: ``python benchmarks/ensemble_drugbank.py --help``
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from scipy import stats
from tqdm import tqdm

from unipka import UnipKa
from unipka._internal.template import get_ensemble


DRUGBANK_SLIM_URL = (
    "https://raw.githubusercontent.com/dhimmel/drugbank/refs/heads/gh-pages/data/drugbank-slim.tsv"
)


def load_sample_smiles(n_molecules: int = 10000, max_atoms: int = 50) -> pd.DataFrame:
    df = pd.read_csv(DRUGBANK_SLIM_URL, sep="\t")
    df = df[df["type"] == "small molecule"].copy()
    df["mol"] = df["inchi"].apply(Chem.MolFromInchi)
    df = df.dropna(subset=["mol"])
    df["n_atoms"] = df["mol"].apply(lambda m: m.GetNumAtoms())
    df = df[df["n_atoms"] < max_atoms + 1].sort_values(by="n_atoms").head(n_molecules)
    df["smiles"] = df["mol"].apply(lambda m: Chem.MolToSmiles(m))
    return df[["smiles"]].reset_index(drop=True)


def _correlation_text(x: list[int] | np.ndarray, y: list[float] | np.ndarray) -> str:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if len(xa) < 2 or np.std(xa) == 0 or np.std(ya) == 0:
        return "correlation n/a (constant x or y, or n<2)"
    rp, pp = stats.pearsonr(xa, ya)
    rs, ps = stats.spearmanr(xa, ya)
    return f"Pearson r = {rp:.4f} (p = {pp:.2e})\nSpearman ρ = {rs:.4f} (p = {ps:.2e})"


def save_benchmark_plots(
    times_s: list[float],
    n_microstates: list[int],
    out_path: Path,
) -> None:
    """Histogram + KDE of timings (replaces deprecated ``sns.distplot``) and microstates vs time."""
    times_ms = np.asarray(times_s, dtype=float) * 1e3
    nm = np.asarray(n_microstates, dtype=int)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(times_ms, kde=True, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Wall time per get_ensemble (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of wall times (hist + KDE)")

    axes[1].scatter(nm, times_ms, alpha=0.35, s=22, c="darkslategray")
    if len(nm) > 1 and np.unique(nm).size > 1:
        sns.regplot(x=nm, y=times_ms, ax=axes[1], scatter=False, color="crimson", line_kws={"lw": 2})
    axes[1].set_xlabel("Total microstates (summed over charge states)")
    axes[1].set_ylabel("Wall time per get_ensemble (ms)")
    axes[1].set_title("Enumeration size vs wall time")
    axes[1].text(
        0.03,
        0.97,
        _correlation_text(nm, times_ms),
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )

    fig.suptitle("DrugBank slim — get_ensemble benchmark", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Repeat the full molecule list this many times (after warmup) for steadier timings.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of full passes over the molecule list before measuring.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("benchmarks/ensemble_drugbank_plots.png"),
        help="Where to save the benchmark figure (hist + KDE of time; scatter microstates vs time).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write the matplotlib figure.",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=10000,
        help="Number of smallest DrugBank small molecules to benchmark (after InChI parse filter).",
    )
    args = parser.parse_args()
    plot_path = None if args.no_plots else args.plot_path

    pka = UnipKa()
    sample_df = load_sample_smiles(args.n_molecules)
    smiles = sample_df["smiles"].tolist()

    for _ in range(args.warmup):
        for smi in smiles:
            get_ensemble(Chem.MolFromSmiles(smi), pka.template_a2b, pka.template_b2a)

    sizes: list[int] = []
    times_s: list[float] = []
    n_microstates: list[int] = []
    for _ in range(args.rounds):
        for smi in tqdm(smiles, desc="get_ensemble"):
            t0 = time.perf_counter()
            ensemble = get_ensemble(Chem.MolFromSmiles(smi), pka.template_a2b, pka.template_b2a)
            dt = time.perf_counter() - t0
            times_s.append(dt)
            sizes.append(len(ensemble))
            n_microstates.append(sum(len(microstates) for microstates in ensemble.values()))
    n = len(times_s)
    mean_s = statistics.mean(times_s)
    print()
    print(f"Timed calls: {n} (warmup={args.warmup}, rounds={args.rounds})")
    print(f"Mean ensemble size (charge states): {statistics.mean(sizes):.3f}")
    print(f"Mean wall time per molecule: {mean_s * 1e3:.3f} ms")
    print(f"Median wall time per molecule: {statistics.median(times_s) * 1e3:.3f} ms")
    print(f"Total wall time (ensemble only): {sum(times_s):.4f} s")
    if n > 1:
        print(f"Std dev time per molecule: {statistics.stdev(times_s) * 1e3:.3f} ms")

    print(f"Mean number of microstates: {statistics.mean(n_microstates):.3f}")
    print(f"Median number of microstates: {statistics.median(n_microstates):.3f}")
    print(f"Total number of microstates: {sum(n_microstates)}")

    times_ms_arr = np.asarray(times_s, dtype=float) * 1e3
    if plot_path is not None:
        save_benchmark_plots(times_s, n_microstates, plot_path)
        print(f"Saved plot: {plot_path.resolve()}")
    print("Microstates vs wall time (ms), same units as scatter y-axis:")
    print(_correlation_text(n_microstates, times_ms_arr))


if __name__ == "__main__":
    main()
