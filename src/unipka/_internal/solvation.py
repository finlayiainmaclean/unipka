import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def get_boltzmann_weighted_average(*, properties: np.ndarray, energies: np.ndarray, T: float = 298):
    """ Some simple Python code to calculate Boltzmann weights from energies in kcal/mol """

    assert len(properties)==len(energies)

    if len(properties)==1:
        return properties[0], np.ones(1)

    energies = energies - np.min(energies)

    R = 3.1668105e-6 # eH/K
    weights = np.exp(-1*energies/(627.509*R*T))
    weights = weights / np.sum(weights)


    return np.average(properties, weights=weights), weights

def embed(mol: Chem.Mol, /, *, num_confs: int = 100, rmsd_threshold: float | None = 0.25) -> Chem.Mol:
    """Embed `nconf` ETKDGv3 conformers and MMFF94‑optimise them in place."""
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if rmsd_threshold:
        params.pruneRmsThresh = rmsd_threshold
    params.numThreads = 0  # use all cores for embedding
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    for cid in range(mol.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        if ff:
            ff.Minimize(maxIts=10000)
    return mol


def get_xtb_solvation(mol: Chem.Mol, charge: int =0, num_cores: int =1) -> float:
    """
    Run GFN2-xTB optimisation in ALPB water and return solvation energy (kcal/mol).
    Safe for multiprocessing (each call uses an isolated scratch dir).
    """

    if shutil.which("xtb") is None:
        raise FileNotFoundError("xtb is either not installed or not found in PATH. Please install xtb first. We recommend using `conda install conda-forge::xtb`.")

    with tempfile.TemporaryDirectory() as tmpdir:

        xyz_file = Path(tmpdir) / "mol.xyz"

        Chem.MolToXYZFile(mol, xyz_file)
        
        cmd = f"xtb {xyz_file} --gfn 2 --opt normal --cycles 50 --alpb water --chrg {charge} --threads {num_cores}"
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(num_cores),
            "MKL_NUM_THREADS": str(num_cores),
            "OPENBLAS_NUM_THREADS": str(num_cores),
            "NUMEXPR_NUM_THREADS": str(num_cores),
            "VECLIB_MAXIMUM_THREADS": str(num_cores),
            "OMP_MAX_ACTIVE_LEVELS": str(num_cores),
            "OMP_STACKSIZE": "1G",
            "XTBPATH": tmpdir,
            "XTBHOME": tmpdir
        })

        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
            shell=True
        )
        # Use a regex to match "TOTAL ENERGY" followed by whitespace and a number
        total_energy_match = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", proc.stdout)
        if total_energy_match:
            total_energy = float(total_energy_match.group(1))
            total_energy *= 627.509  # Eh -> kcal/mol
        else:
            raise ValueError(f"Could not find Gsolv in xTB output:\n{proc.returncode}\n{proc.stdout}\n{proc.stderr}")



        # Regex to extract Gsolv in Eh
        gsolv_match = re.search(r"->\s*Gsolv\s*([-+]?\d*\.\d+|\d+)\s*Eh", proc.stdout)
        if gsolv_match:
            gsolv_hartree = float(gsolv_match.group(1))
            solvation_energy = gsolv_hartree * 627.509  # Eh -> kcal/mol
        else:
            raise ValueError(f"Could not find Gsolv in xTB output:\n{proc.returncode}\n{proc.stdout}\n{proc.stderr}")

        return total_energy, solvation_energy
    

def get_solvation_energy(mol: Chem.Mol, /, *, num_rdkit_confs: int = 50, rmsd_threshold: float | None = 0.25) -> float:
    mol = embed(Chem.AddHs(mol), num_confs=num_rdkit_confs, rmsd_threshold=rmsd_threshold)

    solvation_energies = []
    total_energies = []
    for conf in mol.GetConformers():
        _mol = Chem.Mol(mol)
        _mol.RemoveAllConformers()
        _mol.AddConformer(Chem.Conformer(conf), assignId=True)
        total_energy, solvation_energy = get_xtb_solvation(_mol,num_cores=1)
        total_energies.append(total_energy)
        solvation_energies.append(solvation_energy)

    solvation_energies = np.array(solvation_energies)
    total_energies = np.array(total_energies)

    solvation_energy, _ = get_boltzmann_weighted_average(properties=solvation_energies, energies=total_energies)

    return solvation_energy



