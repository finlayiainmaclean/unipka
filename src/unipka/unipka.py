from collections import defaultdict, deque
import logging
import math
import os
import sys
import warnings
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ._internal.solvation import get_solvation_energy as _get_solvation_energy
from ._internal.draw import calc_base_name, draw_ensemble, get_neutral_base_name
from ._internal.conformer import ConformerGen
from ._internal.dataset import MolDataset
from ._internal.model import UniMolModel
from ._internal.template import (
    FILTER_PATTERNS,
    LN10,
    TRANSLATE_PH,
    _band_from_mols,
    _band_merge,
    _enumerate_template_mols,
    _mol_canonical_key,
    enumerate_template,
    log_sum_exp,
    prot,
    read_template,
)
from ._internal.coordinates import transplant_coordinates
from ._internal.widget import Widget

from .assets import get_model_path, get_pattern_path, load_kpuu_model


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol_free_energy.inference")

R = 8.314  # J/mol/K


class EnumerationError(Exception):
    pass


def _same_mol(mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
    inchi_options = "/FixedH"
    inchi1 = Chem.MolToInchiKey(mol1, options=inchi_options)
    inchi2 = Chem.MolToInchiKey(mol2, options=inchi_options)
    
    return inchi1==inchi2


def validate_acid_base_pair(acid_macrostate, base_macrostate):
    """
    Validate that acid and base macrostates have consistent hydrogen counts.
    Raises ValueError if validation fails.
    
    Parameters:
    -----------
    acid_smiles_list : list
        List of SMILES for acid macrostate
    base_smiles_list : list
        List of SMILES for base macrostate
    """
    def count_hydrogens(smiles):
        """Count total hydrogens in a molecule from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add explicit hydrogens to get accurate count
        mol_with_h = Chem.AddHs(mol)
        return sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == 'H')
        
    
    # Count hydrogens in all acid species
    acid_h_counts = [count_hydrogens(smi) for smi in acid_macrostate]
    base_h_counts = [count_hydrogens(smi) for smi in base_macrostate]
    
    # Check 1: All acid species have same number of hydrogens
    acid_unique_counts = set(acid_h_counts)
    if len(acid_unique_counts) != 1:
        acid_counts_str = ", ".join([f"{smi}: {h}H" for smi, h in zip(acid_macrostate, acid_h_counts)])
        raise ValueError(f"Acid species have different hydrogen counts: {acid_counts_str}")
    
    # Check 2: All base species have same number of hydrogens  
    base_unique_counts = set(base_h_counts)
    if len(base_unique_counts) != 1:
        base_counts_str = ", ".join([f"{smi}: {h}H" for smi, h in zip(base_macrostate, base_h_counts)])
        raise ValueError(f"Base species have different hydrogen counts: {base_counts_str}")
    
    # Check 3: Acid has exactly one more hydrogen than base
    acid_h = acid_h_counts[0]
    base_h = base_h_counts[0] 
    
    if acid_h != base_h + 1:
        raise ValueError(f"Acid should have 1 more hydrogen than base. "
                        f"Got acid: {acid_h}H, base: {base_h}H (difference: {acid_h - base_h})")


class UnipKa(object):
    def __init__(
        self,
        batch_size=32,
        remove_hs=False,
        use_simple_smarts: bool = False,
        ensemble_energy_prune_window: float = 15.,
    ):
        """
        :param ensemble_energy_prune_window: :meth:`_predict_ensemble_free_energy` grows the
            protonation ensemble stepwise, predicts energies after each expansion, and drops
            microstates whose **pH-adjusted** free energy
            (``DfG_m + q * LN10 * (pH - TRANSLATE_PH)``, same form as in :meth:`get_distribution`)
            is more than this value above the current global minimum among **already predicted**
            microstates (same numeric units as ``_predict`` outputs for ``DfG_m``). The pH used is
            the one passed into that method (e.g. ``get_distribution(..., pH=...)``), defaulting to
            7.4 when omitted.
            This is a **heuristic**: high-energy intermediates are removed before later enumeration,
            so low-energy states reachable only through them may be missed.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = get_model_path()
        pattern_path = get_pattern_path(use_simple_smarts=use_simple_smarts)

        self.model = UniMolModel(model_path, output_dim=1, remove_hs=remove_hs).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.params = {"remove_hs": remove_hs}
        self.conformer_gen = ConformerGen(**self.params)
        self.template_a2b, self.template_b2a = read_template(pattern_path)
        self.ensemble_energy_prune_window = ensemble_energy_prune_window


    #### Internal functions ####
    @staticmethod
    def _get_formal_charge(mol):
        """
        Calculate the sum of formal charges on all atoms in the molecule.
        This represents the total formal charge of the microstate.
        """
        if mol is None:
            return float("inf")  # Invalid molecule

        formal_charges = []
        for atom in mol.GetAtoms():
            formal_charges.append(atom.GetFormalCharge())

        abs_formal_charge = np.abs(np.sum(formal_charges))
        abs_atoms_charges = np.sum([abs(charge) for charge in formal_charges])
        return abs_formal_charge, abs_atoms_charges

    @staticmethod
    def _get_distribution_from_free_energy(
        ensemble_free_energy: dict[int, list[tuple[str, Chem.Mol, float]]], /, *, pH: float
    ) -> pd.DataFrame:
        ensemble_boltzmann_factor = defaultdict(list)
        partition_function = 0
        for q, macrostate_free_energy in ensemble_free_energy.items():
            for microstate_smi, microstate_mol, DfGm in macrostate_free_energy:
                boltzmann_factor = math.exp(-DfGm - q * LN10 * (pH - TRANSLATE_PH))
                partition_function += boltzmann_factor
                ensemble_boltzmann_factor[q].append(
                    (microstate_smi, microstate_mol, boltzmann_factor)
                )

        fractions = []
        smiles_list = []
        mols_list = []
        charges = []

        for q, macrostate_boltzmann_factor in ensemble_boltzmann_factor.items():
            for microstate_smi, microstate_mol, boltzmann_factor in macrostate_boltzmann_factor:
                fraction = boltzmann_factor / partition_function
                fractions.append(fraction)
                smiles_list.append(microstate_smi)
                mols_list.append(Chem.Mol(microstate_mol))
                charges.append(q)

        return pd.DataFrame(
            {
                "population": fractions,
                "smiles": smiles_list,
                "mol": mols_list,
                "charge": charges,
            }
        )
    
    def _preprocess_data(self, mols_or_smis: list[str] | list[Chem.Mol]):
        return self.conformer_gen.transform(mols_or_smis)

    @staticmethod
    def _as_mol_list(mols: str | Chem.Mol | list[str] | list[Chem.Mol]) -> tuple[list[Chem.Mol], list[str]]:
        if isinstance(mols, Chem.Mol):
            mol_list = [mols]
        elif isinstance(mols, str):
            mol_list = [Chem.MolFromSmiles(mols)]
        elif isinstance(mols, list):
            if not mols:
                raise ValueError("_predict requires a non-empty list")
            if isinstance(mols[0], Chem.Mol):
                mol_list = list(mols)
            else:
                mol_list = [Chem.MolFromSmiles(s) for s in mols]
      
        smiles = [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in mol_list]
        return mol_list, smiles

    def _predict(self, mols: str | Chem.Mol | list[str] | list[Chem.Mol]):
        mol_list, smiles = self._as_mol_list(mols)

        unimol_input = self._preprocess_data(mol_list)
        dataset = MolDataset(unimol_input)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.model.batch_collate_fn,
        )

        energies: list[float] = []

        for batch in dataloader:
            net_input, _ = self._decorate_torch_batch(batch)
            with torch.no_grad():
                predictions = self.model(**net_input)
                energies.extend(e.item() for e in predictions)

        if len(energies) != len(smiles):
            raise RuntimeError(
                f"Number of predictions ({len(energies)}) "
                f"does not match number of inputs ({len(smiles)})"
            )

        return dict(zip(smiles, energies, strict=True))


    def _decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {"net_input": net_input.to(self.device)}, net_target.to(self.device)
        net_target = None

        return net_input, net_target

    def _predict_micro_pKa(self, mol: Chem.Mol | str, /, *, idx: int, mode: Literal["a2b", "b2a"]):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        new_mol = Chem.RemoveHs(prot(mol, idx, mode))
        if mode == "a2b":
            mol_a, mol_b = mol, new_mol
        else:
            mol_b, mol_a = mol, new_mol
        DfGm = self._predict([mol_a, mol_b])
        key_a = Chem.MolToSmiles(mol_a, canonical=True, isomericSmiles=True)
        key_b = Chem.MolToSmiles(mol_b, canonical=True, isomericSmiles=True)
        return (DfGm[key_b] - DfGm[key_a]) / LN10 + TRANSLATE_PH

    def _predict_macro_pKa(self, mol: Chem.Mol | str, /, *, mode: Literal["a2b", "b2a"]) -> float:

        if isinstance(mol, Chem.Mol):
            smi = Chem.MolToSmiles(mol)
        else:
            smi = mol
        
        macrostate_A, macrostate_B = enumerate_template(smi, self.template_a2b, self.template_b2a, mode)
        if len(macrostate_A)==0 or len(macrostate_B)==0:
            return np.nan
        DfGm_A = self._predict(macrostate_A)
        DfGm_B = self._predict(macrostate_B)
        return log_sum_exp(DfGm_A.values()) - log_sum_exp(DfGm_B.values()) + TRANSLATE_PH

    def _predict_pending_pruned_bands(
        self,
        ensemble: dict[int, dict[str, Chem.Mol]],
        g_cache: dict[str, float],
        q_cache: dict[str, int],
    ) -> None:
        pending: list[Chem.Mol] = []
        pending_q: list[int] = []
        for q, band in ensemble.items():
            for k, m in band.items():
                if k not in g_cache:
                    pending.append(m)
                    pending_q.append(q)
        if not pending:
            return
        pred = self._predict(pending)
        for m, q in zip(pending, pending_q, strict=True):
            sk = _mol_canonical_key(m)
            if sk in pred:
                g_cache[sk] = pred[sk]
                q_cache[sk] = q

    @staticmethod
    def _ph_adjusted_free_energy(DfGm: float, q: int, pH: float) -> float:
        return DfGm + q * LN10 * (pH - TRANSLATE_PH)

    def _prune_ensemble_bands(
        self,
        ensemble: dict[int, dict[str, Chem.Mol]],
        g_cache: dict[str, float],
        q_cache: dict[str, int],
        window: float,
        pH: float,
    ) -> None:
        if not g_cache:
            return
        cutoff = min(self._ph_adjusted_free_energy(g_cache[k], q_cache[k], pH) for k in g_cache) + window
        for q in list(ensemble.keys()):
            band = ensemble[q]
            for k in list(band.keys()):
                g = g_cache.get(k)
                if g is None or k not in q_cache:
                    continue
                if self._ph_adjusted_free_energy(g, q, pH) > cutoff:
                    del band[k]
            if not band:
                del ensemble[q]

    def _get_ensemble_pruned(
        self, mol: Chem.Mol, energy_window: float, pH: float, maxiter: int = 10
    ) -> tuple[dict[int, list[Chem.Mol]], dict[str, float]]:
        """
        Same charge-ladder expansion as ``get_ensemble``, but after each merge step we predict
        new microstates and drop those more than ``energy_window`` above the current global minimum
        **pH-adjusted** free energy (``DfG_m + q * LN10 * (pH - TRANSLATE_PH)``)
        among microstates seen so far.

        If pruning removes every microstate at the query formal charge, the effective window is
        doubled and the expansion is restarted from the input molecule (up to a fixed number of
        relaxations), instead of failing immediately.
        """
        w = float(energy_window)
        if w <= 0:
            w = 1.0
        max_prune_relaxations = 64
        n_relax = 0
        ta, tb = self.template_a2b, self.template_b2a

        while True:
            mol0 = Chem.Mol(mol)
            q0 = Chem.GetFormalCharge(mol0)
            ensemble: dict[int, dict[str, Chem.Mol]] = {q0: _band_from_mols([mol0])}
            g_cache: dict[str, float] = {}
            q_cache: dict[str, int] = {}

            def predict_and_prune() -> None:
                self._predict_pending_pruned_bands(ensemble, g_cache, q_cache)
                self._prune_ensemble_bands(ensemble, g_cache, q_cache, w, pH)

            predict_and_prune()
            if q0 not in ensemble or not ensemble[q0]:
                n_relax += 1
                if n_relax > max_prune_relaxations:
                    raise EnumerationError(
                        "Pruning removed all microstates at the reference charge even after "
                        f"widening the energy window up to {w:g}. "
                        "Try a larger ensemble_energy_prune_window."
                    )
                w *= 2.0
                logger.debug(
                    "Pruning emptied reference charge; retrying with prune window %s (relaxation %s)",
                    w,
                    n_relax,
                )
                continue

            m0_out, m_b1 = _enumerate_template_mols(
                list(ensemble[q0].values()), ta, tb, "a2b", maxiter, 0, FILTER_PATTERNS
            )
            ensemble[q0] = _band_from_mols(m0_out)
            if m_b1:
                ensemble[q0 - 1] = _band_from_mols(m_b1)
            predict_and_prune()

            visited_a2b: set[int] = {q0}
            down_queue: deque[int] = deque()
            if q0 - 1 in ensemble and ensemble[q0 - 1]:
                down_queue.append(q0 - 1)
            n_down = 0
            while down_queue and n_down < maxiter:
                q_src = down_queue.popleft()
                if q_src in visited_a2b:
                    continue
                band = ensemble.get(q_src)
                if not band:
                    continue
                visited_a2b.add(q_src)
                n_down += 1
                _, m_b = _enumerate_template_mols(list(band.values()), ta, tb, "a2b", maxiter, 0, FILTER_PATTERNS)
                if not m_b:
                    continue
                q_dst = q_src - 1
                _band_merge(ensemble.setdefault(q_dst, {}), m_b)
                down_queue.append(q_dst)
                predict_and_prune()

            if q0 not in ensemble or not ensemble[q0]:
                n_relax += 1
                if n_relax > max_prune_relaxations:
                    raise EnumerationError(
                        "Pruning removed all microstates at the reference charge before the b2a pass, "
                        f"even after widening the energy window up to {w:g}. "
                        "Try a larger ensemble_energy_prune_window."
                    )
                w *= 2.0
                logger.debug(
                    "Pruning emptied reference charge before b2a; retrying with prune window %s (relaxation %s)",
                    w,
                    n_relax,
                )
                continue

            m_a1, m0_b2a = _enumerate_template_mols(
                list(ensemble[q0].values()), ta, tb, "b2a", maxiter, 0, FILTER_PATTERNS
            )
            ensemble[q0] = _band_from_mols(m0_b2a)
            visited_b2a: set[int] = {q0}
            up_queue: deque[int] = deque()
            if m_a1:
                ensemble[q0 + 1] = _band_from_mols(m_a1)
                up_queue.append(q0 + 1)
            predict_and_prune()

            n_up = 0
            while up_queue and n_up < maxiter:
                q_src = up_queue.popleft()
                if q_src in visited_b2a:
                    continue
                band = ensemble.get(q_src)
                if not band:
                    continue
                visited_b2a.add(q_src)
                n_up += 1
                m_a, _ = _enumerate_template_mols(list(band.values()), ta, tb, "b2a", maxiter, 0, FILTER_PATTERNS)
                if not m_a:
                    continue
                q_dst = q_src + 1
                _band_merge(ensemble.setdefault(q_dst, {}), m_a)
                up_queue.append(q_dst)
                predict_and_prune()

            out = {q: sorted(band.values(), key=_mol_canonical_key) for q, band in ensemble.items() if band}
            return out, g_cache

    def _predict_ensemble_free_energy(
        self,
        mol: Chem.Mol,
        *,
        prune_window: Optional[float] = None,
        pH: float = 7.4,
    ) -> tuple[dict[int, list[Chem.Mol]], dict[int, list[tuple[str, Chem.Mol, float]]]]:
        """``pH`` is used for pH-adjusted pruning only; returned ``DfG_m`` values are unchanged."""
        window = self.ensemble_energy_prune_window if prune_window is None else prune_window
        query_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        w = float(window)
        max_charge_relaxations = 32
        ensemble: dict[int, list[Chem.Mol]]
        g_cache: dict[str, float]
        for relax in range(max_charge_relaxations + 1):
            ensemble, g_cache = self._get_ensemble_pruned(mol, w, pH)
            if len(ensemble.keys()) >= 2:
                break
            if relax == max_charge_relaxations:
                raise EnumerationError(
                    f"Failed to enumerate microstates across 2 charge states for {query_smi}. "
                    "Try enumerating manually and calling `get_macro_pka_from_macrostates`, "
                    "or a larger ensemble_energy_prune_window for this pH."
                )
            w *= 2.0
            logger.debug(
                "pH-adjusted pruning left fewer than 2 charge states; retrying with window %s (relaxation %s)",
                w,
                relax + 1,
            )

        ensemble_free_energy: dict[int, list[tuple[str, Chem.Mol, float]]] = {}
        for q, macrostate in ensemble.items():
            row: list[tuple[str, Chem.Mol, float]] = []
            for m in macrostate:
                s = _mol_canonical_key(m)
                if s not in g_cache:
                    logger.warning("Missing cached energy for %s at charge %s after pruned ensemble", s, q)
                    continue
                row.append((s, Chem.Mol(m), g_cache[s]))
            ensemble_free_energy[q] = row

        if len(ensemble_free_energy) == 0:
            raise ValueError("Could not process any microstates")
        return ensemble, ensemble_free_energy

    #### Public functions ####
    def get_macro_pka_from_macrostates(self, *, acid_macrostate: list[str | Chem.Mol], base_macrostate: list[str | Chem.Mol]) -> float:

        
        
        
        if isinstance(acid_macrostate[0], Chem.Mol):
            acid_macrostate = [Chem.MolToSmiles(mol) for mol in acid_macrostate]

        if isinstance(base_macrostate[0], Chem.Mol):
            base_macrostate = [Chem.MolToSmiles(mol) for mol in base_macrostate]

        validate_acid_base_pair(acid_macrostate=acid_macrostate, base_macrostate=base_macrostate)


        DfGm_A = self._predict(acid_macrostate)
        DfGm_B = self._predict(base_macrostate)
        return log_sum_exp(DfGm_A.values()) - log_sum_exp(DfGm_B.values()) + TRANSLATE_PH

    def get_acidic_macro_pka(self, mol: Chem.Mol | str, /) -> float:
        return self._predict_macro_pKa(mol, mode="a2b")

    def get_basic_macro_pka(self, mol: Chem.Mol | str, /) -> float:
        return self._predict_macro_pKa(mol, mode="b2a")

    def get_acidic_micro_pka(self, mol: Chem.Mol | str, /, *, idx: int) -> float:
        return self._predict_micro_pKa(mol, mode="a2b", idx=idx)

    def get_basic_micro_pka(self, mol: Chem.Mol | str, /, *, idx: int) -> float:
        return self._predict_micro_pKa(mol, mode="b2a", idx=idx)
    
    def get_dominant_microstate(self, mol: Chem.Mol | str, /, *, pH: float) -> Chem.Mol:

        df = self.get_distribution(mol, pH=pH)
        protomer_mol =  df.iloc[0].mol
        return protomer_mol

    def draw_distribution(self, mol: Chem.Mol | str, /, mode: Literal["matplotlib", "jupyter"] = "matplotlib") -> pd.DataFrame:

        
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)


        # Free energy predictions from your model, grouped by charge
        ensemble, ensemble_free_energy = self._predict_ensemble_free_energy(mol)

        pHs = np.linspace(0, 14, 1000)
        fractions = defaultdict(list)
        name_mapping = dict()
        neutral_base_name = get_neutral_base_name(ensemble)

        for q, macrostate in ensemble_free_energy.items():
            for i, (microstate_smi, _microstate_mol, _dg) in enumerate(macrostate):
                name_mapping[microstate_smi] = f"{i+1}-{calc_base_name(neutral_base_name, q)}"
        distribution_dfs = []
        for pH in pHs:
            distribution_df = self._get_distribution_from_free_energy(ensemble_free_energy, pH=pH)
            distribution_df['pH'] = pH
            distribution_dfs.append(distribution_df)
            for _, row in distribution_df.iterrows():
                microstate = row['smiles']
                fraction = row['population']
                fractions[name_mapping[microstate]].append(fraction)

        distribution_df = pd.concat(distribution_dfs)
        distribution_df['name'] = distribution_df.smiles.apply(name_mapping.get)


        match mode:
            case "jupyter":
                return Widget(distribution_df)

            case "matplotlib":
                plt.figure(figsize=(14, 3), dpi=200)
                for base_name, fraction_curve in fractions.items():
                    plt.plot(pHs, fraction_curve, label=base_name.replace("<sub>", "$_{").replace("</sub>", "}$").replace("<sup>", "$^{").replace("</sup>", "}$"))
                plt.xlabel("pH")
                plt.ylabel("fraction")
                plt.legend()
                plt.show()
                draw_ensemble(ensemble)
            case _:
                raise ValueError(f"{mode} not a vaid mode. Choose from `matplotlib` and `jupyter`")
                
    def get_distribution(self, mol: Chem.Mol | str, /, *, pH: float = 7.4) -> pd.DataFrame:

        
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)


        # Free energy predictions from your model, grouped by charge
        _, ensemble_free_energy = self._predict_ensemble_free_energy(mol, pH=pH)

        records = []
        partition_function = 0.0

        # Collect Boltzmann weights and energy terms
        for q, macrostate_free_energy in ensemble_free_energy.items():
            for microstate_smi, microstate_mol, DfGm in macrostate_free_energy:
                G_pH = self._ph_adjusted_free_energy(DfGm, q, pH)
                boltzmann_factor = math.exp(-G_pH)
                records.append((q, microstate_smi, microstate_mol, DfGm, G_pH, boltzmann_factor))
                partition_function += boltzmann_factor

        # Normalize to get population w_i(pH)
        df = pd.DataFrame(
            records,
            columns=["charge", "smiles", "mol", "free_energy", "ph_adjusted_free_energy", "boltzmann_factor"],
        )

        df["relative_ph_adjusted_free_energy"] = df.ph_adjusted_free_energy - df.ph_adjusted_free_energy.min()
        df["relative_free_energy"] = df.free_energy - df.free_energy.min()
        df["population"] = df["boltzmann_factor"] / partition_function

        # Sort for readability
        df = df.sort_values(by="population", ascending=False).reset_index(drop=True)

        if mol.GetNumConformers() > 0:
            df["mol"] = df["mol"].apply(lambda x: transplant_coordinates(mol, x))
        df["is_query_mol"] = df["mol"].apply(lambda x: _same_mol(mol, x))

        return df

    def get_state_penalty(self, mol: Chem.Mol | str, /, *, T: float = 298.15, pH: float = 7.4) -> float:
        """
        Calculate the state penalty (SP) according to the Lawrenz concept.

        Selects formally neutral microstates that minimize atom-centered charges,
        preferring non-zwitterionic forms over zwitterionic counterparts.
        """
        
        df = self.get_distribution(mol, pH=pH)

        # Calculate formal charges for all molecules
        charge_results = df["mol"].apply(self._get_formal_charge)
        df["abs_formal_charge"] = [result[0] for result in charge_results]
        df["abs_atoms_charges"] = [result[1] for result in charge_results]

        # Step 1: Find microstates with minimum absolute formal charge (preferably 0)
        min_abs_formal_charge = df["abs_formal_charge"].min()
        neutral_candidates = df[df["abs_formal_charge"] == min_abs_formal_charge].copy()

        if min_abs_formal_charge == 0:
            # Step 2: Among neutral microstates, prefer those with minimum atom-centered charges
            # This favors non-zwitterionic forms over zwitterionic forms
            min_atom_charges = neutral_candidates["abs_atoms_charges"].min()
            reference_microstates_df = neutral_candidates[
                neutral_candidates["abs_atoms_charges"] == min_atom_charges
            ].copy()
        else:
            # No truly neutral forms exist - use microstates with minimum formal charge
            # Among these, still prefer those with minimum atom-centered charges
            min_atom_charges = neutral_candidates["abs_atoms_charges"].min()
            reference_microstates_df = neutral_candidates[
                neutral_candidates["abs_atoms_charges"] == min_atom_charges
            ].copy()

        # Sort reference microstates by population for inspection
        reference_microstates_df = reference_microstates_df.sort_values(by="population", ascending=False).reset_index(
            drop=True
        )

        # Calculate sum of reference microstate populations
        sum_reference_pop = reference_microstates_df["population"].sum()

        if sum_reference_pop <= 0:
            raise ValueError("Error: No population in reference microstates!")

        if sum_reference_pop < 1e-10:
            raise ValueError(
                f"Warning: Very low reference population ({sum_reference_pop:.2e}). State penalty may be unreliable."
            )

        # Calculate state penalty: SP = -RT * ln(sum of reference populations)
        SP_J_mol = -R * T * math.log(sum_reference_pop)
        SP_kcal_mol = SP_J_mol / 4184  # Convert to kcal/mol

        return SP_kcal_mol, reference_microstates_df
    
    @staticmethod
    def get_solvation_energy(mol: Chem.Mol | str, /) -> float:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return _get_solvation_energy(mol)
    
    def predict_brain_penetrance(self, mol: Chem.Mol) -> float:
        sp, ref_df = self.get_state_penalty(mol, pH=7.4)
        mol = ref_df.iloc[0].mol
        G_solv = _get_solvation_energy(mol)
        logD = self.get_logd(mol, pH=7.4)
        clf = load_kpuu_model()
        X= np.array([[G_solv, logD, sp]])
        return clf.predict_proba(X)[0,1]
    
    def get_logd(self, mol: Chem.Mol | str, /, *,  pH: float) -> float:
        """
        Compute logD(pH) from microstate populations and logP values.

        Parameters:
        - df: DataFrame output from compute_microstate_populations_at_pH, must contain:
            - 'mol': RDKit Mol object
            - 'charge': formal charge
            - 'population': w_i(pH)

        Returns:
        - logD (float): pH-dependent distribution coefficient
        """

        

        df = self.get_distribution(mol, pH=pH)

        logP_list = []
        weighted_linear_logP = []

        for _, row in df.iterrows():
            mol = row["mol"]
            charge = row["charge"]
            pop = row["population"]

            # logP for neutral species
            if charge == 0:
                logP = Crippen.MolLogP(mol)
            else:
                logP = -2.0  # fixed logP for ionic species

            logP_list.append(logP)
            weighted_linear_logP.append(pop * (10**logP))

        # Compute logD from weighted sum in linear space
        logd = np.log10(sum(weighted_linear_logP))

        # Optional: include in the DataFrame if you want to return it
        df["logP"] = logP_list
        df["weighted_linear_logP"] = weighted_linear_logP

        return logd
    
    def draw_logd_distribution(self, mol: Chem.Mol | str, /, mode: Literal["matplotlib"] = "matplotlib") -> pd.DataFrame:
        """
        Draw logD distribution across pH range.
        
        Parameters:
        - mol: RDKit Mol object or SMILES string
        - mode: Plotting mode ("matplotlib")
        
        Returns:
        - DataFrame containing pH and logD values
        """
        
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        # Free energy predictions from your model, grouped by charge
        _, ensemble_free_energy = self._predict_ensemble_free_energy(mol)

        pHs = np.linspace(0, 14, 1000)
   
        distribution_dfs = []
        logd_values = []

        logp_cache = dict()
        
        for pH in pHs:
            distribution_df = self._get_distribution_from_free_energy(ensemble_free_energy, pH=pH)
            distribution_df['pH'] = pH
            distribution_dfs.append(distribution_df)
            
            # Calculate logD for this pH
            logP_list = []
            weighted_linear_logP = []

            for _, row in distribution_df.iterrows():
                charge = row["charge"]
                pop = row["population"]
                smi_microstate = row['smiles']

                # logP for neutral species
                if charge == 0:
                    logP = logp_cache.get(smi_microstate)
                    if not logP:
                        mol_microstate = Chem.MolFromSmiles(smi_microstate)
                        logP = Crippen.MolLogP(mol_microstate)
                        logp_cache[smi_microstate] = logP
                else:
                    logP = -2.0  # fixed logP for ionic species

                logP_list.append(logP)
                weighted_linear_logP.append(pop * (10**logP))

            # Compute logD from weighted sum in linear space
            logd = np.log10(sum(weighted_linear_logP))
            logd_values.append(logd)
        

        match mode:
            case "matplotlib":
                plt.figure(figsize=(14, 3), dpi=200)
                plt.plot(pHs, logd_values, linewidth=2, color='blue', label='logD')
                plt.xlabel("pH")
                plt.ylabel("logD")
                plt.grid(True, alpha=0.3)
                plt.show()
            case _:
                raise ValueError(f"{mode} not a valid mode. Choose from `matplotlib`")
        

    

