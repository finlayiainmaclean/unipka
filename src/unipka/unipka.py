from collections import defaultdict
import logging
import math
import os
import sys
import warnings
from typing import Dict, Literal, Tuple

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
from ._internal.template import LN10, TRANSLATE_PH, enumerate_template, get_ensemble, log_sum_exp, prot, read_template
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
logging.disable(50)

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
    def __init__(self, batch_size=32, remove_hs=False, use_simple_smarts: bool = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = get_model_path()
        pattern_path = get_pattern_path(use_simple_smarts=use_simple_smarts)

        self.model = UniMolModel(model_path, output_dim=1, remove_hs=remove_hs).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.params = {"remove_hs": remove_hs}
        self.conformer_gen = ConformerGen(**self.params)
        self.template_a2b, self.template_b2a = read_template(pattern_path)


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
    def _get_distribution_from_free_energy(ensemble_free_energy: Dict[int, Dict[str, float]], / , *, pH: float) -> pd.DataFrame:
        ensemble_boltzmann_factor = defaultdict(list)
        partition_function = 0
        for q, macrostate_free_energy in ensemble_free_energy.items():
            for microstate, DfGm in macrostate_free_energy:
                boltzmann_factor = math.exp(-DfGm - q * LN10 * (pH - TRANSLATE_PH))
                partition_function += boltzmann_factor
                ensemble_boltzmann_factor[q].append((microstate, boltzmann_factor))
        
        # Create lists for DataFrame columns
        fractions = []
        microstates = []
        charges = []
        
        for q, macrostate_boltzmann_factor in ensemble_boltzmann_factor.items():
            for microstate, boltzmann_factor in macrostate_boltzmann_factor:
                fraction = boltzmann_factor / partition_function
                fractions.append(fraction)
                microstates.append(microstate)
                charges.append(q)
        
        return pd.DataFrame({
            'population': fractions,
            'smiles': microstates,
            'charge': charges
        })
    
    def _preprocess_data(self, smiles_list):
        inputs = self.conformer_gen.transform(smiles_list)
        return inputs

    def _predict(self, smiles: list[str] | str):
        if isinstance(smiles, str):
            smiles = [smiles]
        unimol_input = self._preprocess_data(smiles)
        dataset = MolDataset(unimol_input)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.model.batch_collate_fn,
        )

        results = {}
        for batch in dataloader:
            net_input, _ = self._decorate_torch_batch(batch)
            with torch.no_grad():
                predictions = self.model(**net_input)
                for smiles, energy in zip(smiles, predictions):
                    results[smiles] = energy.item()
        return results

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

        if isinstance(mol, Chem.Mol):
            smi = Chem.MolToSmiles(mol)
        else:
            smi = mol

        mol = Chem.MolFromSmiles(smi)
        new_mol = Chem.RemoveHs(prot(mol, idx, mode))
        new_smi = Chem.MolToSmiles(new_mol)
        if mode == "a2b":
            smi_A = smi
            smi_B = new_smi
        elif mode == "b2a":
            smi_B = smi
            smi_A = new_smi
        DfGm = self._predict([smi_A, smi_B])
        pKa = (DfGm[smi_B] - DfGm[smi_A]) / LN10 + TRANSLATE_PH
        return pKa

    def _predict_macro_pKa(self, mol: Chem.Mol | str, /, *, mode: Literal["a2b", "b2a"]) -> float:

        if isinstance(mol, Chem.Mol):
            smi = Chem.MolToSmiles(mol)
        else:
            smi = mol
        
        macrostate_A, macrostate_B = enumerate_template(smi, self.template_a2b, self.template_b2a, mode)
        DfGm_A = self._predict(macrostate_A)
        DfGm_B = self._predict(macrostate_B)
        return log_sum_exp(DfGm_A.values()) - log_sum_exp(DfGm_B.values()) + TRANSLATE_PH


    def _predict_ensemble_free_energy(self, smi: str) -> Dict[int, Tuple[str, float]]:
        
        ensemble = get_ensemble(smi, self.template_a2b, self.template_b2a)

        if len(ensemble.keys())<2:
                raise EnumerationError(f"Failed to enumerate microstates across 2 charge states for {smi}. "
                                       "Try enumerating manually and calling `get_macro_pka_from_macrostates`")

        
        ensemble_free_energy = dict()
        for q, macrostate in ensemble.items():
            prediction = self._predict(macrostate)
            _ensemble_free_energy = []
            for microstate in macrostate:
                if microstate in prediction:
                    _ensemble_free_energy.append((microstate, prediction[microstate]))
            ensemble_free_energy[q] = _ensemble_free_energy

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

        query_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))

        # Free energy predictions from your model, grouped by charge
        ensemble, ensemble_free_energy = self._predict_ensemble_free_energy(query_smi)


        pHs = np.linspace(0, 14, 1000)
        fractions = defaultdict(list)
        name_mapping = dict()
        neutral_base_name = get_neutral_base_name(ensemble_free_energy)

        for q, macrostate in ensemble_free_energy.items():
            for i, (microstate, _) in enumerate(macrostate):
                name_mapping[microstate] = f"{i+1}-{calc_base_name(neutral_base_name, q)}"
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

        query_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))

        # Free energy predictions from your model, grouped by charge
        _, ensemble_free_energy = self._predict_ensemble_free_energy(query_smi)

        records = []
        partition_function = 0.0

        # Collect Boltzmann weights and energy terms
        for q, macrostate_free_energy in ensemble_free_energy.items():
            for microstate_smi, DfGm in macrostate_free_energy:
                G_pH = DfGm + q * LN10 * (pH - TRANSLATE_PH)  # pH-adjusted free energy
                boltzmann_factor = math.exp(-G_pH)
                records.append((q, microstate_smi, DfGm, G_pH, boltzmann_factor))
                partition_function += boltzmann_factor

        # Normalize to get population w_i(pH)
        df = pd.DataFrame(
            records, columns=["charge", "smiles", "free_energy", "ph_adjusted_free_energy", "boltzmann_factor"]
        )

        df["relative_ph_adjusted_free_energy"] = df.ph_adjusted_free_energy - df.ph_adjusted_free_energy.min()
        df["relative_free_energy"] = df.free_energy - df.free_energy.min()
        df["population"] = df["boltzmann_factor"] / partition_function

        # Sort for readability
        df = df.sort_values(by="population", ascending=False).reset_index(drop=True)

        # Optional: add mol objects and coordinate mapping
        df["mol"] = df.smiles.apply(Chem.MolFromSmiles)

        if mol.GetNumConformers() > 0:
            df["mol"] = df.mol.apply(lambda x: transplant_coordinates(mol, x))  # if you have this
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

        query_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))

        # Free energy predictions from your model, grouped by charge
        _, ensemble_free_energy = self._predict_ensemble_free_energy(query_smi)

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
        

    

