import logging
import math
import os
import sys
import warnings
from collections import defaultdict, deque
from typing import Any, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from torch.utils.data import DataLoader

from ._internal.conformer import ConformerGen, UnimolFeatures
from ._internal.coordinates import transplant_coordinates
from ._internal.dataset import MolDataset
from ._internal.draw import calc_base_name, draw_ensemble, get_neutral_base_name
from ._internal.model import UniMolModel
from ._internal.solvation import get_solvation_energy as _get_solvation_energy
from ._internal.tautomers import tautomer_seeds_at_formal_charge
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
    prot_template_mol,
    read_template,
    sanitize_filter_mols,
    stereo_filter_mols,
)
from ._internal.temporal_activity import heartbeat
from ._internal.widget import Widget
from .assets import get_model_path, get_pattern_path, load_kpuu_model

RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]
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
    """Raised when microstate enumeration fails for a molecule."""


def _same_mol(mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
    inchi_options = "/FixedH"
    inchi1 = Chem.MolToInchiKey(mol1, options=inchi_options)
    inchi2 = Chem.MolToInchiKey(mol2, options=inchi_options)

    return inchi1 == inchi2


def validate_acid_base_pair(
    acid_macrostate: list[str], base_macrostate: list[str]
) -> None:
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

    def count_hydrogens(smiles: str) -> int:
        """Count total hydrogens in a molecule from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Add explicit hydrogens to get accurate count
        mol_with_h = Chem.AddHs(mol)
        return sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == "H")

    # Count hydrogens in all acid species
    acid_h_counts = [count_hydrogens(smi) for smi in acid_macrostate]
    base_h_counts = [count_hydrogens(smi) for smi in base_macrostate]

    # Check 1: All acid species have same number of hydrogens
    acid_unique_counts = set(acid_h_counts)
    if len(acid_unique_counts) != 1:
        acid_counts_str = ", ".join(
            [f"{smi}: {h}H" for smi, h in zip(acid_macrostate, acid_h_counts)]
        )
        raise ValueError(
            f"Acid species have different hydrogen counts: {acid_counts_str}"
        )

    # Check 2: All base species have same number of hydrogens
    base_unique_counts = set(base_h_counts)
    if len(base_unique_counts) != 1:
        base_counts_str = ", ".join(
            [f"{smi}: {h}H" for smi, h in zip(base_macrostate, base_h_counts)]
        )
        raise ValueError(
            f"Base species have different hydrogen counts: {base_counts_str}"
        )

    # Check 3: Acid has exactly one more hydrogen than base
    acid_h = acid_h_counts[0]
    base_h = base_h_counts[0]

    if acid_h != base_h + 1:
        raise ValueError(
            f"Acid should have 1 more hydrogen than base. "
            f"Got acid: {acid_h}H, base: {base_h}H (difference: {acid_h - base_h})"
        )


class UnipKa:
    """Wrapper around the UnipKa model for pKa, logD, and microstate calculations."""

    def __init__(
        self,
        batch_size: int = 32,
        remove_hs: bool = False,
        use_simple_smarts: bool = True,
        beam_width: Optional[int] = 20,
        charge_limits: Optional[Tuple[int, int]] = (-2, 2),
        enumerate_tautomers: bool = False,
    ) -> None:
        """
        :param ensemble_beam_width: After each scoring step, at each formal charge ``q`` the
            merged pool is reduced to at most this many microstates with lowest model ``DfG_m``
            (``None`` keeps every microstate that passed template enumeration and charge clipping).

        :param ensemble_formal_charge_limits: ``(q_min, q_max)`` inclusive bounds on total
            molecular formal charge during enumeration. The interval is widened if needed so the
            input molecule's charge is always included. ``None`` disables this clipping (legacy
            behaviour: only ``maxiter`` bounds the charge ladder).
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = get_model_path()
        pattern_path = get_pattern_path(use_simple_smarts=use_simple_smarts)

        self.model = UniMolModel(str(model_path), output_dim=1, remove_hs=remove_hs).to(
            self.device
        )
        self.model.eval()
        self.batch_size = batch_size
        self.params = {"remove_hs": remove_hs}
        self.conformer_gen = ConformerGen(**self.params)
        self.template_a2b, self.template_b2a = read_template(pattern_path)
        self.beam_width = beam_width
        self.charge_limits = charge_limits
        self.enumerate_tautomers = enumerate_tautomers

    #### Internal functions ####
    @staticmethod
    def _get_formal_charge(mol: Chem.Mol | None) -> tuple[float, float]:
        """
        Calculate the sum of formal charges on all atoms in the molecule.
        This represents the total formal charge of the microstate.
        """
        if mol is None:
            return float("inf"), float("inf")  # Invalid molecule

        formal_charges = []
        for atom in mol.GetAtoms():
            formal_charges.append(atom.GetFormalCharge())

        abs_formal_charge = np.abs(np.sum(formal_charges))
        abs_atoms_charges = np.sum([abs(charge) for charge in formal_charges])
        return abs_formal_charge, abs_atoms_charges

    @staticmethod
    def _get_distribution_from_free_energy(
        ensemble_free_energy: dict[int, list[tuple[str, Chem.Mol, float]]],
        /,
        *,
        pH: float,
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
            for (
                microstate_smi,
                microstate_mol,
                boltzmann_factor,
            ) in macrostate_boltzmann_factor:
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

    def _preprocess_data(
        self, mols_or_smis: list[str] | list[Chem.Mol]
    ) -> list[UnimolFeatures]:
        return self.conformer_gen.transform(mols_or_smis)

    @staticmethod
    def _as_mol_list(
        mols: str | Chem.Mol | list[str] | list[Chem.Mol],
    ) -> tuple[list[Chem.Mol], list[str]]:
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

        smiles = [
            Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in mol_list
        ]
        return mol_list, smiles

    def _predict(
        self, mols: str | Chem.Mol | list[str] | list[Chem.Mol]
    ) -> dict[str, float]:
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
            heartbeat()
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

    def _decorate_torch_batch(
        self, batch: tuple[Any, Any]
    ) -> tuple[dict[str, Any], Any | None]:
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = (
                {k: v.to(self.device) for k, v in net_input.items()},
                net_target.to(self.device),
            )
        else:
            net_input, net_target = (
                {"net_input": net_input.to(self.device)},
                net_target.to(self.device),
            )
        net_target = None

        return net_input, net_target

    def _predict_micro_pKa(
        self, mol: Chem.Mol | str, /, *, idx: int, mode: Literal["a2b", "b2a"]
    ) -> float:
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

    def _predict_macro_pKa(
        self, mol: Chem.Mol | str, /, *, mode: Literal["a2b", "b2a"]
    ) -> float:

        if isinstance(mol, Chem.Mol):
            smi = Chem.MolToSmiles(mol)
        else:
            smi = mol

        macrostate_A, macrostate_B = enumerate_template(
            smi, self.template_a2b, self.template_b2a, mode
        )
        if len(macrostate_A) == 0 or len(macrostate_B) == 0:
            return np.nan
        DfGm_A = self._predict(macrostate_A)
        DfGm_B = self._predict(macrostate_B)
        return (
            log_sum_exp(list(DfGm_A.values()))
            - log_sum_exp(list(DfGm_B.values()))
            + TRANSLATE_PH
        )

    def _predict_pending_pruned_bands(
        self,
        ensemble: dict[int, dict[str, Chem.Mol]],
        g_cache: dict[str, float],
        q_cache: dict[str, int],
    ) -> None:
        pending: list[Chem.Mol] = []
        pending_q: list[int] = []
        for q, band in ensemble.items():
            heartbeat()
            for k, m in band.items():
                heartbeat()
                if k not in g_cache:
                    pending.append(m)
                    pending_q.append(q)
        if not pending:
            return
        heartbeat()
        pred = self._predict(pending)
        for m, q in zip(pending, pending_q, strict=True):
            heartbeat()
            sk = _mol_canonical_key(m)
            if sk in pred:
                g_cache[sk] = pred[sk]
                q_cache[sk] = q

    @staticmethod
    def _ph_adjusted_free_energy(DfGm: float, q: int, pH: float) -> float:
        return DfGm + q * LN10 * (pH - TRANSLATE_PH)

    @staticmethod
    def _effective_formal_charge_limits(
        q0: int, limits: Tuple[int, int]
    ) -> tuple[int, int]:
        lo, hi = limits
        if lo > hi:
            lo, hi = hi, lo
        return min(lo, q0), max(hi, q0)

    @staticmethod
    def _clip_ensemble_charge_range(
        ensemble: dict[int, dict[str, Chem.Mol]], q_lo: int, q_hi: int
    ) -> None:
        for q in list(ensemble.keys()):
            if q < q_lo or q > q_hi:
                del ensemble[q]

    def _beam_prune_ensemble_bands(
        self,
        ensemble: dict[int, dict[str, Chem.Mol]],
        g_cache: dict[str, float],
        beam_width: int,
    ) -> None:
        """Keep at most ``beam_width`` lowest-``DfG_m`` microstates per formal charge (paper beam)."""
        if beam_width <= 0:
            return
        for q in list(ensemble.keys()):
            heartbeat()
            band = ensemble[q]
            scored = [(k, g_cache[k]) for k in band if k in g_cache]
            unscored = [k for k in band if k not in g_cache]
            scored.sort(key=lambda kv: kv[1])
            keep = {k for k, _ in scored[:beam_width]}
            keep.update(unscored)
            for k in list(band.keys()):
                if k not in keep:
                    del band[k]
            if not band:
                del ensemble[q]

    def _get_ensemble_pruned(
        self, mol: Chem.Mol, maxiter: int = 10
    ) -> tuple[dict[int, list[Chem.Mol]], dict[str, float]]:
        """
        Stepwise beam search enumeration within a formal-charge window.

        Each global iteration:
        - expands every current charge-state beam by one protonation step and one deprotonation
          step (template-driven edits);
        - scores newly seen microstates with the model;
        - prunes each charge-state beam to at most ``beam_width`` lowest-``DfG_m`` microstates.

        Terminates when (i) no new states are generated, (ii) all beams become stationary, or
        (iii) ``maxiter`` iterations have executed.
        """
        ta, tb = self.template_a2b, self.template_b2a
        mol0 = Chem.RemoveHs(Chem.Mol(mol))
        q0 = Chem.GetFormalCharge(mol0)

        lim = self.charge_limits
        if lim is None:
            q_lo, q_hi = -(10**9), 10**9
        else:
            q_lo, q_hi = self._effective_formal_charge_limits(q0, lim)

        heartbeat()
        from ._internal.tautomers import tautomer_seeds_at_formal_charge

        seeds = tautomer_seeds_at_formal_charge(
            mol0, q0, expand=self.enumerate_tautomers
        )
        ensemble: dict[int, dict[str, Chem.Mol]] = {
            q0: {_mol_canonical_key(m): m for m in seeds}
        }
        visited: set[str] = set(ensemble[q0].keys())
        g_cache: dict[str, float] = {}
        q_cache: dict[str, int] = {k: q0 for k in ensemble[q0]}

        def predict_and_prune() -> None:
            heartbeat()
            self._predict_pending_pruned_bands(ensemble, g_cache, q_cache)
            bw = self.beam_width
            if bw is not None:
                self._beam_prune_ensemble_bands(ensemble, g_cache, bw)
            if lim is not None:
                self._clip_ensemble_charge_range(ensemble, q_lo, q_hi)

        def expand_from_band(
            m: Chem.Mol, mode: Literal["a2b", "b2a"]
        ) -> list[Chem.Mol]:
            # One-step neighbor generation via SMARTS templates.
            heartbeat()
            template = ta if mode == "a2b" else tb
            _sites, products = prot_template_mol(template, m, mode)
            products = sanitize_filter_mols(products, FILTER_PATTERNS)
            products = stereo_filter_mols(products)
            for _prod in products:
                heartbeat()
            return products

        prev_sig: dict[int, frozenset[str]] = {}

        for _it in range(maxiter):
            heartbeat()
            predict_and_prune()

            sig = {q: frozenset(band.keys()) for q, band in ensemble.items()}
            if prev_sig and all(
                sig.get(q) == prev_sig.get(q) for q in set(sig) | set(prev_sig)
            ):
                break  # (ii) beams stationary
            prev_sig = sig

            new_by_q: dict[int, dict[str, Chem.Mol]] = defaultdict(dict)
            n_new = 0

            # Snapshot current beams so we don't expand newly added states in the same iteration.
            current = {q: list(band.values()) for q, band in ensemble.items()}
            for q, mols in current.items():
                if not mols:
                    continue
                heartbeat()

                # deprotonation: q -> q-1
                q_down = q - 1
                if q_lo <= q_down <= q_hi:
                    for m in mols:
                        heartbeat()
                        for prod in expand_from_band(m, "a2b"):
                            k = _mol_canonical_key(prod)
                            if k in visited:
                                continue
                            visited.add(k)
                            new_by_q[q_down][k] = prod
                            q_cache[k] = q_down
                            n_new += 1

                # protonation: q -> q+1
                q_up = q + 1
                if q_lo <= q_up <= q_hi:
                    for m in mols:
                        heartbeat()
                        for prod in expand_from_band(m, "b2a"):
                            k = _mol_canonical_key(prod)
                            if k in visited:
                                continue
                            visited.add(k)
                            new_by_q[q_up][k] = prod
                            q_cache[k] = q_up
                            n_new += 1

            if n_new == 0:
                break  # (i) no new states generated

            for q, band in new_by_q.items():
                heartbeat()
                dest = ensemble.setdefault(q, {})
                for k, m in band.items():
                    dest.setdefault(k, m)

        heartbeat()
        predict_and_prune()
        out = {
            q: sorted(band.values(), key=_mol_canonical_key)
            for q, band in ensemble.items()
            if band
        }
        return out, g_cache

    def _get_ensemble_unpruned(
        self, mol: Chem.Mol, maxiter: int = 10
    ) -> tuple[dict[int, list[Chem.Mol]], dict[str, float]]:
        """
        Charge-ladder expansion like :meth:`_get_ensemble_pruned`, but without incremental
        scoring or per-charge beam pruning (full enumeration, single batched ``_predict`` at the end).

        Each ``_enumerate_template_mols`` call already stops when pools stop growing or after
        ``maxiter`` inner iterations. The downward and upward BFS each visit at most ``maxiter``
        new formal-charge bands; expansion stops when queues are empty (no new microstates).

        Formal-charge bounds match :meth:`_get_ensemble_pruned` (``ensemble_formal_charge_limits``,
        widened to include the input charge).
        """
        ta, tb = self.template_a2b, self.template_b2a
        mol0 = Chem.Mol(mol)
        q0 = Chem.GetFormalCharge(mol0)
        lim = self.charge_limits
        if lim is None:
            q_lo, q_hi = -(10**9), 10**9
        else:
            q_lo, q_hi = self._effective_formal_charge_limits(q0, lim)
        heartbeat()
        seeds = tautomer_seeds_at_formal_charge(
            mol0, q0, expand=self.enumerate_tautomers
        )
        ensemble: dict[int, dict[str, Chem.Mol]] = {q0: _band_from_mols(seeds)}

        heartbeat()
        m0_out, m_b1 = _enumerate_template_mols(
            list(ensemble[q0].values()), ta, tb, "a2b", maxiter, 0, FILTER_PATTERNS
        )
        ensemble[q0] = _band_from_mols(m0_out)
        if m_b1 and q_lo <= (q0 - 1) <= q_hi:
            ensemble[q0 - 1] = _band_from_mols(m_b1)

        visited_a2b: set[int] = {q0}
        down_queue: deque[int] = deque()
        if q_lo <= (q0 - 1) <= q_hi and q0 - 1 in ensemble and ensemble[q0 - 1]:
            down_queue.append(q0 - 1)
        n_down = 0
        while down_queue and n_down < maxiter:
            heartbeat()
            q_src = down_queue.popleft()
            if q_src in visited_a2b:
                continue
            band = ensemble.get(q_src)
            if not band:
                continue
            visited_a2b.add(q_src)
            n_down += 1
            heartbeat()
            _, m_b = _enumerate_template_mols(
                list(band.values()), ta, tb, "a2b", maxiter, 0, FILTER_PATTERNS
            )
            if not m_b:
                continue
            q_dst = q_src - 1
            if q_dst < q_lo:
                continue
            _band_merge(ensemble.setdefault(q_dst, {}), m_b)
            down_queue.append(q_dst)

        heartbeat()
        m_a1, m0_b2a = _enumerate_template_mols(
            list(ensemble[q0].values()), ta, tb, "b2a", maxiter, 0, FILTER_PATTERNS
        )
        ensemble[q0] = _band_from_mols(m0_b2a)
        visited_b2a: set[int] = {q0}
        up_queue: deque[int] = deque()
        if m_a1 and q_lo <= (q0 + 1) <= q_hi:
            ensemble[q0 + 1] = _band_from_mols(m_a1)
            up_queue.append(q0 + 1)

        n_up = 0
        while up_queue and n_up < maxiter:
            heartbeat()
            q_src = up_queue.popleft()
            if q_src in visited_b2a:
                continue
            band = ensemble.get(q_src)
            if not band:
                continue
            visited_b2a.add(q_src)
            n_up += 1
            heartbeat()
            m_a, _ = _enumerate_template_mols(
                list(band.values()), ta, tb, "b2a", maxiter, 0, FILTER_PATTERNS
            )
            if not m_a:
                continue
            q_dst = q_src + 1
            if q_dst > q_hi:
                continue
            _band_merge(ensemble.setdefault(q_dst, {}), m_a)
            up_queue.append(q_dst)

        out = {
            q: sorted(band.values(), key=_mol_canonical_key)
            for q, band in ensemble.items()
            if band
        }
        all_mols = [m for macrostate in out.values() for m in macrostate]
        heartbeat()
        pred = self._predict(all_mols) if all_mols else {}
        g_cache: dict[str, float] = {}
        for m in all_mols:
            sk = _mol_canonical_key(m)
            if sk in pred:
                g_cache[sk] = pred[sk]
        return out, g_cache

    def _predict_ensemble_free_energy(
        self,
        mol: Chem.Mol,
    ) -> tuple[dict[int, list[Chem.Mol]], dict[int, list[tuple[str, Chem.Mol, float]]]]:
        """Enumerate microstates with per-charge beam pruning; returned ``DfG_m`` values are raw model outputs."""
        query_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        heartbeat()

        ensemble: dict[int, list[Chem.Mol]]
        g_cache: dict[str, float]
        try:
            ensemble, g_cache = self._get_ensemble_pruned(mol)
            heartbeat()
        except EnumerationError as e:
            logger.warning(f"Enumeration error: {e}. Retrying with unpruned ensemble.")
            heartbeat()
            ensemble, g_cache = self._get_ensemble_unpruned(mol)
            heartbeat()

        if len(ensemble.keys()) < 2:
            raise EnumerationError(
                f"Failed to enumerate microstates across 2 charge states for {query_smi}. "
                "Try widening ``ensemble_formal_charge_limits``, using ``ensemble_beam_width=None``, "
                "or enumerating manually and calling ``get_macro_pka_from_macrostates``."
            )

        ensemble_free_energy: dict[int, list[tuple[str, Chem.Mol, float]]] = {}
        for q, macrostate in ensemble.items():
            heartbeat()
            row: list[tuple[str, Chem.Mol, float]] = []
            for m in macrostate:
                s = _mol_canonical_key(m)
                if s not in g_cache:
                    logger.warning(
                        "Missing cached energy for %s at charge %s after pruned ensemble",
                        s,
                        q,
                    )
                    continue
                row.append((s, Chem.Mol(m), g_cache[s]))
            ensemble_free_energy[q] = row

        if len(ensemble_free_energy) == 0:
            raise ValueError("Could not process any microstates")
        return ensemble, ensemble_free_energy

    #### Public functions ####
    def get_macro_pka_from_macrostates(
        self,
        *,
        acid_macrostate: list[str | Chem.Mol],
        base_macrostate: list[str | Chem.Mol],
    ) -> float:
        """Compute macroscopic pKa from explicit acid and base macrostate ensembles."""

        if isinstance(acid_macrostate[0], Chem.Mol):
            acid_macrostate = [Chem.MolToSmiles(mol) for mol in acid_macrostate]

        if isinstance(base_macrostate[0], Chem.Mol):
            base_macrostate = [Chem.MolToSmiles(mol) for mol in base_macrostate]

        validate_acid_base_pair(
            acid_macrostate=acid_macrostate, base_macrostate=base_macrostate
        )

        DfGm_A = self._predict(acid_macrostate)
        DfGm_B = self._predict(base_macrostate)
        return (
            log_sum_exp(list(DfGm_A.values()))
            - log_sum_exp(list(DfGm_B.values()))
            + TRANSLATE_PH
        )

    def get_acidic_macro_pka(self, mol: Chem.Mol | str, /) -> float:
        """Return the acidic macroscopic pKa for a molecule."""
        return self._predict_macro_pKa(mol, mode="a2b")

    def get_basic_macro_pka(self, mol: Chem.Mol | str, /) -> float:
        """Return the basic macroscopic pKa for a molecule."""
        return self._predict_macro_pKa(mol, mode="b2a")

    def get_acidic_micro_pka(self, mol: Chem.Mol | str, /, *, idx: int) -> float:
        """Return the acidic micro-pKa for the ionizable site at ``idx``."""
        return self._predict_micro_pKa(mol, mode="a2b", idx=idx)

    def get_basic_micro_pka(self, mol: Chem.Mol | str, /, *, idx: int) -> float:
        """Return the basic micro-pKa for the ionizable site at ``idx``."""
        return self._predict_micro_pKa(mol, mode="b2a", idx=idx)

    def get_dominant_microstate(self, mol: Chem.Mol | str, /, *, pH: float) -> Chem.Mol:
        """Return the most populated microstate at the given pH."""

        df = self.get_distribution(mol, pH=pH)
        return df.iloc[0].mol

    def draw_distribution(
        self,
        mol: Chem.Mol | str,
        /,
        mode: Literal["matplotlib", "jupyter"] = "matplotlib",
    ) -> pd.DataFrame:
        """Plot or display the microstate distribution across pH."""

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
                name_mapping[microstate_smi] = (
                    f"{i + 1}-{calc_base_name(neutral_base_name, q)}"
                )
        distribution_dfs = []
        for pH in pHs:
            distribution_df = self._get_distribution_from_free_energy(
                ensemble_free_energy, pH=pH
            )
            distribution_df["pH"] = pH
            distribution_dfs.append(distribution_df)
            for _, row in distribution_df.iterrows():
                microstate = row["smiles"]
                fraction = row["population"]
                fractions[name_mapping[microstate]].append(fraction)

        distribution_df = pd.concat(distribution_dfs)
        distribution_df["name"] = distribution_df.smiles.apply(name_mapping.get)

        match mode:
            case "jupyter":
                return Widget(distribution_df)

            case "matplotlib":
                plt.figure(figsize=(14, 3), dpi=200)
                for base_name, fraction_curve in fractions.items():
                    plt.plot(
                        pHs,
                        fraction_curve,
                        label=base_name.replace("<sub>", "$_{")
                        .replace("</sub>", "}$")
                        .replace("<sup>", "$^{")
                        .replace("</sup>", "}$"),
                    )
                plt.xlabel("pH")
                plt.ylabel("fraction")
                plt.legend()
                plt.show()
                draw_ensemble(ensemble)
                return distribution_df
            case _:
                raise ValueError(
                    f"{mode} not a vaid mode. Choose from `matplotlib` and `jupyter`"
                )

    def get_distribution(
        self, mol: Chem.Mol | str, /, *, pH: float | list[float] = 7.4
    ) -> pd.DataFrame:
        """Return microstate populations and free energies at one or more pH values."""

        if isinstance(pH, (int, float)):
            pHs = [pH]
        else:
            pHs = pH
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        # Free energy predictions from your model, grouped by charge
        _, ensemble_free_energy = self._predict_ensemble_free_energy(mol)

        dfs = []

        for pH in pHs:
            records = []
            partition_function = 0.0

            # Collect Boltzmann weights and energy terms
            for q, macrostate_free_energy in ensemble_free_energy.items():
                for microstate_smi, microstate_mol, DfGm in macrostate_free_energy:
                    G_pH = self._ph_adjusted_free_energy(DfGm, q, pH)
                    boltzmann_factor = math.exp(-G_pH)
                    records.append(
                        (
                            q,
                            microstate_smi,
                            microstate_mol,
                            DfGm,
                            G_pH,
                            boltzmann_factor,
                            pH,
                        )
                    )
                    partition_function += boltzmann_factor

            # Normalize to get population w_i(pH)
            df = pd.DataFrame(
                records,
                columns=[
                    "charge",
                    "smiles",
                    "mol",
                    "free_energy",
                    "ph_adjusted_free_energy",
                    "boltzmann_factor",
                    "ph",
                ],
            )

            df["relative_ph_adjusted_free_energy"] = (
                df.ph_adjusted_free_energy - df.ph_adjusted_free_energy.min()
            )
            df["relative_free_energy"] = df.free_energy - df.free_energy.min()
            df["population"] = df["boltzmann_factor"] / partition_function

            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # Sort for readability
        df = df.sort_values(by=["ph", "population"], ascending=False).reset_index(
            drop=True
        )

        if mol.GetNumConformers() > 0:
            df["mol"] = df["mol"].apply(lambda x: transplant_coordinates(mol, x))
        df["is_query_mol"] = df["mol"].apply(lambda x: _same_mol(mol, x))

        return df

    def get_state_penalty(
        self, mol: Chem.Mol | str, /, *, T: float = 298.15, pH: float = 7.4
    ) -> tuple[float, pd.DataFrame]:
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
        reference_microstates_df = reference_microstates_df.sort_values(
            by="population", ascending=False
        ).reset_index(drop=True)

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
        """Return the aqueous solvation free energy for a molecule."""
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return _get_solvation_energy(mol)

    def predict_brain_penetrance(self, mol: Chem.Mol) -> float:
        """Predict blood-brain barrier penetration probability using the Kp,uu model."""
        sp, ref_df = self.get_state_penalty(mol, pH=7.4)
        mol = ref_df.iloc[0].mol
        G_solv = _get_solvation_energy(mol)
        logD = self.get_logd(mol, pH=7.4)
        clf = load_kpuu_model()
        X = np.array([[G_solv, logD, sp]])
        return clf.predict_proba(X)[0, 1]

    def get_logd(self, mol: Chem.Mol | str, /, *, pH: float) -> float:
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

    def draw_logd_distribution(
        self, mol: Chem.Mol | str, /, mode: Literal["matplotlib"] = "matplotlib"
    ) -> pd.DataFrame:
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
            distribution_df = self._get_distribution_from_free_energy(
                ensemble_free_energy, pH=pH
            )
            distribution_df["pH"] = pH
            distribution_dfs.append(distribution_df)

            # Calculate logD for this pH
            logP_list = []
            weighted_linear_logP = []

            for _, row in distribution_df.iterrows():
                charge = row["charge"]
                pop = row["population"]
                smi_microstate = row["smiles"]

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
                plt.plot(pHs, logd_values, linewidth=2, color="blue", label="logD")
                plt.xlabel("pH")
                plt.ylabel("logD")
                plt.grid(True, alpha=0.3)
                plt.show()
        return pd.DataFrame({"pH": pHs, "logD": logd_values})


if __name__ == "__main__":
    smi = "C#Cc1cc2c(N(C)Cc3ccc(-c4c(C(=O)O)cnn4C)cc3)ncnc2cc1NCc1ccc(O)cc1N1CCNCC1"
    calc = UnipKa()
    calc.get_distribution(smi)
