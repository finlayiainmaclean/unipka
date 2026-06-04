import copy
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Geometry import Point3D
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

def get_coordinates(mol: Chem.Mol, conf_id: int | None = 0) -> np.ndarray:
    """Get coordinates from molecular conformer."""
    conf_id = int(conf_id)
    coords = mol.GetConformer(conf_id).GetPositions()
    return coords

def set_coordinates(mol: Chem.Mol, coords: np.ndarray, conf_id: int = 0):
    """Overwrite (or create) conformer with supplied Cartesian coordinates.

    Overwrites (or creates) conformer `conf_id` with the supplied
    Cartesian coordinates (Å).  `coords` must be shape (N_atoms, 3).
    """
    conf_id = int(conf_id)
    n_atoms = mol.GetNumAtoms()
    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords shape {coords.shape} does not match atom count {n_atoms}")

    # make sure the conformer exists
    if not has_conformer(mol, conf_id=conf_id):
        conf = Chem.Conformer(n_atoms)
        conf.SetId(conf_id)
        mol.AddConformer(conf)

    conf = mol.GetConformer(conf_id)
    for i in range(mol.GetNumAtoms()):
        x,y,z = coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))


def mmff_optimise(
    mol: Chem.Mol, constrained_atom_idxs: list[int] | None = None
) -> tuple[Chem.Mol, float | None]:
    """Optimize molecular geometry using MMFF force field."""
    # Define a force field with constraints on non-hydrogen atoms
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if properties is None:
        logger.warning(f"MMFF properties could not be initialized for molecule {Chem.MolToSmiles(mol)} (possibly contains unsupported atoms like transition metals). Skipping MMFF optimization.")
        return mol, None

    ff = AllChem.MMFFGetMoleculeForceField(mol, properties)
    if ff is None:
        logger.warning(f"MMFF force field could not be initialized for molecule {Chem.MolToSmiles(mol)} (possibly contains unsupported atoms like transition metals). Skipping MMFF optimization.")
        return mol, None

    if constrained_atom_idxs is not None:
        for atom in mol.GetAtoms():
            if atom.GetIdx() in constrained_atom_idxs:
                ff.AddFixedPoint(atom.GetIdx())
    try:
        ff.Minimize()
    except Exception:
        logger.warning(f"Failed to minimize molecule {Chem.MolToSmiles(mol)}")
    # Get the optimized structure and energy
    energy = ff.CalcEnergy()
    return mol, energy

def has_conformer(mol: Chem.Mol, /, *, conf_id: int):
    conformer_ids = {conf.GetId() for conf in mol.GetConformers()}
    return conf_id in conformer_ids


def _prepare_for_match(mol: Chem.Mol) -> Chem.Mol:
    mol = copy.deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    return mol


def _ref_atom_map_for_query(ref_noh: Chem.Mol, query_noh: Chem.Mol) -> tuple[int, ...]:
    """Map each query heavy-atom index to the matching reference heavy-atom index."""
    ref_copy = _prepare_for_match(ref_noh)
    query_copy = _prepare_for_match(query_noh)

    match = ref_copy.GetSubstructMatch(query_copy)
    if len(match) == ref_noh.GetNumAtoms():
        return match

    mcs = rdFMCS.FindMCS(
        [ref_copy, query_copy],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
    )
    if mcs.numAtoms != ref_noh.GetNumAtoms():
        raise ValueError(
            f"MCS matched {mcs.numAtoms} atoms, expected {ref_noh.GetNumAtoms()} "
            f"for {Chem.MolToSmiles(query_noh)}"
        )

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        raise ValueError(f"Failed to build MCS query for {Chem.MolToSmiles(query_noh)}")

    ref_match = ref_copy.GetSubstructMatch(mcs_mol)
    query_match = query_copy.GetSubstructMatch(mcs_mol)
    if len(ref_match) != ref_noh.GetNumAtoms() or len(query_match) != query_noh.GetNumAtoms():
        raise ValueError(
            f"MCS substructure match incomplete for {Chem.MolToSmiles(query_noh)}"
        )

    logger.debug(
        "Substructure match failed; mapped coordinates with MCS for %s",
        Chem.MolToSmiles(query_noh),
    )

    mapping: list[int | None] = [None] * query_noh.GetNumAtoms()
    for ref_idx, query_idx in zip(ref_match, query_match, strict=True):
        mapping[query_idx] = ref_idx
    if any(idx is None for idx in mapping):
        raise ValueError(f"MCS atom mapping incomplete for {Chem.MolToSmiles(query_noh)}")
    return tuple(mapping)


def transplant_coordinates(ref: Chem.Mol, query: Chem.Mol) -> Chem.Mol:
    """Transplant coordinates from reference molecule to query molecule."""
    DISTANCE_THRESHOLD = 0.25

    query_noh = Chem.RemoveHs(query)
    ref_noh = Chem.RemoveHs(ref)

    if ref_noh.GetNumAtoms() != query_noh.GetNumAtoms():
        raise ValueError(
            f"Atom count mismatch: ref has {ref_noh.GetNumAtoms()}, "
            f"query has {query_noh.GetNumAtoms()}"
        )

    match = _ref_atom_map_for_query(ref_noh, query_noh)
    coords = get_coordinates(ref_noh)
    set_coordinates(query_noh, coords[np.array(match)])  # Set coords of heavy atoms
    query = Chem.AddHs(query_noh, addCoords=True)  # Add any missing hydrogens

    query_coords = get_coordinates(query)  # with explicit hydrogens
    ref_coords = get_coordinates(ref)  # with explicit hydrogens

    dist = cdist(query_coords, ref_coords)
    q_ix, r_ix = zip(*np.argwhere(dist < DISTANCE_THRESHOLD).astype(int), strict=False)
    q_ix = np.array(q_ix)
    r_ix = np.array(r_ix)

    query_coords[q_ix] = ref_coords[r_ix]  # Replace coords of any atom under distance threshold

    set_coordinates(query, query_coords)

    query, _ = mmff_optimise(query, constrained_atom_idxs=q_ix)
    return query