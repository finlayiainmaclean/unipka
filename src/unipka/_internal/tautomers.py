"""RDKit tautomer enumeration helpers."""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def _canon_key(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def tautomers_of_mol(
    mol: Chem.Mol, *, max_tautomers: Optional[int] = None
) -> list[Chem.Mol]:
    """
    Return distinct tautomeric forms of ``mol`` (canonical isomeric SMILES deduped),
    including the input tautomer when enumeration succeeds. On failure, returns ``[mol]``.
    """
    m = Chem.Mol(mol)
    try:
        te = rdMolStandardize.TautomerEnumerator()
        if max_tautomers is not None:
            te.SetMaxTautomers(max_tautomers)
        res = te.Enumerate(m)
    except Exception:
        return [Chem.Mol(mol)]

    out: dict[str, Chem.Mol] = {}
    for t in res:
        try:
            c = Chem.RemoveHs(Chem.Mol(t))
            k = _canon_key(c)
            out.setdefault(k, c)
        except Exception:
            continue
    return list(out.values()) if out else [Chem.Mol(mol)]


def tautomer_seeds_at_formal_charge(
    mol: Chem.Mol,
    q_target: int,
    *,
    expand: bool,
    max_tautomers: Optional[int] = None,
) -> list[Chem.Mol]:
    """If ``expand``, all tautomers matching ``q_target``; otherwise a single copy of ``mol``."""
    if not expand:
        return [Chem.Mol(mol)]
    seeds = tautomers_of_mol(mol, max_tautomers=max_tautomers)
    same_q = [Chem.Mol(x) for x in seeds if Chem.GetFormalCharge(x) == q_target]
    return same_q if same_q else [Chem.Mol(mol)]
