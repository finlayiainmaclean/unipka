import math
from collections import deque
from typing import Callable, Dict, List, OrderedDict, Sequence, Tuple, Union

import pandas as pd
from rdkit import Chem

from .temporal_activity import heartbeat


class ProtonationError(Exception):
    pass


FILTER_PATTERNS = list(
    map(
        Chem.MolFromSmarts,
        [
            "[#6X5]",
            "[#7X5]",
            "[#8X4]",
            "[*r]=[*r]=[*r]",
            "[#1]-[*+1]~[*-1]",
            "[#1]-[*+1]=,:[*]-,:[*-1]",
            "[#1]-[*+1]-,:[*]=,:[*-1]",
            "[*+2]",
            "[*-2]",
            "[#1]-[#8+1].[#8-1,#7-1,#6-1]",
            "[#1]-[#7+1,#8+1].[#7-1,#6-1]",
            "[#1]-[#8+1].[#8-1,#6-1]",
            "[#1]-[#7+1].[#8-1]-[C](-[C,#1])(-[C,#1])",
            # "[#6;!$([#6]-,:[*]=,:[*]);!$([#6]-,:[#7,#8,#16])]=[C](-[O,N,S]-[#1])",
            # "[#6]-,=[C](-[O,N,S])(-[O,N,S]-[#1])",
            "[OX1]=[C]-[OH2+1]",
            "[NX1,NX2H1,NX3H2]=[C]-[O]-[H]",
            "[#6-1]=[*]-[*]",
            "[cX2-1]",
            "[N+1](=O)-[O]-[H]",
        ],
    )
)
LN10 = math.log(10)
TRANSLATE_PH = 6.504894871171601

# Compiled SMARTS for template rows (avoids MolFromSmarts per match per molecule).
_SMARTS_PATTERN_CACHE: Dict[str, Chem.Mol] = {}


def _mol_from_smarts_cached(smarts: str) -> Chem.Mol:
    mol = _SMARTS_PATTERN_CACHE.get(smarts)
    if mol is None:
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts!r}")
        _SMARTS_PATTERN_CACHE[smarts] = mol
    return mol


def match_template(template: pd.DataFrame, mol: Chem.Mol, verbose: bool = False) -> list:
    """
    Find protonation site using templates

    Params:
    ----
    `template`: `pandas.Dataframe` with columns of substructure names, SMARTS patterns, protonation indices and acid/base flags

    `mol`: Molecule

    `verbose`: Boolean flag for printing matching results

    Return:
    ----
    A set of matched indices to be (de)protonated
    """
    mol = Chem.AddHs(mol)
    matches = []
    for idx, name, smarts, index, acid_base in template.itertuples():
        pattern = _mol_from_smarts_cached(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        else:
            index = int(index)
            for m in match:
                matches.append(m[index])
                if verbose:
                    print(f"find index {m[index]} in pattern {name} smarts {smarts}")
    return list(set(matches))


def _mol_canonical_key(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _dedup_mols(mols: Sequence[Chem.Mol]) -> List[Chem.Mol]:
    d: Dict[str, Chem.Mol] = {}
    for m in mols:
        d.setdefault(_mol_canonical_key(m), m)
    return list(d.values())


def _smis_to_mols(smi: Union[str, List[str]]) -> List[Chem.Mol]:
    if isinstance(smi, str):
        return [Chem.MolFromSmiles(smi)]
    return [Chem.MolFromSmiles(s) for s in smi]


def _mols_to_canonical_smiles(mols: Sequence[Chem.Mol]) -> List[str]:
    return [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in mols]


def prot_template_mol(template: pd.DataFrame, mol: Chem.Mol, mode: str) -> Tuple[List[int], List[Chem.Mol]]:
    sites = match_template(template, mol)
    products: Dict[str, Chem.Mol] = {}
    for si, site in enumerate(sites):
        heartbeat({"prot_template_mol": "site", "mode": mode, "si": si, "n_sites": len(sites)})
        pm = prot(mol, site, mode)
        if pm is None:
            continue
        rh = Chem.RemoveHs(pm)
        products.setdefault(_mol_canonical_key(rh), rh)
    return sites, list(products.values())


def prot_template(template: pd.DataFrame, smi: str, mode: str) -> Tuple[List[int], List[str]]:
    """
    Protonate / Deprotonate a SMILES at every found site in the template

    Params:
    ----
    `template`: `pandas.Dataframe` with columns of substructure names, SMARTS patterns, protonation indices and acid/base flags

    `smi`: The SMILES to be processed

    `mode`: `a2b` means deprotonization, with a hydrogen atom or a heavy atom at `idx`; `b2a` means protonization, with a heavy atom at `idx`
    """
    mol = Chem.MolFromSmiles(smi)
    sites, mols = prot_template_mol(template, mol, mode)
    return sites, _mols_to_canonical_smiles(mols)


def cnt_stereo_atom_mol(mol: Chem.Mol) -> int:
    return sum(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms())


def cnt_stereo_atom(smi: str) -> int:
    """
    Count the stereo atoms in a SMILES
    """
    return cnt_stereo_atom_mol(Chem.MolFromSmiles(smi))


def stereo_filter_mols(mols: List[Chem.Mol]) -> List[Chem.Mol]:
    filtered: Dict[str, Tuple[Chem.Mol, int]] = {}
    for mol in mols:
        key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        stereo_cnt = cnt_stereo_atom_mol(mol)
        if key not in filtered:
            filtered[key] = (mol, stereo_cnt)
        elif stereo_cnt > filtered[key][1]:
            filtered[key] = (mol, stereo_cnt)
    return [t[0] for t in filtered.values()]


def stereo_filter(smis: List[str]) -> List[str]:
    """
    A filter against SMILES losing stereochemical information in structure processing.
    """
    mols = [Chem.MolFromSmiles(s) for s in smis]
    return _mols_to_canonical_smiles(stereo_filter_mols(mols))


def sanitize_checker_mol(mol: Chem.Mol, filter_patterns: List[Chem.Mol], verbose: bool = False) -> bool:
    m = Chem.AddHs(Chem.Mol(mol))
    for pattern in filter_patterns:
        match = m.GetSubstructMatches(pattern)
        if match:
            if verbose:
                print(f"pattern {pattern}")
            return False
    try:
        Chem.SanitizeMol(m)
    except Exception as e:
        print("cannot sanitize", repr(e))
        return False
    return True


def sanitize_checker(smi: str, filter_patterns: List[Chem.Mol], verbose: bool = False) -> bool:
    """
    Check if a SMILES can be sanitized and does not contain unreasonable chemical structures.

    Params:
    ----
    `smi`: The SMILES to be check.

    `filter_patterns`: Unreasonable chemical structures.

    `verbose`: If True, matched unreasonable chemical structures will be printed.

    Return:
    ----
    If the SMILES should be filtered.
    """
    return sanitize_checker_mol(Chem.MolFromSmiles(smi), filter_patterns, verbose=verbose)


def sanitize_filter_mols(mols: List[Chem.Mol], filter_patterns: List[Chem.Mol] = FILTER_PATTERNS) -> List[Chem.Mol]:
    return [m for m in mols if sanitize_checker_mol(m, filter_patterns)]


def sanitize_filter(smis: List[str], filter_patterns: List[Chem.Mol] = FILTER_PATTERNS) -> List[str]:
    """
    A filter for SMILES can be sanitized and does not contain unreasonable chemical structures.

    Params:
    ----
    `smis`: The list of SMILES.

    `filter_patterns`: Unreasonable chemical structures.

    Return:
    ----
    The list of SMILES filtered.
    """

    return _mols_to_canonical_smiles(sanitize_filter_mols([Chem.MolFromSmiles(s) for s in smis], filter_patterns))


def make_filter(filter_param: OrderedDict) -> Callable:
    """
    Make a sequential SMILES filter

    Params:
    ----
    `filter_param`: An `collections.OrderedDict` whose keys are single filter functions and the corresponding values are their parameter dictionary.

    Return:
    ----
    The sequential filter function
    """

    def seq_filter(smis):
        for single_filter, param in filter_param.items():
            smis = single_filter(smis, **param)
        return smis

    return seq_filter


_DEFAULT_ENUMERATION_FILTER = make_filter(
    {sanitize_filter: {"filter_patterns": FILTER_PATTERNS}, stereo_filter: {}}
)

_DEFAULT_ENUMERATION_FILTER_MOLS = make_filter(
    {sanitize_filter_mols: {"filter_patterns": FILTER_PATTERNS}, stereo_filter_mols: {}}
)


def prot(mol: Chem.Mol, idx: int, mode: str) -> Chem.Mol:
    """
    Protonate / Deprotonate a molecule at a specified site

    Params:
    ----
    `mol`: Molecule

    `idx`: Index of reaction

    `mode`: `a2b` means deprotonization, with a hydrogen atom or a heavy atom at `idx`; `b2a` means protonization, with a heavy atom at `idx`

    Return:
    ----
    `mol_prot`: (De)protonated molecule
    """
    mw = Chem.RWMol(mol)
    if mode == "a2b":
        atom_H = mw.GetAtomWithIdx(idx)
        if atom_H.GetAtomicNum() == 1:
            atom_A = atom_H.GetNeighbors()[0]
            charge_A = atom_A.GetFormalCharge()
            atom_A.SetFormalCharge(charge_A - 1)
            mw.RemoveAtom(idx)
            mol_prot = mw.GetMol()
        else:
            charge_H = atom_H.GetFormalCharge()
            numH_H = atom_H.GetTotalNumHs()
            if numH_H==0:
                raise ProtonationError(f"No hydrogen at atom index {idx} to remove.")
            atom_H.SetFormalCharge(charge_H - 1)
            new_numH_H = numH_H - 1
            atom_H.SetNumExplicitHs(new_numH_H)
            atom_H.UpdatePropertyCache()
            mol_prot = Chem.AddHs(mw)
    elif mode == "b2a":
        atom_B = mw.GetAtomWithIdx(idx)
        charge_B = atom_B.GetFormalCharge()
        atom_B.SetFormalCharge(charge_B + 1)
        numH_B = atom_B.GetNumExplicitHs()
        atom_B.SetNumExplicitHs(numH_B + 1)
        mol_prot = Chem.AddHs(mw)
    try:
        Chem.SanitizeMol(mol_prot)
    except Exception as e:
        return None
    mol_prot = Chem.MolFromSmiles(Chem.MolToSmiles(mol_prot, canonical=False))
    mol_prot = Chem.AddHs(mol_prot)
    return mol_prot


def read_template(template_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a protonation template.

    Params:
    ----
    `template_file`: path of `.csv`-like template, with columns of substructure names, SMARTS patterns, protonation indices and acid/base flags

    Return:
    ----
    `template_a2b`, `template_b2a`: acid to base and base to acid templates
    """
    template = pd.read_csv(template_file, sep="\t")
    template_a2b = template[template.Acid_or_base == "A"]
    template_b2a = template[template.Acid_or_base == "B"]
    return template_a2b, template_b2a


def _enumerate_template_mols(
    mols: List[Chem.Mol],
    template_a2b: pd.DataFrame,
    template_b2a: pd.DataFrame,
    mode: str,
    maxiter: int,
    verbose: int,
    filter_patterns: List[Chem.Mol],
) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
    """Enumerate microstates while keeping `Chem.Mol` objects (SMILES only at public API boundaries)."""
    if mode == "a2b":
        mols_A_pool, mols_B_pool = list(mols), []
    elif mode == "b2a":
        mols_A_pool, mols_B_pool = [], list(mols)
    else:
        raise ValueError(f"mode must be 'a2b' or 'b2a', got {mode!r}")

    if filter_patterns is FILTER_PATTERNS:
        filters = _DEFAULT_ENUMERATION_FILTER_MOLS
    else:
        filters = make_filter(
            {sanitize_filter_mols: {"filter_patterns": filter_patterns}, stereo_filter_mols: {}}
        )

    pool_length_A = -1
    pool_length_B = -1
    i = 0
    while (len(mols_A_pool) != pool_length_A or len(mols_B_pool) != pool_length_B) and i < maxiter:
        heartbeat(
            {
                "_enumerate_template_mols": "iter",
                "mode": mode,
                "i": i,
                "n_A": len(mols_A_pool),
                "n_B": len(mols_B_pool),
            }
        )
        pool_length_A, pool_length_B = len(mols_A_pool), len(mols_B_pool)
        if verbose > 0:
            print(f"iter {i}: {pool_length_A} acid, {pool_length_B} base")
        if verbose > 1:
            print(
                f"iter {i}, acid: {_mols_to_canonical_smiles(mols_A_pool)}, "
                f"base: {_mols_to_canonical_smiles(mols_B_pool)}"
            )
        if (mode == "a2b" and (i + 1) % 2) or (mode == "b2a" and i % 2):
            mols_A_tmp_pool = []
            for mj, mol in enumerate(mols_A_pool):
                heartbeat(
                    {"_enumerate_template_mols": "expand_a2b", "mode": mode, "outer_i": i, "mj": mj}
                )
                mols_B_pool += filters(prot_template_mol(template_a2b, mol, "a2b")[1])
                mols_A_tmp_pool += filters([Chem.Mol(mol)])
            mols_A_pool += mols_A_tmp_pool
        elif (mode == "b2a" and (i + 1) % 2) or (mode == "a2b" and i % 2):
            mols_B_tmp_pool = []
            for mj, mol in enumerate(mols_B_pool):
                heartbeat(
                    {"_enumerate_template_mols": "expand_b2a", "mode": mode, "outer_i": i, "mj": mj}
                )
                mols_A_pool += filters(prot_template_mol(template_b2a, mol, "b2a")[1])
                mols_B_tmp_pool += filters([Chem.Mol(mol)])
            mols_B_pool += mols_B_tmp_pool
        mols_A_pool = filters(mols_A_pool)
        mols_B_pool = filters(mols_B_pool)
        mols_A_pool = _dedup_mols(mols_A_pool)
        mols_B_pool = _dedup_mols(mols_B_pool)
        i += 1
    if verbose > 0:
        print(f"iter {i}: {pool_length_A} acid, {pool_length_B} base")
    if verbose > 1:
        print(
            f"iter {i}, acid: {_mols_to_canonical_smiles(mols_A_pool)}, "
            f"base: {_mols_to_canonical_smiles(mols_B_pool)}"
        )
    return mols_A_pool, mols_B_pool


def enumerate_template(
    smi: Union[str, List[str]],
    template_a2b: pd.DataFrame,
    template_b2a: pd.DataFrame,
    mode: str = "a2b",
    maxiter: int = 10,
    verbose: int = 0,
    filter_patterns: List[Chem.Mol] = FILTER_PATTERNS,
) -> Tuple[List[str], List[str]]:
    """
    Enumerate all the (de)protonation results of one SMILES.

    Params:
    ----
    `smi`: The smiles to be processed.

    `template_a2b`: `pandas.Dataframe` with columns of substructure names, SMARTS patterns, deprotonation indices and acid flags.

    `template_b2a`: `pandas.Dataframe` with columns of substructure names, SMARTS patterns, protonation indices and base flags.

    `mode`:
        - "a2b": `smi` is an acid to be deprotonated.
        - "b2a": `smi` is a base to be protonated.

    `maxiter`: Max iteration number of template matching and microstate pool growth.

    `verbose`:
        - 0: Silent mode.
        - 1: Print the length of microstate pools in each iteration.
        - 2: Print the content of microstate pools in each iteration.

    `filter_patterns`: Unreasonable chemical structures.

    Return:
    ----
    A microstate pool and B microstate pool after enumeration.
    """
    mols_in = _smis_to_mols(smi)
    mols_a, mols_b = _enumerate_template_mols(
        mols_in, template_a2b, template_b2a, mode, maxiter, verbose, filter_patterns
    )
    return _mols_to_canonical_smiles(mols_a), _mols_to_canonical_smiles(mols_b)


def log_sum_exp(DfGm: List[float]) -> float:
    return math.log10(sum([math.exp(-g) for g in DfGm]))


def _band_from_mols(mols: Sequence[Chem.Mol]) -> Dict[str, Chem.Mol]:
    return {_mol_canonical_key(m): m for m in mols}


def _band_merge(dest: Dict[str, Chem.Mol], mols: Sequence[Chem.Mol]) -> None:
    for m in mols:
        dest.setdefault(_mol_canonical_key(m), m)


def get_ensemble(
    mol: Chem.Mol, template_a2b: pd.DataFrame, template_b2a: pd.DataFrame, maxiter: int = 10
) -> Dict[int, List[Chem.Mol]]:
    """
    Build charge-state microstate pools by walking deprotonation / protonation steps.

    Expansion is queued by formal-charge level; each source charge is expanded at most once
    per direction (visited sets), with at most ``maxiter`` ``enumerate_template`` calls per
    direction so behaviour stays bounded like the previous fixed-length loops.

    Returns one ``Chem.Mol`` per microstate per formal charge (sorted by canonical isomeric SMILES).
    """
    mol0 = Chem.Mol(mol)
    q0 = Chem.GetFormalCharge(mol0)
    ensemble: Dict[int, Dict[str, Chem.Mol]] = {q0: _band_from_mols([mol0])}

    heartbeat({"get_ensemble": "start_a2b_shell", "q0": q0})
    m0_out, m_b1 = _enumerate_template_mols(
        list(ensemble[q0].values()), template_a2b, template_b2a, "a2b", maxiter, 0, FILTER_PATTERNS
    )
    ensemble[q0] = _band_from_mols(m0_out)
    visited_a2b: set[int] = {q0}

    down_queue: deque[int] = deque()
    if m_b1:
        ensemble[q0 - 1] = _band_from_mols(m_b1)
        down_queue.append(q0 - 1)

    n_down = 0
    while down_queue and n_down < maxiter:
        heartbeat({"get_ensemble": "down_queue", "n_down": n_down, "q_src_peek": down_queue[0] if down_queue else None})
        q_src = down_queue.popleft()
        if q_src in visited_a2b:
            continue
        band = ensemble.get(q_src)
        if not band:
            continue
        visited_a2b.add(q_src)
        n_down += 1
        _, m_b = _enumerate_template_mols(
            list(band.values()), template_a2b, template_b2a, "a2b", maxiter, 0, FILTER_PATTERNS
        )
        if not m_b:
            continue
        q_dst = q_src - 1
        _band_merge(ensemble.setdefault(q_dst, {}), m_b)
        down_queue.append(q_dst)

    heartbeat({"get_ensemble": "start_b2a_shell", "q0": q0})
    m_a1, m0_b2a = _enumerate_template_mols(
        list(ensemble[q0].values()), template_a2b, template_b2a, "b2a", maxiter, 0, FILTER_PATTERNS
    )
    ensemble[q0] = _band_from_mols(m0_b2a)
    visited_b2a: set[int] = {q0}

    up_queue: deque[int] = deque()
    if m_a1:
        ensemble[q0 + 1] = _band_from_mols(m_a1)
        up_queue.append(q0 + 1)

    n_up = 0
    while up_queue and n_up < maxiter:
        heartbeat({"get_ensemble": "up_queue", "n_up": n_up, "q_src_peek": up_queue[0] if up_queue else None})
        q_src = up_queue.popleft()
        if q_src in visited_b2a:
            continue
        band = ensemble.get(q_src)
        if not band:
            continue
        visited_b2a.add(q_src)
        n_up += 1
        m_a, _ = _enumerate_template_mols(
            list(band.values()), template_a2b, template_b2a, "b2a", maxiter, 0, FILTER_PATTERNS
        )
        if not m_a:
            continue
        q_dst = q_src + 1
        _band_merge(ensemble.setdefault(q_dst, {}), m_a)
        up_queue.append(q_dst)

    return {q: sorted(band.values(), key=_mol_canonical_key) for q, band in ensemble.items()}
