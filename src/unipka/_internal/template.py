import math
from typing import Callable, Dict, List, OrderedDict, Tuple, Union

import pandas as pd
from rdkit import Chem

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
        pattern = Chem.MolFromSmarts(smarts)
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
    sites = match_template(template, mol)
    smis = []
    for site in sites:
        smis.append(Chem.CanonSmiles(Chem.MolToSmiles(Chem.RemoveHs(prot(mol, site, mode)))))
    return sites, list(set(smis))


def cnt_stereo_atom(smi: str) -> int:
    """
    Count the stereo atoms in a SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    return sum([str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()])


def stereo_filter(smis: List[str]) -> List[str]:
    """
    A filter against SMILES losing stereochemical information in structure processing.
    """
    filtered_smi_dict = dict()
    for smi in smis:
        nonstereo_smi = Chem.CanonSmiles(smi, useChiral=0)
        stereo_cnt = cnt_stereo_atom(smi)
        if nonstereo_smi not in filtered_smi_dict:
            filtered_smi_dict[nonstereo_smi] = (smi, stereo_cnt)
        else:
            if stereo_cnt > filtered_smi_dict[nonstereo_smi][1]:
                filtered_smi_dict[nonstereo_smi] = (smi, stereo_cnt)
    return [value[0] for value in filtered_smi_dict.values()]


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
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    for pattern in filter_patterns:
        match = mol.GetSubstructMatches(pattern)
        if match:
            if verbose:
                print(f"pattern {pattern}")
            return False
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print("cannot sanitize", repr(e))
        return False
    return True


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

    def _checker(smi):
        return sanitize_checker(smi, filter_patterns)

    return list(filter(_checker, smis))


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
            atom_H.SetFormalCharge(charge_H - 1)
            atom_H.SetNumExplicitHs(numH_H - 1)
            atom_H.UpdatePropertyCache()
            mol_prot = Chem.AddHs(mw)
    elif mode == "b2a":
        atom_B = mw.GetAtomWithIdx(idx)
        charge_B = atom_B.GetFormalCharge()
        atom_B.SetFormalCharge(charge_B + 1)
        numH_B = atom_B.GetNumExplicitHs()
        atom_B.SetNumExplicitHs(numH_B + 1)
        mol_prot = Chem.AddHs(mw)
    Chem.SanitizeMol(mol_prot)
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


def enumerate_template(
    smi: Union[str, List[str]],
    template_a2b: pd.DataFrame,
    template_b2a: pd.DataFrame,
    mode: str = "A",
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
    if isinstance(smi, str):
        smis = [smi]
    else:
        smis = list(smi)

    def enum_func(x):
        return [x]  # TODO: Tautomerism enumeration

    if mode == "a2b":
        smis_A_pool, smis_B_pool = smis, []
    elif mode == "b2a":
        smis_A_pool, smis_B_pool = [], smis
    filters = make_filter({sanitize_filter: {"filter_patterns": filter_patterns}, stereo_filter: {}})
    pool_length_A = -1
    pool_length_B = -1
    i = 0
    while (len(smis_A_pool) != pool_length_A or len(smis_B_pool) != pool_length_B) and i < maxiter:
        pool_length_A, pool_length_B = len(smis_A_pool), len(smis_B_pool)
        if verbose > 0:
            print(f"iter {i}: {pool_length_A} acid, {pool_length_B} base")
        if verbose > 1:
            print(f"iter {i}, acid: {smis_A_pool}, base: {smis_B_pool}")
        if (mode == "a2b" and (i + 1) % 2) or (mode == "b2a" and i % 2):
            smis_A_tmp_pool = []
            for smi in smis_A_pool:
                smis_B_pool += filters(prot_template(template_a2b, smi, "a2b")[1])
                smis_A_tmp_pool += filters(
                    [Chem.CanonSmiles(Chem.MolToSmiles(mol)) for mol in enum_func(Chem.MolFromSmiles(smi))]
                )
            smis_A_pool += smis_A_tmp_pool
        elif (mode == "b2a" and (i + 1) % 2) or (mode == "a2b" and i % 2):
            smis_B_tmp_pool = []
            for smi in smis_B_pool:
                smis_A_pool += filters(prot_template(template_b2a, smi, "b2a")[1])
                smis_B_tmp_pool += filters(
                    [Chem.CanonSmiles(Chem.MolToSmiles(mol)) for mol in enum_func(Chem.MolFromSmiles(smi))]
                )
            smis_B_pool += smis_B_tmp_pool
        smis_A_pool = filters(smis_A_pool)
        smis_B_pool = filters(smis_B_pool)
        smis_A_pool = list(set(smis_A_pool))
        smis_B_pool = list(set(smis_B_pool))
        i += 1
    if verbose > 0:
        print(f"iter {i}: {pool_length_A} acid, {pool_length_B} base")
    if verbose > 1:
        print(f"iter {i}, acid: {smis_A_pool}, base: {smis_B_pool}")
    smis_A_pool = list(map(Chem.CanonSmiles, smis_A_pool))
    smis_B_pool = list(map(Chem.CanonSmiles, smis_B_pool))
    return smis_A_pool, smis_B_pool


def log_sum_exp(DfGm: List[float]) -> float:
    return math.log10(sum([math.exp(-g) for g in DfGm]))


def get_ensemble(
    smi: str, template_a2b: pd.DataFrame, template_b2a: pd.DataFrame, maxiter: int = 10
) -> Dict[int, List[str]]:
    ensemble = dict()
    q0 = Chem.GetFormalCharge(Chem.MolFromSmiles(smi))
    ensemble[q0] = [smi]

    smis_0 = [smi]

    smis_0, smis_b1 = enumerate_template(smis_0, template_a2b, template_b2a, maxiter=maxiter, mode="a2b")
    if smis_b1:
        ensemble[q0 - 1] = smis_b1
    q = q0 - 2
    for _ in range(maxiter):
        if q + 1 in ensemble:
            _, smis_b = enumerate_template(ensemble[q + 1], template_a2b, template_b2a, maxiter=maxiter, mode="a2b")
            if smis_b:
                ensemble[q] = smis_b
            else:
                break
        q -= 1

    smis_a1, smis_0 = enumerate_template(smis_0, template_a2b, template_b2a, maxiter=maxiter, mode="b2a")
    if smis_a1:
        ensemble[q0 + 1] = smis_a1
    q = q0 + 2
    for _ in range(maxiter):
        if q - 1 in ensemble:
            smis_a, _ = enumerate_template(ensemble[q - 1], template_a2b, template_b2a, maxiter=maxiter, mode="b2a")
            if smis_a:
                ensemble[q] = smis_a
            else:
                break
        q += 1

    ensemble[q0] = smis_0
    return ensemble
