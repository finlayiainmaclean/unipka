from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from unipka import UnipKa
from unipka._internal.coordinates import get_coordinates, transplant_coordinates


def _embed(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    return mol


def test_transplant_coordinates_substructure_match():
    ref = _embed(Chem.MolFromSmiles("Oc1ccccn1"))
    query = Chem.MolFromSmiles("Oc1ccccn1")

    out = transplant_coordinates(ref, query)
    assert out.GetNumConformers() == 1
    assert out.GetNumAtoms() == ref.GetNumAtoms()


def test_transplant_coordinates_mcs_fallback_for_tautomers():
    ref = _embed(Chem.MolFromSmiles("Oc1ccccn1"))
    query = Chem.MolFromSmiles("O=c1cccc[nH]1")

    out = transplant_coordinates(ref, query)
    ref_coords = get_coordinates(ref)
    out_coords = get_coordinates(out)

    ref_heavy = Chem.RemoveHs(ref)
    out_heavy = Chem.RemoveHs(out)
    ref_heavy_coords = get_coordinates(ref_heavy)
    out_heavy_coords = get_coordinates(out_heavy)

    assert out.GetNumConformers() == 1
    assert ref_heavy.GetNumAtoms() == out_heavy.GetNumAtoms()
    assert ref_heavy_coords.shape == out_heavy_coords.shape
    assert ref_heavy_coords.shape[0] > 0


def test_get_distribution_with_tautomers_and_3d_coords():
    ref_smi = "CCC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1"
    ref = _embed(Chem.MolFromSmiles(ref_smi))

    calc = UnipKa(enumerate_tautomers=True, batch_size=16)
    df = calc.get_distribution(ref)

    assert not df.empty
    assert df["mol"].apply(lambda m: m.GetNumConformers() > 0).all()
