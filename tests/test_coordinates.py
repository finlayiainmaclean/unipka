from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist

from unipka import UnipKa
from unipka._internal.conformer import ConformerGen
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


def test_mmff_optimise_unsupported_metals():
    from unipka._internal.coordinates import mmff_optimise

    mol = Chem.MolFromSmiles("[Fe]")
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    mol.AddConformer(conf)

    res_mol, energy = mmff_optimise(mol)
    assert res_mol is mol
    assert energy is None


def test_transplant_coordinates_unsupported_metals():
    ref = Chem.MolFromSmiles("[Fe]")
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    ref.AddConformer(conf)

    query = Chem.MolFromSmiles("[Fe]")

    out = transplant_coordinates(ref, query)
    assert out.GetNumConformers() == 1
    assert out.GetNumAtoms() == 1


def test_posed_microstates_share_the_pose():
    """Every microstate of a posed input is featurised in that pose, with placed hydrogens."""
    pose = _embed(Chem.MolFromSmiles("NC(=[NH2+])c1ccccc1"))
    heavy_pose = Chem.RemoveHs(pose)
    gen = ConformerGen()
    gen.pose = heavy_pose
    # the input itself (has a conformer, no H) and a microstate enumerated from it (no conformer)
    for mol in (heavy_pose, Chem.MolFromSmiles("N=C(N)c1ccccc1")):
        feats = gen.single_process(mol)
        dist = feats["src_distance"][1:-1, 1:-1]  # drop BOS/EOS
        assert (dist + 99 * np.eye(len(dist))).min() > 0.5  # no H stacked at the origin
        n = heavy_pose.GetNumAtoms()
        np.testing.assert_allclose(
            np.sort(pdist(feats["src_coord"][1 : n + 1])),
            np.sort(pdist(get_coordinates(heavy_pose))),
            atol=1e-3,
        )
