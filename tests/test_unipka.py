import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from unipka.unipka import EnumerationError, Microstate, UnipKa


@pytest.fixture
def unipka_calc():
    """Fixture providing UnipKa calculator instance."""
    return UnipKa(batch_size=16)


@pytest.fixture
def sample_molecules():
    """Fixture providing sample molecules for testing."""
    return {
        "piperidine": "C1CCNCC1",
        "acetic_acid": "CC(=O)O",
        "phenol": "c1ccc(cc1)O",
        "aniline": "c1ccc(cc1)N",
        "imidazole": "c1c[nH]cn1",
    }


def test_init_default_params():
    calc = UnipKa()
    assert calc.batch_size == 32
    assert calc.device.type in ["cpu", "cuda"]
    assert hasattr(calc, "model")
    assert hasattr(calc, "conformer_gen")
    assert hasattr(calc, "template_a2b")
    assert hasattr(calc, "template_b2a")


def test_init_custom_params():
    calc = UnipKa(batch_size=16, remove_hs=True, enumerate_tautomers=True)
    assert calc.batch_size == 16
    assert calc.params["remove_hs"]
    assert calc.device.type == "cpu"
    assert calc.enumerate_tautomers is True


def test_get_acidic_macro_pka_string_input(unipka_calc, sample_molecules):
    pka = unipka_calc.get_acidic_macro_pka(sample_molecules["acetic_acid"])
    assert isinstance(pka, float)
    assert 0 < pka < 14  # Reasonable pKa range


def test_get_acidic_macro_pka_mol_input(unipka_calc, sample_molecules):
    mol = Chem.MolFromSmiles(sample_molecules["acetic_acid"])
    pka = unipka_calc.get_acidic_macro_pka(mol)
    assert isinstance(pka, float)
    assert 0 < pka < 14


def test_get_basic_macro_pka_string_input(unipka_calc, sample_molecules):
    pka = unipka_calc.get_basic_macro_pka(sample_molecules["piperidine"])
    assert isinstance(pka, float)
    assert np.isclose(pka, 11, atol=1)


def test_get_basic_macro_pka_mol_input(unipka_calc, sample_molecules):
    mol = Chem.MolFromSmiles(sample_molecules["piperidine"])
    pka = unipka_calc.get_basic_macro_pka(mol)
    assert isinstance(pka, float)
    assert np.isclose(pka, 11, atol=1)


def test_get_acidic_micro_pka(unipka_calc, sample_molecules):
    pka = unipka_calc.get_acidic_micro_pka(sample_molecules["phenol"], idx=0)
    assert isinstance(pka, float)
    assert np.isclose(pka, 10, atol=1)


def test_get_distribution_string_input(unipka_calc, sample_molecules):
    df = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=7.4)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "population" in df.columns
    assert "smiles" in df.columns
    assert "charge" in df.columns
    assert "mol" in df.columns
    assert np.isclose(df["population"].sum(), 1.0, atol=1e-6)


def test_get_distribution_mol_input(unipka_calc, sample_molecules):
    mol = Chem.MolFromSmiles(sample_molecules["piperidine"])
    df = unipka_calc.get_distribution(mol, pH=7.4)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert np.isclose(df["population"].sum(), 1.0, atol=1e-6)


def test_get_distribution_different_ph(unipka_calc, sample_molecules):
    df1 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=2.0)
    df2 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=12.0)

    # Distributions should be different at different pH
    assert not df1["population"].equals(df2["population"])


def test_get_distribution_multiple_ph(unipka_calc, sample_molecules):
    pHs = [2.0, 7.4, 12.0]
    smi = sample_molecules["piperidine"]

    df_single = unipka_calc.get_distribution(smi, pH=7.4)
    df_multi = unipka_calc.get_distribution(smi, pH=pHs)

    n_microstates = len(df_single)
    assert len(df_multi) == n_microstates * len(pHs)
    assert set(df_multi["ph"]) == set(pHs)
    assert "ph" in df_multi.columns

    for ph in pHs:
        block = df_multi[df_multi["ph"] == ph]
        assert len(block) == n_microstates
        assert np.isclose(block["population"].sum(), 1.0, atol=1e-6)

    microstates_per_ph = {
        ph: set(df_multi[df_multi["ph"] == ph]["smiles"]) for ph in pHs
    }
    assert len(microstates_per_ph) == len(pHs)
    assert all(s == microstates_per_ph[pHs[0]] for s in microstates_per_ph.values())


def test_get_dominant_microstate(unipka_calc, sample_molecules):
    mol = unipka_calc.get_dominant_microstate(sample_molecules["piperidine"], pH=7.4)
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumAtoms() > 0


def test_get_logd(unipka_calc, sample_molecules):
    logd = unipka_calc.get_logd(sample_molecules["piperidine"], pH=7.4)
    assert isinstance(logd, float)
    assert -10 < logd < 10  # Reasonable logD range


def test_get_state_penalty(unipka_calc, sample_molecules):
    sp, reference_df = unipka_calc.get_state_penalty(
        sample_molecules["piperidine"], pH=7.4
    )
    assert isinstance(sp, float)
    assert sp >= 0  # State penalty should be non-negative
    assert isinstance(reference_df, pd.DataFrame)
    assert not reference_df.empty


def test_get_macro_pka_from_macrostates_string_input(unipka_calc):
    # Simple test with manually defined macrostates
    macrostate_a = ["CC(=O)O"]  # acetic acid
    macrostate_b = ["CC(=O)[O-]"]  # acetate
    pka = unipka_calc.get_macro_pka_from_macrostates(
        acid_macrostate=macrostate_a, base_macrostate=macrostate_b
    )
    assert isinstance(pka, float)
    assert 0 < pka < 14


def test_get_macro_pka_from_macrostates_mol_input(unipka_calc):
    # Test with Mol objects
    mol_a = [Chem.MolFromSmiles("CC(=O)O")]
    mol_b = [Chem.MolFromSmiles("CC(=O)[O-]")]
    pka = unipka_calc.get_macro_pka_from_macrostates(
        acid_macrostate=mol_a, base_macrostate=mol_b
    )
    assert isinstance(pka, float)
    assert 0 < pka < 14


def test_get_formal_charge_neutral(unipka_calc):
    mol = Chem.MolFromSmiles("CCO")  # ethanol
    abs_formal, abs_atoms = unipka_calc._get_formal_charge(mol)
    assert abs_formal == 0
    assert abs_atoms == 0


def test_get_formal_charge_charged(unipka_calc):
    mol = Chem.MolFromSmiles("CC(=O)[O-]")  # acetate
    abs_formal, abs_atoms = unipka_calc._get_formal_charge(mol)
    assert abs_formal == 1
    assert abs_atoms == 1


def test_get_formal_charge_none(unipka_calc):
    result = unipka_calc._get_formal_charge(None)
    assert result == (float("inf"), float("inf"))


def test_get_distribution_from_free_energy(unipka_calc):
    # Mock ensemble free energy data (canonical SMILES, mol, ΔfG°)
    ensemble_free_energy = {
        0: [("CCO", Chem.MolFromSmiles("CCO"), -5.0)],  # neutral
        1: [("CC[OH2+]", Chem.MolFromSmiles("CC[OH2+]"), -3.0)],  # protonated
    }
    df = unipka_calc._get_distribution_from_free_energy(ensemble_free_energy, pH=7.4)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "population" in df.columns
    assert "smiles" in df.columns
    assert "mol" in df.columns
    assert "charge" in df.columns
    assert all(isinstance(m, Chem.Mol) for m in df["mol"])
    assert np.isclose(df["population"].sum(), 1.0, atol=1e-6)


def test_predict_single_molecule(unipka_calc):
    result = unipka_calc._predict("CCO")
    assert isinstance(result, dict)
    assert "CCO" in result
    assert isinstance(result["CCO"], float)


def test_predict_multiple_molecules(unipka_calc):
    molecules = ["CCO", "CCC"]
    result = unipka_calc._predict(molecules)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert all(mol in result for mol in molecules)
    assert all(isinstance(energy, float) for energy in result.values())


def test_predict_accepts_mol_list(unipka_calc):
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC")]
    result = unipka_calc._predict(mols)
    assert len(result) == 2
    assert all(isinstance(energy, float) for energy in result.values())


def test_invalid_smiles(unipka_calc):
    # Test with invalid SMILES - should handle gracefully
    with pytest.raises((ValueError, AttributeError, Exception)):
        unipka_calc.get_distribution("invalid_smiles", pH=7.4)


def test_enumeration_error(unipka_calc):
    # Test molecule that might fail enumeration
    simple_molecule = Chem.MolFromSmiles("C")  # methane - no ionizable groups
    with pytest.raises(EnumerationError):
        unipka_calc._predict_ensemble_free_energy(simple_molecule)


def test_empty_macrostate_lists(unipka_calc):
    with pytest.raises((IndexError, ValueError)):
        unipka_calc.get_macro_pka_from_macrostates(
            acid_macrostate=[], base_macrostate=[]
        )


def test_ph_extreme_values(unipka_calc, sample_molecules):
    # Test with extreme pH values
    df1 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=-5)
    df2 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=20)
    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert not df1.empty
    assert not df2.empty


def test_string_mol_consistency(unipka_calc, sample_molecules):
    """Test that string and Mol inputs give same results."""
    smi = sample_molecules["piperidine"]
    mol = Chem.MolFromSmiles(smi)

    pka_str = unipka_calc.get_basic_macro_pka(smi)
    pka_mol = unipka_calc.get_basic_macro_pka(mol)

    assert np.isclose(pka_str, pka_mol, atol=1e-6)


def test_distribution_sum_to_one(unipka_calc, sample_molecules):
    """Test that microstate populations sum to 1."""
    for smi in sample_molecules.values():
        df = unipka_calc.get_distribution(smi, pH=7.4)
        total_pop = df["population"].sum()
        assert np.isclose(total_pop, 1.0, atol=1e-6)


def test_dominant_microstate_consistency(unipka_calc, sample_molecules):
    """Test that dominant microstate matches highest population."""
    smi = sample_molecules["piperidine"]
    df = unipka_calc.get_distribution(smi, pH=7.4)
    dominant = unipka_calc.get_dominant_microstate(smi, pH=7.4)

    # Get the highest population microstate from distribution
    top_row = df.iloc[0]  # Already sorted by population descending

    # Compare SMILES strings (dominant microstate mol should match top population)
    dominant_smi = Chem.MolToSmiles(dominant)
    top_smi = top_row["smiles"]

    assert dominant_smi == top_smi or Chem.CanonSmiles(
        dominant_smi
    ) == Chem.CanonSmiles(top_smi)


def test_get_microstates_returns_acidic_site_in_range(unipka_calc, sample_molecules):
    """Phenol has a single acidic site with pKa ~10."""
    results = unipka_calc.get_microstates(
        sample_molecules["phenol"], min_pka=0, max_pka=14
    )
    assert len(results) == 1
    ms = results[0]
    assert isinstance(ms, Microstate)
    assert isinstance(ms.acid, Chem.Mol)
    assert isinstance(ms.base, Chem.Mol)
    assert isinstance(ms.pka, float)
    assert isinstance(ms.idx, int)
    assert np.isclose(ms.pka, 10, atol=1)

    # Acid is the (neutral) phenol, base is the deprotonated phenoxide.
    assert Chem.MolToSmiles(ms.acid) == Chem.CanonSmiles("Oc1ccccc1")
    assert Chem.MolToSmiles(ms.base) == Chem.CanonSmiles("[O-]c1ccccc1")
    assert Chem.GetFormalCharge(ms.base) == Chem.GetFormalCharge(ms.acid) - 1


def test_get_microstates_returns_basic_site_in_range(unipka_calc, sample_molecules):
    """Piperidine has a single basic site with pKa ~11."""
    results = unipka_calc.get_microstates(
        sample_molecules["piperidine"], min_pka=0, max_pka=14
    )
    assert len(results) == 1
    ms = results[0]
    assert np.isclose(ms.pka, 11, atol=1)

    # The input is the base; protonating gives the conjugate acid.
    assert Chem.MolToSmiles(ms.base) == Chem.CanonSmiles("C1CCNCC1")
    assert Chem.MolToSmiles(ms.acid) == Chem.CanonSmiles("C1CC[NH2+]CC1")
    assert Chem.GetFormalCharge(ms.acid) == Chem.GetFormalCharge(ms.base) + 1


def test_get_microstates_pka_range_filters_sites(unipka_calc, sample_molecules):
    """A range excluding the site's pKa returns no microstates."""
    # Piperidine's basic pKa (~11) falls outside 0-5.
    results = unipka_calc.get_microstates(
        sample_molecules["piperidine"], min_pka=0, max_pka=5
    )
    assert results == []

    # ...but is included once the range covers it.
    results_wide = unipka_calc.get_microstates(
        sample_molecules["piperidine"], min_pka=0, max_pka=14
    )
    assert len(results_wide) == 1
    assert 5 < results_wide[0].pka <= 14


def test_get_microstates_string_and_mol_input_consistent(unipka_calc, sample_molecules):
    smi = sample_molecules["acetic_acid"]
    mol = Chem.MolFromSmiles(smi)

    res_str = unipka_calc.get_microstates(smi, min_pka=0, max_pka=14)
    res_mol = unipka_calc.get_microstates(mol, min_pka=0, max_pka=14)

    assert len(res_str) == len(res_mol) == 1
    assert np.isclose(res_str[0].pka, res_mol[0].pka, atol=1e-6)
    assert Chem.MolToSmiles(res_str[0].acid) == Chem.MolToSmiles(res_mol[0].acid)
    assert Chem.MolToSmiles(res_str[0].base) == Chem.MolToSmiles(res_mol[0].base)
    assert res_str[0].pka < 7  # acetic acid is acidic (~4.6)


def test_get_microstates_no_ionizable_sites_returns_empty(unipka_calc):
    """A molecule with no ionizable groups yields no microstates."""
    results = unipka_calc.get_microstates("CCCC", min_pka=0, max_pka=14)
    assert results == []


def test_workflow_piperidine(unipka_calc):
    """Test complete workflow for piperidine."""
    smi = "C1CCNCC1"

    # Get basic pKa
    pka = unipka_calc.get_basic_macro_pka(smi)
    assert isinstance(pka, float)

    # Get distribution at physiological pH
    df = unipka_calc.get_distribution(smi, pH=7.4)
    assert not df.empty

    # Get dominant microstate
    dominant = unipka_calc.get_dominant_microstate(smi, pH=7.4)
    assert isinstance(dominant, Chem.Mol)

    # Get logD
    logd = unipka_calc.get_logd(smi, pH=7.4)
    assert isinstance(logd, float)

    # Get state penalty
    sp, ref_df = unipka_calc.get_state_penalty(smi, pH=7.4)
    assert isinstance(sp, float)
    assert isinstance(ref_df, pd.DataFrame)


def test_enumerate_tautomers_workflow():
    calc = UnipKa(enumerate_tautomers=True, batch_size=16)
    smi = "c1c[nH]cn1"  # Imidazole
    df = calc.get_distribution(smi, pH=7.4)
    assert not df.empty
    assert "population" in df.columns
    print(df)


def test_2_hydroxypyridine_tautomer_enumeration():
    """Test that 2-hydroxypyridine tautomers (lactam/lactim forms) are enumerated when enabled."""
    smi = "Oc1ccccn1"

    # 1. Without tautomer enumeration
    calc_no = UnipKa(enumerate_tautomers=False, batch_size=16)
    df_no = calc_no.get_distribution(smi)
    smiles_no = df_no["smiles"].tolist()

    # Should contain the lactim form but NOT the lactam (2-pyridone) form
    assert "Oc1ccccn1" in smiles_no
    assert "O=c1cccc[nH]1" not in smiles_no

    # 2. With tautomer enumeration
    calc_yes = UnipKa(enumerate_tautomers=True, batch_size=16)
    df_yes = calc_yes.get_distribution(smi)
    smiles_yes = df_yes["smiles"].tolist()

    # Should now contain both lactim and lactam forms
    assert "Oc1ccccn1" in smiles_yes
    assert "O=c1cccc[nH]1" in smiles_yes
