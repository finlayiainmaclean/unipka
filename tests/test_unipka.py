import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from unipka.unipka import UnipKa, EnumerationError


@pytest.fixture
def unipka_calc():
    """Fixture providing UnipKa calculator instance."""
    return UnipKa(batch_size=16, use_gpu=False)


@pytest.fixture
def sample_molecules():
    """Fixture providing sample molecules for testing."""
    return {
        "piperidine": "C1CCNCC1",
        "acetic_acid": "CC(=O)O",
        "phenol": "c1ccc(cc1)O",
        "aniline": "c1ccc(cc1)N",
        "imidazole": "c1c[nH]cn1"
    }


class TestUnipKaInitialization:
    def test_init_default_params(self):
        calc = UnipKa()
        assert calc.batch_size == 32
        assert calc.device.type in ['cpu', 'cuda']
        assert hasattr(calc, 'model')
        assert hasattr(calc, 'conformer_gen')
        assert hasattr(calc, 'template_a2b')
        assert hasattr(calc, 'template_b2a')

    def test_init_custom_params(self):
        calc = UnipKa(batch_size=16, remove_hs=True, use_gpu=False)
        assert calc.batch_size == 16
        assert calc.params["remove_hs"]
        assert calc.device.type == 'cpu'


class TestUnipKaPublicMethods:
    def test_get_acidic_macro_pka_string_input(self, unipka_calc, sample_molecules):
        pka = unipka_calc.get_acidic_macro_pka(sample_molecules["acetic_acid"])
        assert isinstance(pka, float)
        assert 0 < pka < 14  # Reasonable pKa range

    def test_get_acidic_macro_pka_mol_input(self, unipka_calc, sample_molecules):
        mol = Chem.MolFromSmiles(sample_molecules["acetic_acid"])
        pka = unipka_calc.get_acidic_macro_pka(mol)
        assert isinstance(pka, float)
        assert 0 < pka < 14

    def test_get_basic_macro_pka_string_input(self, unipka_calc, sample_molecules):
        pka = unipka_calc.get_basic_macro_pka(sample_molecules["piperidine"])
        assert isinstance(pka, float)
        assert np.isclose(pka, 11, atol=1)

    def test_get_basic_macro_pka_mol_input(self, unipka_calc, sample_molecules):
        mol = Chem.MolFromSmiles(sample_molecules["piperidine"])
        pka = unipka_calc.get_basic_macro_pka(mol)
        assert isinstance(pka, float)
        assert np.isclose(pka, 11, atol=1)

    def test_get_acidic_micro_pka(self, unipka_calc, sample_molecules):
        pka = unipka_calc.get_acidic_micro_pka(sample_molecules["phenol"], idx=0)
        assert isinstance(pka, float)
        assert np.isclose(pka, 10, atol=1)

    def test_get_distribution_string_input(self, unipka_calc, sample_molecules):
        df = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=7.4)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'population' in df.columns
        assert 'smiles' in df.columns
        assert 'charge' in df.columns
        assert 'mol' in df.columns
        assert np.isclose(df['population'].sum(), 1.0, atol=1e-6)

    def test_get_distribution_mol_input(self, unipka_calc, sample_molecules):
        mol = Chem.MolFromSmiles(sample_molecules["piperidine"])
        df = unipka_calc.get_distribution(mol, pH=7.4)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert np.isclose(df['population'].sum(), 1.0, atol=1e-6)

    def test_get_distribution_different_ph(self, unipka_calc, sample_molecules):
        df1 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=2.0)
        df2 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=12.0)
        
        # Distributions should be different at different pH
        assert not df1['population'].equals(df2['population'])

    def test_get_dominant_microstate(self, unipka_calc, sample_molecules):
        mol = unipka_calc.get_dominant_microstate(sample_molecules["piperidine"], pH=7.4)
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumAtoms() > 0

    def test_get_logd(self, unipka_calc, sample_molecules):
        logd = unipka_calc.get_logd(sample_molecules["piperidine"], pH=7.4)
        assert isinstance(logd, float)
        assert -10 < logd < 10  # Reasonable logD range

    def test_get_state_penalty(self, unipka_calc, sample_molecules):
        sp, reference_df = unipka_calc.get_state_penalty(sample_molecules["piperidine"], pH=7.4)
        assert isinstance(sp, float)
        assert sp >= 0  # State penalty should be non-negative
        assert isinstance(reference_df, pd.DataFrame)
        assert not reference_df.empty

    def test_get_macro_pka_from_macrostates_string_input(self, unipka_calc):
        # Simple test with manually defined macrostates
        macrostate_a = ["CC(=O)O"]  # acetic acid
        macrostate_b = ["CC(=O)[O-]"]  # acetate
        pka = unipka_calc.get_macro_pka_from_macrostates(macrostate_a, macrostate_b)
        assert isinstance(pka, float)
        assert 0 < pka < 14

    def test_get_macro_pka_from_macrostates_mol_input(self, unipka_calc):
        # Test with Mol objects
        mol_a = [Chem.MolFromSmiles("CC(=O)O")]
        mol_b = [Chem.MolFromSmiles("CC(=O)[O-]")]
        pka = unipka_calc.get_macro_pka_from_macrostates(mol_a, mol_b)
        assert isinstance(pka, float)
        assert 0 < pka < 14


class TestUnipKaPrivateMethods:
    def test_get_formal_charge_neutral(self, unipka_calc):
        mol = Chem.MolFromSmiles("CCO")  # ethanol
        abs_formal, abs_atoms = unipka_calc._get_formal_charge(mol)
        assert abs_formal == 0
        assert abs_atoms == 0

    def test_get_formal_charge_charged(self, unipka_calc):
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # acetate
        abs_formal, abs_atoms = unipka_calc._get_formal_charge(mol)
        assert abs_formal == 1
        assert abs_atoms == 1

    def test_get_formal_charge_none(self, unipka_calc):
        result = unipka_calc._get_formal_charge(None)
        assert result == float("inf")

    def test_get_distribution_from_free_energy(self, unipka_calc):
        # Mock ensemble free energy data
        ensemble_free_energy = {
            0: [("CCO", -5.0)],  # neutral
            1: [("CC[OH2+]", -3.0)]  # protonated
        }
        df = unipka_calc._get_distribution_from_free_energy(ensemble_free_energy, pH=7.4)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'population' in df.columns
        assert 'smiles' in df.columns
        assert 'charge' in df.columns
        assert np.isclose(df['population'].sum(), 1.0, atol=1e-6)

    def test_predict_single_molecule(self, unipka_calc):
        result = unipka_calc._predict("CCO")
        assert isinstance(result, dict)
        assert "CCO" in result
        assert isinstance(result["CCO"], float)

    def test_predict_multiple_molecules(self, unipka_calc):
        molecules = ["CCO", "CCC"]
        result = unipka_calc._predict(molecules)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(mol in result for mol in molecules)
        assert all(isinstance(energy, float) for energy in result.values())


class TestUnipKaErrorHandling:
    def test_invalid_smiles(self, unipka_calc):
        # Test with invalid SMILES - should handle gracefully
        with pytest.raises((ValueError, AttributeError, Exception)):
            unipka_calc.get_distribution("invalid_smiles", pH=7.4)

    def test_enumeration_error(self, unipka_calc):
        # Test molecule that might fail enumeration
        simple_molecule = "C"  # methane - no ionizable groups
        with pytest.raises(EnumerationError):
            unipka_calc._predict_ensemble_free_energy(simple_molecule)

    def test_empty_macrostate_lists(self, unipka_calc):
        with pytest.raises((IndexError, ValueError)):
            unipka_calc.get_macro_pka_from_macrostates([], [])

    def test_ph_extreme_values(self, unipka_calc, sample_molecules):
        # Test with extreme pH values
        df1 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=-5)
        df2 = unipka_calc.get_distribution(sample_molecules["piperidine"], pH=20)
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert not df1.empty
        assert not df2.empty


class TestUnipKaConsistency:
    def test_string_mol_consistency(self, unipka_calc, sample_molecules):
        """Test that string and Mol inputs give same results."""
        smi = sample_molecules["piperidine"]
        mol = Chem.MolFromSmiles(smi)
        
        pka_str = unipka_calc.get_basic_macro_pka(smi)
        pka_mol = unipka_calc.get_basic_macro_pka(mol)
        
        assert np.isclose(pka_str, pka_mol, atol=1e-6)

    def test_distribution_sum_to_one(self, unipka_calc, sample_molecules):
        """Test that microstate populations sum to 1."""
        for smi in sample_molecules.values():
            df = unipka_calc.get_distribution(smi, pH=7.4)
            total_pop = df['population'].sum()
            assert np.isclose(total_pop, 1.0, atol=1e-6)

    def test_dominant_microstate_consistency(self, unipka_calc, sample_molecules):
        """Test that dominant microstate matches highest population."""
        smi = sample_molecules["piperidine"]
        df = unipka_calc.get_distribution(smi, pH=7.4)
        dominant = unipka_calc.get_dominant_microstate(smi, pH=7.4)
        
        # Get the highest population microstate from distribution
        top_row = df.iloc[0]  # Already sorted by population descending
        
        # Compare SMILES strings (dominant microstate mol should match top population)
        dominant_smi = Chem.MolToSmiles(dominant)
        top_smi = top_row['smiles']
        
        assert dominant_smi == top_smi or Chem.CanonSmiles(dominant_smi) == Chem.CanonSmiles(top_smi)


class TestUnipKaIntegration:
    def test_workflow_piperidine(self, unipka_calc):
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