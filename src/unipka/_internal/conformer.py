import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Final, Sequence, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix
from .dict import DICT, DICT_CHARGE, Dictionary

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _detect_cpus() -> int:
    env = os.environ.get("TORCH_NUM_THREADS")
    if env:
        return max(1, int(env))
    try:
        return max(1, len(os.sched_getaffinity(0)))  # Linux: respects cpusets/cgroups
    except AttributeError:
        return max(1, multiprocessing.cpu_count())


NUM_CPUS: Final[int] = _detect_cpus()


class ConformerGen(object):
    """
    Generate conformers for molecules given as SMILES strings or RDKit ``Chem.Mol`` instances.
    """

    def __init__(self, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        self._init_features(**params)
        # Cache by canonical SMILES so repeated inputs skip embedding + featurization.
        # Bound to the instance so config changes via a new ConformerGen get a fresh cache.
        cache_size = params.get("cache_size", 4096)
        self._cached_process_smi = lru_cache(maxsize=cache_size)(self._process_smi)

    def _init_features(self, **params):
        """
        Initializes the features of the ConformerGen object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                       These can include the random seed, maximum number of atoms, data type,
                       generation method, generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get("seed", 42)
        self.max_atoms = params.get("max_atoms", 256)
        self.data_type = params.get("data_type", "molecule")
        self.method = params.get("method", "rdkit_random")
        self.mode = params.get("mode", "fast")
        self.remove_hs = params.get("remove_hs", False)
        self.dict_dir = params.get("dict_dir", "dict")
        self.dictionary = Dictionary.load_from_str(DICT)
        self.dictionary.add_symbol("[MASK]", is_special=True)
        self.charge_dictionary = Dictionary.load_from_str(DICT_CHARGE)
        self.charge_dictionary.add_symbol("[MASK]", is_special=True)

    def single_process(self, mol_or_smi: Union[str, Chem.Mol]):
        """
        Processes a single molecule (SMILES string or RDKit Mol) to generate conformers.

        :param mol_or_smi: SMILES string or `Chem.Mol`.
        :return: A unimolecular data representation (dictionary) of the molecule.
        :raises ValueError: If the conformer generation method is unrecognized.
        """
        if self.method != "rdkit_random":
            raise ValueError("Unknown conformer generation method: {}".format(self.method))
        if isinstance(mol_or_smi, str):
            return self._cached_process_smi(mol_or_smi)
        # Mols carrying a pre-computed conformer would lose their coords on a SMILES
        # round-trip, so bypass the cache and process them directly.
        if mol_or_smi.GetNumConformers() > 0:
            return self._process_mol(mol_or_smi)
        smi = Chem.MolToSmiles(mol_or_smi, canonical=True, isomericSmiles=True)
        return self._cached_process_smi(smi)

    def _process_smi(self, smi: str):
        mol = Chem.MolFromSmiles(smi)
        return self._process_mol(mol)

    def _process_mol(self, mol: Chem.Mol):
        atoms, coordinates, charges = inner_mol2coords(
            mol, seed=self.seed, mode=self.mode, remove_hs=self.remove_hs
        )
        return coords2unimol(
            atoms,
            coordinates,
            charges,
            self.dictionary,
            self.charge_dictionary,
            self.max_atoms,
            remove_hs=self.remove_hs,
        )

    def transform_raw(self, atoms_list, coordinates_list, charges_list):
        inputs = []
        for atoms, coordinates, charges in zip(atoms_list, coordinates_list, charges_list):
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    charges,
                    self.dictionary,
                    self.charge_dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def transform(self, mols_or_smis: Sequence[Union[str, Chem.Mol]]):

        n = len(mols_or_smis)
        logger.info(f"Start generating conformers for {n} molecules")
        # RDKit's EmbedMolecule releases the GIL, so threads give real parallelism here.
        # For tiny batches the pool setup overhead dominates; fall back to sequential.
        workers = min(NUM_CPUS, n)
        if workers <= 1:
            inputs = [self.single_process(item) for item in mols_or_smis]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                inputs = list(ex.map(self.single_process, mols_or_smis))
        failed_cnt = np.mean([(item["src_coord"] == 0.0).all() for item in inputs])
        logger.info("Succeed to generate conformers for {:.2f}% of molecules.".format((1 - failed_cnt) * 100))
        failed_3d_cnt = np.mean([(item["src_coord"][:, 2] == 0.0).all() for item in inputs])
        logger.info("Succeed to generate 3d conformers for {:.2f}% of molecules.".format((1 - failed_3d_cnt) * 100))
        return inputs


def _etkdg_params(seed: int) -> AllChem.ETKDGv3:
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    # Random starting coords are more robust on macrocycles / awkward rings and
    # usually converge in fewer attempts than the default 2D-init pathway.
    p.useRandomCoords = True
    p.numThreads = 1
    # NB: numThreads on EmbedParameters is only consulted by EmbedMultipleConfs,
    # so we don't set it here. Cross-molecule parallelism lives in transform().
    return p


def inner_mol2coords(mol: Chem.Mol, seed=42, mode="fast", remove_hs=True):
    """
    Convert an RDKit molecule (with implicit Hs) into 3D coordinates per atom.

    If 3D embedding fails, may fall back to 2D coordinates. Optionally strips hydrogens.

    :param mol: RDKit molecule (typically without explicit Hs; Hs are added internally).
    :param seed: Random seed for conformer embedding.
    :param mode: ``'fast'`` skips MMFF relaxation (ETKDG geometry is used as-is);
                 ``'heavy'`` runs a capped MMFF relaxation and retries embedding on failure.
    :param remove_hs: If True, drop hydrogen atoms from returned atom/coordinate lists.

    :return: ``(atoms, coordinates, charges)``.
    """
    mol = AllChem.AddHs(Chem.Mol(mol))
    label = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    atoms, charges = [], []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
        charges.append(atom.GetFormalCharge())
    assert len(atoms) > 0, f"No atoms in molecule: {label}"
    if mol.GetNumConformers() > 0:
        coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    else:
        try:
            res = AllChem.EmbedMolecule(mol, _etkdg_params(seed))
            if res == 0:
                if mode == "heavy":
                    try:
                        print("Running MMFF optimization")
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
                    except Exception:
                        pass
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            elif res == -1 and mode == "heavy":
                AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
                try:
                    print("Running MMFF optimization")
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
                    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
                except Exception:
                    AllChem.Compute2DCoords(mol)
                    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            else:
                AllChem.Compute2DCoords(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        except Exception:
            print("Failed to generate conformer, replace with zeros.")
            coordinates = np.zeros((len(atoms), 3))
    assert len(atoms) == len(coordinates), f"coordinates shape is not align with {label}"
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != "H"]
        atoms_no_h = [atom for atom in atoms if atom != "H"]
        coordinates_no_h = coordinates[idx]
        charges_no_h = [charges[i] for i in idx]
        assert len(atoms_no_h) == len(coordinates_no_h), f"coordinates shape is not align with {label}"
        return atoms_no_h, coordinates_no_h, charges_no_h
    else:
        return atoms, coordinates, charges


def inner_smi2coords(smi, seed=42, mode="heavy", remove_hs=True):
    """
    Same as :func:`inner_mol2coords` but takes a SMILES string.
    """
    mol = Chem.MolFromSmiles(smi)
    return inner_mol2coords(mol, seed=seed, mode=mode, remove_hs=remove_hs)


def inner_coords(atoms, coordinates, charges, remove_hs=True):
    """
    Processes a list of atoms and their corresponding coordinates to remove hydrogen atoms if specified.
    This function takes a list of atom symbols and their corresponding coordinates and optionally removes hydrogen atoms from the output. It includes assertions to ensure the integrity of the data and uses numpy for efficient processing of the coordinates.

    :param atoms: (list) A list of atom symbols (e.g., ['C', 'H', 'O']).
    :param coordinates: (list of tuples or list of lists) Coordinates corresponding to each atom in the `atoms` list.
    :param remove_hs: (bool, optional) A flag to indicate whether hydrogen atoms should be removed from the output.
                      Defaults to True.

    :return: A tuple containing two elements; the filtered list of atom symbols and their corresponding coordinates.
             If `remove_hs` is False, the original lists are returned.

    :raises AssertionError: If the length of `atoms` list does not match the length of `coordinates` list.
    """
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != "H"]
        atoms_no_h = [atom for atom in atoms if atom != "H"]
        coordinates_no_h = coordinates[idx]
        charges_no_h = [charges[i] for i in idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h, charges_no_h
    else:
        return atoms, coordinates, charges


def coords2unimol(atoms, coordinates, charges, dictionary, charge_dictionary, max_atoms=256, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param atoms: (list) List of atom symbols.
    :param coordinates: (ndarray) Array of atomic coordinates.
    :param dictionary: (Dictionary) An object that maps atom symbols to unique integers.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A dictionary containing the molecular representation with tokens, distances, coordinates, and edge types.
    """
    atoms, coordinates, charges = inner_coords(atoms, coordinates, charges, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    charges = np.array(charges).astype(str)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
        charges = charges[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    src_charges = np.array(
        [charge_dictionary.bos()] + [charge_dictionary.index(charge) for charge in charges] + [charge_dictionary.eos()]
    )
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1, 3)), src_coord, np.zeros((1, 3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
        "src_tokens": src_tokens.astype(int),
        "src_charges": src_charges.astype(int),
        "src_distance": src_distance.astype(np.float32),
        "src_coord": src_coord.astype(np.float32),
        "src_edge_type": src_edge_type.astype(int),
    }
