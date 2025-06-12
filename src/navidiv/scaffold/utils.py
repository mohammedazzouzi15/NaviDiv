"""Set of functions to generate a scaffold using the GNN embeddings
and calculate the score based on the charge distribution of the atoms in the scaffold.

For this we use the FORMED_PROP package which is a GNN model trained on molecular properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from FORMED_PROP.data.Dataloader3D import Dataloader3D
from FORMED_PROP.trainers.Trainer3d import Trainer3d
from hydra import (
    compose,
    initialize_config_dir,
)
from hydra.core.global_hydra import GlobalHydra
from rdkit import Chem


class ModelLoader:
    def __init__(self):
        self.data_loader = None
        self.model = None
        self.trainer = None

    def load_model(self, checkpoint_path):
        # Load the graph neural network model from the specified path
        # This is a placeholder for the actual model loading logic
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base=None,
            config_dir="/media/mohammed/Work/FORMED_PROP/configuration_files/conf",
        ):
            cfg = compose(
                config_name="config_pain_embed",
            )
        self.model = f"Loaded model from {checkpoint_path}"
        trainer = Trainer3d(cfg)
        trainer.load_model()
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)
        self.data_loader = Dataloader3D(cfg)
        self.trainer = trainer

    def get_embeddings(self, smiles):
        # Generate embeddings for the given molecular structures
        # This is a placeholder for the actual embedding generation logic
        if isinstance(smiles, str):
            smiles = [smiles]
        if not isinstance(smiles, list):
            raise ValueError("smiles should be a list of SMILES strings")
        mol_loader = self.data_loader.get_data_loader_from_smiles(smiles)
        per_atom_energy = None
        target = None
        per_atom_energy_list = []
        for batch in mol_loader:
            for i in range(batch.natoms.shape[0]):
                # batch to batch id with size 1
                batch[i].batch = [0]
                per_atom_energy = self.trainer.model.get_node_embedding(
                    batch[i]
                )
                per_atom_energy = np.abs(
                    per_atom_energy.cpu().detach().numpy()
                )
                per_atom_energy_list.append(per_atom_energy)

            target = self.trainer.model(batch)["energy"].cpu().detach().numpy()
        return per_atom_energy_list, target


def find_min_threshold(lst, target_sum):
    """Find the minimum threshold value such that the sum of elements in the list
    that are greater than this threshold is above the target_sum.

    Args:
        lst (list of float): List of numbers.
        target_sum (float): Target sum that the elements should exceed.

    Returns:
        float: Minimum threshold value.
    """

    def is_valid_threshold(threshold):
        return sum(x for x in lst if x > threshold) > target_sum

    low, high = min(lst), max(lst)
    while high - low > 1e-6:  # Precision of the threshold
        mid = (low + high) / 2
        if is_valid_threshold(mid):
            low = mid
        else:
            high = mid

    return low


def change_all_non_ring_atoms_to_non_aromatic(mol):
    """Change all non-ring atoms to non-aromatic"""
    for atom in mol.GetAtoms():
        if not atom.IsInRing():
            atom.SetIsAromatic(False)
    return mol


def annotate_molecule(mol, charges):
    for atom_number, atom in enumerate(mol.GetAtoms()):
        charge_sum = charges[atom_number]
        atom.SetProp("atomNote", f"{charge_sum:.2f}")
    mol = Chem.RemoveAllHs(mol)
    return mol


def get_scaffold(mol, threshold=0.9):
    """Get the scaffold of the molecules with only highly contributing atoms"""
    charges = [float(atom.GetProp("atomNote")) for atom in mol.GetAtoms()]
    charge_limit = find_min_threshold(charges, sum(charges) * threshold)
    atoms_to_keep = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if charges[atom.GetIdx()] > charge_limit
    }

    # Include atoms connected to two highlighted atoms
    atoms_connected = set()
    for atom in mol.GetAtoms():
        connected_atoms = [
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
        ]
        if len(set(connected_atoms).intersection(atoms_to_keep)) == 2:
            atoms_connected.add(atom.GetIdx())
    atoms_to_keep.update(atoms_connected)
    new_mol = Chem.RWMol(mol)
    new_mol.BeginBatchEdit()
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atoms_to_keep:
            new_mol.RemoveAtom(atom.GetIdx())
    new_mol.CommitBatchEdit()
    new_mol = change_all_non_ring_atoms_to_non_aromatic(new_mol)
    #print(f"Number of atoms in the scaffold: {len(new_mol.GetAtoms())}")
    # new_mol = annotate_molecule(new_mol, charges)
    return new_mol


def _get_score(mol, threshold=0.9):
    """Generate a molecule by deleting atoms with charge less than the threshold"""
    charges = [float(atom.GetProp("atomNote")) for atom in mol.GetAtoms()]
    charge_limit = find_min_threshold(charges, sum(charges) * threshold)
    atoms_to_keep = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if charges[atom.GetIdx()] > charge_limit
    }

    # Include atoms connected to two highlighted atoms
    atoms_connected = set()
    for atom in mol.GetAtoms():
        connected_atoms = [
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
        ]
        if len(set(connected_atoms).intersection(atoms_to_keep)) == 2:
            atoms_connected.add(atom.GetIdx())
    atoms_to_keep.update(atoms_connected)
    score = len(atoms_to_keep) / len(mol.GetAtoms())
    return score


def add_tag(label: str, text: str = "True"):
    """A simple decorator to tag a class"""

    def wrapper(cls):
        setattr(cls, label, text)
        return cls

    return wrapper


def get_score(smiles, model_loader):
    embeddings, target = model_loader.get_embeddings(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    # Normalize embeddings for color mapping
    if embeddings is None:
        return 0.0
    molecule = annotate_molecule(molecule, embeddings)
    score = _get_score(molecule)
    return score


class Params:
    def __init__(self, checkpoints, threshold) -> None:
        self.checkpoints = checkpoints
        self.threshold = threshold


class scaffoldscore:
    def __init__(self, params: Params):
        # mp.set_start_method("spawn", force=True)

        self.number_of_endpoints = 1
        self.smiles_type = "rdkit_smiles"
        self.checkpoints = params.checkpoint
        """[
            "/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/S1_exc/PainnEmbed/epoch=20-val_mae=0.15-val_mse=0.00.ckpt",
            "/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/T1_exc/PainnEmbed/epoch=90-val_mae=0.15-val_mse=0.00.ckpt",
            "/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/T2_exc/PainnEmbed/epoch=74-val_mae=0.21-val_mse=0.00.ckpt",
        ]"
        """
        self.model_loader = ModelLoader()
        self.model_loader.load_model(self.checkpoints)
        self.threshold = params.threshold

    def get_scaffold(self, smiles):
        """Get the scaffold for a given SMILES string"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = annotate_molecule(mol, scores[0])
        scaffold = get_scaffold(mol, self.threshold)
        return Chem.MolToSmiles(scaffold, isomericSmiles=True)


if __name__ == "__main__":
    params = Params(
        checkpoint="/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/S1_exc/PainnEmbed/epoch=20-val_mae=0.15-val_mse=0.00.ckpt",
        threshold=0.9,
    )
    csv_paths = [
        "/media/mohammed/Work/SF_generative/reinvent_runs/280425_DAP2_Substructure_high/results_all/tsne_results280425.csv",
        "/media/mohammed/Work/SF_generative/reinvent_runs/250425_DAP2_NOSubstructure_high/results_all/tsne_results250425.csv",
        "/media/mohammed/Work/SF_generative/reinvent_runs/210425_DAP_Substructure_high/results_all/tsne_results210425.csv",
    ]
    csv_path = "/media/mohammed/Work/SF_generative/reinvent_runs/280425_DAP2_Substructure_high/results_all/tsne_results280425.csv"
    scorer = scaffoldscore(params)

    for csv_path in csv_paths:
        df_smiles = pd.read_csv(csv_path)
        smiles = df_smiles["smiles"].values
        scores = scorer.get_score(smiles)
        df_smiles["scores_chargeS1"] = [x[0] for x in scores]
        df_smiles["scores_chargeT1"] = [x[1] for x in scores]
        df_smiles["scores_chargeT2"] = [x[2] for x in scores]
        df_smiles.to_csv(
            csv_path.replace(".csv", "_charge.csv"),
            index=False,
        )
        print("number of smiles with score > 0.9")
        print(
            len(df_smiles[df_smiles["scores_chargeS1"] > 0.9]),
            len(df_smiles[df_smiles["scores_chargeT1"] > 0.9]),
            len(df_smiles[df_smiles["scores_chargeT2"] > 0.9]),
        )
    # smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
