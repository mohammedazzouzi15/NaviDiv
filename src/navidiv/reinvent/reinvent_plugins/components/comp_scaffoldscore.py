"""Compute scores considering thetransition charge embeddings"""

from __future__ import annotations

__all__ = ["scaffoldscore"]
import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from reinvent.scoring.utils import suppress_output
from reinvent_plugins.components.component_results import ComponentResults

from reinvent_plugins.normalize import normalize_smiles

from .model import ModelLoader

logger = logging.getLogger("reinvent")


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


@add_tag("__parameters")
@dataclass
class Params:
    def __init__(self, checkpoints) -> None:
        self.checkpoints = checkpoints[0]


@add_tag("__component")
class scaffoldscore:
    def __init__(self, params: Params):
        # mp.set_start_method("spawn", force=True)

        logger.info("Using scaffold score")
        logger.info(f"Checkpoints: {params.checkpoints}")
        self.number_of_endpoints = 1
        self.smiles_type = "rdkit_smiles"
        self.model_loader_list = []
        self.checkpoints = eval(params.checkpoints)
        """[
            "/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/S1_exc/PainnEmbed/epoch=20-val_mae=0.15-val_mse=0.00.ckpt",
            "/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/T1_exc/PainnEmbed/epoch=90-val_mae=0.15-val_mse=0.00.ckpt",
            "/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/T2_exc/PainnEmbed/epoch=74-val_mae=0.21-val_mse=0.00.ckpt",
        ]"
        """
        for checkpoint in self.checkpoints:
            logger.info(f"Loading model from {checkpoint}")
            model_loader = ModelLoader()
            model_loader.load_model(checkpoint)
            self.model_loader_list.append(model_loader)

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> np.array:
        with suppress_output():
            scores = []
            for smiles in smilies:
                score_int = []
                for model_loader in self.model_loader_list:
                    score_int.append(get_score(smiles, model_loader))
                score = np.max(score_int)
                scores.append(score)
            return ComponentResults([np.array(scores)])
