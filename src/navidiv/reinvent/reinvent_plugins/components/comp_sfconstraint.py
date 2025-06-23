__all__ = ["RotRingBond", "CountRings"]
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem

from reinvent_plugins.mol_cache import molcache


from reinvent_plugins.components.component_results import ComponentResults


def add_tag(label: str, text: str = "True"):
    """A simple decorator to tag a class"""

    def wrapper(cls):
        setattr(cls, label, text)
        return cls

    return wrapper


def count_rotatable_bonds_between_rings(smiles):
    # Parse the SMILES string to generate a molecule object
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    rotatable_bonds_between_rings = 0

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if (
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE
            and bond.GetBeginAtom().GetDegree() > 1
            and bond.GetEndAtom().GetDegree() > 1
        ):
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            if begin_atom.IsInRing() and end_atom.IsInRing():
                rotatable_bonds_between_rings += 1

    return rotatable_bonds_between_rings




def count_single_bonds_between_aromatic_rings(smiles):
    # Parse the SMILES string to generate a molecule object
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    single_bonds_between_aromatic_rings = 0

    for bond in mol.GetBonds():
        # Check if the bond is a single bond
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        # Skip bonds that are part of a ring
        if bond.IsInRing():
            continue

        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        # Check if both atoms are in aromatic rings
        if (
            begin_atom.GetIsAromatic()
            and begin_atom.IsInRing()
            and end_atom.GetIsAromatic()
            and end_atom.IsInRing()
        ):
            single_bonds_between_aromatic_rings += 1

    return single_bonds_between_aromatic_rings


def count_number_of_rings(mol: Chem.Mol) -> int:
    """
    Counts the number of rings in an RDKit molecule.

    Parameters:
        mol (Chem.Mol): The input RDKit molecule.

    Returns:
        int: The number of rings in the molecule.
    """
    if mol is None:
        return 0

    return mol.GetRingInfo().NumRings()


@add_tag("__parameters")
@dataclass
class Parameters:
    n_bonds: int = 3
    n_rings: int = 4


@add_tag("__component")
class RotRingBond:
    def __init__(self, parameters: Parameters):
        self.n_bonds = parameters.n_bonds

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for mol in mols:
            try:
                score = count_single_bonds_between_aromatic_rings(Chem.MolToSmiles(mol))
            except ValueError:
                score = 0

            if score == 1 or score == 2:
                score = 1
            else:
                score = 0
            scores.append(score)
        return ComponentResults([np.array(scores)])


@add_tag("__component")
class CountRings:
    def __init__(self, parameters: Parameters):
        self.n_rings = parameters.n_rings

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for mol in mols:
            score = count_number_of_rings(mol)
            if score > self.n_rings:
                score = 0
            scores.append(score)
        return ComponentResults([np.array(scores)])
