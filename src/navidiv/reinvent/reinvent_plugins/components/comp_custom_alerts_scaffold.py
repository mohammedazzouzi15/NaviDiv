"""Compute scores with RDKit's QED"""

__all__ = ["CustomAlertsScaffold"]
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic.dataclasses import dataclass
from rdkit.Chem.Scaffolds import MurckoScaffold

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
REPL = Chem.MolFromSmarts("[*]")


def get_scaffold(mol, cases: str = "all"):
    """Get the scaffold of a molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        cases (str): Cases to consider for scaffold generation.
            "bajorath": Use Bajorath's method.
            "real_bm": Use real Bajorath's method.
            "csk": Use CSK method.
            "csk_bm": Use CSK with Bajorath's method.
            "murcko": Use Murcko's method.
    """
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)  # important for canonization of CSK!
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    try:
        match cases:
            case "bajorath":
                scaff = AllChem.DeleteSubstructs(scaff, PATT)
            case "real_bm":
                scaff = AllChem.ReplaceSubstructs(scaff, PATT, REPL)[0]
            case "csk":
                scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
            case "csk_bm":
                scaff = AllChem.DeleteSubstructs(scaff, PATT)
                scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
    except Exception as e:
        print(f"Error in get_scaffold: {e}")
        return None
    return Chem.MolToSmiles(scaff)


@add_tag("__parameters")
@dataclass
class Parameters:
    frag: List[List[str]]
    type: List[str]


@add_tag("__component", "filter")
class CustomAlertsScaffold:
    def __init__(self, params: Parameters):
        # FIXME: read from file?
        self.frag = params.frag[0]  # assume there is only one endpoint...
        self.type = params.type[0]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        match = []

        for mol in mols:
            if not mol:
                score = False  # FIXME: rely on pre-processing?
            else:
                score = any(
                    [get_scaffold(mol, cases=self.type) == subst for subst in self.frag]
                )

            match.append(score)

        scores = [1 - m for m in match]

        return ComponentResults([np.array(scores, dtype=float)])
