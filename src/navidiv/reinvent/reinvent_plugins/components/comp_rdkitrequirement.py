"""Compute scores with Cheminformatics requirements"""

from __future__ import annotations

__all__ = ["rdkitrequirement"]
import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache

logger = logging.getLogger("reinvent")


def add_tag(label: str, text: str = "True"):
    """A simple decorator to tag a class"""

    def wrapper(cls):
        setattr(cls, label, text)
        return cls

    return wrapper


@add_tag("__parameters")
@dataclass
class Params:
    def __init__(
        self,
        max_number_of_rings: int,
        max_num_atoms: int,
        max_ring_size: int,
    ) -> None:
        self.max_number_of_rings = max_number_of_rings[0]
        self.max_num_atoms = max_num_atoms[0]
        self.max_ring_size = max_ring_size[0]


@add_tag("__component")
class rdkitrequirement:
    def __init__(self, params: Params):
        # mp.set_start_method("spawn", force=True)
        logger.info("Using rdkit requirement")
        self.max_number_of_rings = params.max_number_of_rings
        self.max_num_atoms = params.max_num_atoms
        self.max_ring_size = params.max_ring_size

    @molcache
    def __call__(self, mols: list[Chem.Mol]) -> np.array:
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(0.0)
                continue
            num_atoms = mol.GetNumAtoms()
            if num_atoms > self.max_num_atoms:
                scores.append(0.0)
                continue
            for a in mol.GetRingInfo().AtomRings():
                if len(set(a)) > self.max_ring_size:
                    scores.append(0.0)
                    continue
            scores.append(1.0)
        return ComponentResults([np.array(scores)])
