"""Compute scores with RDKit's QED"""

__all__ = ["CustomAlertsNgrams"]
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    frag: List[List[str]]


@add_tag("__component", "filter")
class CustomAlertsNgrams:
    def __init__(self, params: Parameters):
        # FIXME: read from file?
        self.frag = params.frag[0]  # assume there is only one endpoint...
        self.smiles_type = "rdkit_smiles"

    def __call__(self, smilies: List[str]) -> np.array:
        match = []
        for smi in smilies:
            score = any([ngram in smi for ngram in self.frag])
            match.append(score)

        scores = [1 - m for m in match]

        return ComponentResults([np.array(scores, dtype=float)])
