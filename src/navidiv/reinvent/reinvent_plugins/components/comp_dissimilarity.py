"""Tanimoto similarity and Jaccard distance"""

from __future__ import annotations

__all__ = ["Dissimilarity"]

from typing import List
import logging
import numpy as np
from pydantic.dataclasses import dataclass

from rdkit import DataStructs
from rdkit import Chem
from .component_results import ComponentResults
from .add_tag import add_tag
from rdkit.Chem import rdFingerprintGenerator


logger = logging.getLogger("reinvent")


def get_fingerprints(
    smilies: list[str],
) -> list[DataStructs.ExplicitBitVect]:
    """Generate Morgan fingerprints for a list of molecules."""
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smilies]
    # Remove None values (invalid SMILES)
    molecules = [
        mol if mol is not None else Chem.MolFromSmiles("C") for mol in molecules
    ]

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=5, fpSize=2048, countSimulation=True
    )
    fingerprints = [mfpgen.GetFingerprint(mol) for mol in molecules]
    return fingerprints


def calculate_similarity(fp_list1, fp_list2):
    """Calculate the Tanimoto similarity between two sets of fingerprints."""
    return np.array(
        [DataStructs.BulkTanimotoSimilarity(fp1, fp_list2) for fp1 in fp_list1]
    )


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    frag: List[List[str]]
    threshold: List[float]


@add_tag("__component", "filter")
class Dissimilarity:
    """Compute the Tanimoto similarity

    Scoring component to compute the Tanimoto similarity between the provided
    SMILES and the generated molecule.  Supports fingerprint radius, count
    fingerprints and the use of pharmacophore-like features (see
    https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z).
    """

    def __init__(self, params: Parameters):
        self.frag = params.frag[0]
        self.fp_to_avoid = get_fingerprints(self.frag)
        self.threshold = params.threshold[0]

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []
        query_fingerprints = get_fingerprints(smilies)
        similarity_matrix = calculate_similarity(query_fingerprints, self.fp_to_avoid)
        similarity_matrix = np.max(similarity_matrix, axis=1)
        for i in range(len(similarity_matrix)):
            if similarity_matrix[i] > self.threshold:
                scores.append(0.0)
            else:
                scores.append(1.0)

        return ComponentResults([np.array(scores, dtype=float)])
