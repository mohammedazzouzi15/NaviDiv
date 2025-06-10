"""Tanimoto similarity and Jaccard distance"""

from __future__ import annotations

__all__ = ["TanimotoDisimilarity"]


import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def get_fingerprints(
    smilies: list[str],
) -> list[rdFingerprintGenerator.Fingerprint]:
    """Generate Morgan fingerprints for a list of molecules."""
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smilies]
    # Remove None values (invalid SMILES)
    molecules = [
        mol if mol is not None else Chem.MolFromSmiles("C")
        for mol in molecules
    ]
    if not molecules:
        return []
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


if __name__ == "__main__":
    # Example usage
    smiles1 = ["CCO", "CCN", "CCCNCOO"]
    smiles2 = ["CCO", "CCN", "CCCC"]

    fps1 = get_fingerprints(smiles1)
    fps2 = get_fingerprints(smiles2)

    similarity_matrix = calculate_similarity(fps1, fps2)
    print("Tanimoto Similarity Matrix:")
    print(similarity_matrix)
    # Example usage
