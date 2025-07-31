import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from navidiv.scorer import BaseScore


def get_fingerprints(molecules):
    """Generate Morgan fingerprints for a list of molecules."""
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


class OriginalSimScorer(BaseScore):
    """Handles fragment scoring and analysis for molecular datasets."""

    def __init__(
        self,
        threshold: float = 0.8,
        output_path: str | None = None,
        smiles_list_to_compare_to: list[str] | None = None,
    ) -> None:
        """Initialize FragmentScore.

        Args:
            min_count_fragments (int): Minimum count for fragments to be
                considered.
            output_path (str | None): Path to save output files.
        """
        super().__init__(output_path=output_path)
        self._csv_name = "Original_similarity"
        self._similarity = []
        self.threshold = threshold
        self._min_count_fragments = 0
        self.smiles_list_to_compare_to = smiles_list_to_compare_to
        
        # Create index mapping for faster lookups
        self._smiles_to_compare_to_index = {
            smiles: idx for idx, smiles in enumerate(smiles_list_to_compare_to)
        }
        
        self.original_fps = get_fingerprints(
            [
                Chem.MolFromSmiles(smiles)
                for smiles in smiles_list_to_compare_to
                if smiles != "None"
            ]
        )
        
        # Initialize mapping for generated smiles (will be set in get_count)
        self._smiles_list_index = {}

    def get_clusters(self, similarity):
        """Get clusters of similar molecules based on a similarity threshold."""
        # np.fill_diagonal(similarity, 0)
        clusters = set()

        for i in range(similarity.shape[0]):
            if not np.any(similarity[i, :] > self.threshold):
                clusters.add(i)
        return clusters

    def _get_max_similarity(self, smiles: str) -> float:
        """Get the maximum similarity of a molecule to the original dataset."""
        if not hasattr(self, "_smiles_list"):
            msg = (
                "SMILES list has not been calculated. "
                "Please run get_count first."
            )
            raise ValueError(msg)

        if smiles in self._smiles_list_index:
            smiles_index = self._smiles_list_index[smiles]
        else:
            return 0.0
        return np.max(self._similarity_to_original[smiles_index, :])

    def get_count(self, smiles_list: list[str]) -> tuple[pd.DataFrame, None]:
        """Calculate the percentage of each fragment in the dataset.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            tuple: DataFrame with fragment info, None (for compatibility)
        """
        self._mol_smiles = [
            Chem.MolFromSmiles(smiles)
            for smiles in smiles_list
            if smiles != "None"
        ]
        self._mol_smiles = [mol for mol in self._mol_smiles if mol is not None]
        self._smiles_list = [Chem.MolToSmiles(mol) for mol in self._mol_smiles]

        # Create index mapping for faster lookups
        self._smiles_list_index = {
            smiles: idx for idx, smiles in enumerate(self._smiles_list)
        }

        search_fps = get_fingerprints(self._mol_smiles)

        self._similarity_to_original = calculate_similarity(
            search_fps, self.original_fps
        )

        clusters_smiles = self.smiles_list_to_compare_to
        fragments, over_represented_fragments = self._from_list_to_count_df(
            self._smiles_list,
            clusters_smiles,
            total_number_of_ngrams=len(self._smiles_list),
        )
        self._fragments_df = fragments
        self._fragments_df["max_similarity"] = self._fragments_df[
            "Substructure"
        ].apply(lambda x: self._get_max_similarity(x))
        return fragments, over_represented_fragments

    def _count_pattern_occurrences(
        self, _: list[str], reference_smiles: str
    ) -> int:
        """Count molecules similar to a reference molecule in original dataset.

        Args:
            _: Not used in similarity scoring but kept for interface
                compatibility.
            reference_smiles: Reference molecule to compare against.

        Returns:
            Number of molecules similar to the reference in original dataset.
        """
        if reference_smiles not in self._smiles_to_compare_to_index:
            return 0
            
        reference_index = self._smiles_to_compare_to_index[reference_smiles]
        return 1 + len(
            [
                i
                for i in range(self._similarity_to_original.shape[0])
                if (
                    self._similarity_to_original[i, reference_index]
                    > self.threshold
                )
            ]
        )

    def _comparison_function(
        self,
        smiles: str | None = None,
        fragment: str | None = None,
    ) -> bool:
        """Check if molecules are similar to same molecules in original set.

        For original similarity scoring, this checks if both molecules have
        similar counterparts in the original dataset.
        """
        if fragment not in self._smiles_to_compare_to_index:
            return False
        if smiles not in self._smiles_list_index:
            return False
            
        reference_index = self._smiles_to_compare_to_index[fragment]
        smiles_index = self._smiles_list_index[smiles]
        return (
            self._similarity_to_original[smiles_index, reference_index]
            > self.threshold
        )

    def additional_metrics(self) -> dict[str, float]:
        """Calculate additional metrics for the scorer."""
        mean_distance = np.mean(self._similarity_to_original)
        std_distance = np.std(self._similarity_to_original)
        molecules_above_threshold = np.sum(
            np.any(self._similarity_to_original > self.threshold, axis=1)
        )
        return {
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "molecules_above_threshold": molecules_above_threshold,
        }

    def get_score(
        self,
        smiles_list: list[str],
        scores: list[float],
        additional_columns_df: {} = {},
    ) -> pd.DataFrame:
        """Get the score for the fragments.

        Args:
            smiles_list (list[str]): List of generated SMILES strings.
            scores (list[float]): List of scores for the generated SMILES.

        Returns:
            pd.DataFrame: DataFrame with score metrics.
        """
        self.get_count(smiles_list)
        self.add_score_metrics(smiles_list, scores, additional_columns_df)
        dict_results = {}
        dict_results = {**dict_results, **self.additional_metrics()}

        return dict_results
