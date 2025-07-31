

import itertools

import pandas as pd

from navidiv.scorer import BaseScore
from navidiv.stringbased.utils import levenshtein


class NgramScorer(BaseScore):
    """Handles fragment scoring and analysis for molecular datasets."""

    def __init__(
        self,
        ngram_size: int = 10,
        output_path: str | None = None,
    ) -> None:
        """Initialize FragmentScore.

        Args:
            min_count_fragments (int): Minimum count for fragments to be
                considered.
            output_path (str | None): Path to save output files.
        """
        super().__init__(output_path=output_path)
        self._min_count_fragments = 10
        self._csv_name = f"ngrams_{ngram_size}"
        self.ngram_size = ngram_size
        self.add_num_atoms = False

    def get_count(self, smiles_list: list[str]) -> tuple[pd.DataFrame, None]:
        """Calculate the percentage of each fragment in the dataset.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            tuple: DataFrame with fragment info, None (for compatibility)
        """
        self._ngrams = []
        for smiles in smiles_list:
            self._ngrams.extend(
                [
                    smiles[i : i + self.ngram_size]
                    for i in range(len(smiles) - self.ngram_size + 1)
                ]
            )
        fragments, over_represented_fragments = self._from_list_to_count_df(
            smiles_list,
            self._ngrams,
        )
        self._fragments_df = fragments
        return fragments, over_represented_fragments

    def _count_pattern_occurrences(self, smiles_list: list[str], ngram: str) -> int:
        """Count occurrences of an n-gram pattern in the dataset."""
        return sum(1 for smiles in smiles_list if ngram in smiles)

    def _comparison_function(
        self,
        smiles: str | None = None,
        fragment: str | None = None,
        mol=None,
    ) -> bool:
        """Check if the fragment is present in the SMILES string or molecule."""
        if smiles is None:
            return False

        if fragment is None:
            return False

        return fragment in smiles

    def additional_metrics(self):
        """Calculate additional metrics for the scorer."""
        from collections import Counter

        count_ngrams = Counter(self._ngrams)
        ngrams_above_10 = {
            ngram: count for ngram, count in count_ngrams.items() if count > 10
        }
        ngrams_above_5 = {
            ngram: count for ngram, count in count_ngrams.items() if count > 5
        }
        ngrams_below_2 = {
            ngram: count for ngram, count in count_ngrams.items() if count < 2
        }

        return {
            "Appeared more than 10 times": len(ngrams_above_10),
            "Appeared more than 5 times": len(ngrams_above_5),
            "Appeared once": len(ngrams_below_2),
        }

    def old_additional_metrics(self):
        """Calculate additional metrics for the scorer."""
        import random

        levenshtein_distances = []
        ngrams = set(self._ngrams)
        print(
            f"Calculating Levenshtein distances for {len(ngrams)} unique ngrams."
        )
        random_samples = random.sample(list(ngrams), min(1000, len(ngrams)))
        ngrams = random_samples if len(ngrams) > 1000 else ngrams

        for i, j in itertools.combinations(ngrams, 2):
            if i != j:
                distance = levenshtein(i, j)
                levenshtein_distances.append(distance)
        if levenshtein_distances:
            mean_distance = sum(levenshtein_distances) / len(
                levenshtein_distances
            )
            median_distance = sorted(levenshtein_distances)[
                len(levenshtein_distances) // 2
            ]
            return {
                "mean_levenshtein_distance": mean_distance,
                "median_levenshtein_distance": median_distance,
            }
        return {
            "mean_levenshtein_distance": None,
            "median_levenshtein_distance": None,
        }
