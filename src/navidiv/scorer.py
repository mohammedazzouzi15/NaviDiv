
import logging
import time  
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem


class BaseScore:
    """Handles fragment scoring and analysis for molecular datasets."""

    def __init__(
        self,
        output_path: str | None = None,
    ) -> None:
        """Initialize the BaseScore class.

        Args:
            output_path (str | None): Path to save output files.
        """
        self._output_path = output_path
        self.add_num_atoms = True
        self._csv_name = "Default"
        self._fragments_df = None
        self._filtered_fragments = None  # renamed from selected_fragments
        self._min_count_fragments = 3
        self.overrepresented_fragments_min_perc = 20
        self.overrepresented_fragments = None
        self.selection_criteria = {
            "Count_perc_per_molecule": 1,
            "Count_perc": 1,
            "diff_median_score": -5,
            "median_score_fragment": 0,
        }

    def update_selection_criteria(
        self,
        selection_criteria: dict[str, float],
    ) -> None:
        """Update the selection criteria for overrepresented fragments.

        Args:
            selection_criteria (dict[str, float]): Dictionary of selection
                criteria.

        Example:
            selection_criteria = {
                "Count_perc_per_molecule": 1,
                "Count_perc": 1,
                "diff_median_score": -5,
                "median_score_fragment": 0,
            }
            fragment_scorer.update_selection_criteria(selection_criteria)
        """
        self.selection_criteria = selection_criteria

    def get_count(self, smiles_list: list[str]) -> tuple[pd.DataFrame, None]:
        """Calculate the percentage of each fragment in the dataset.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            tuple: DataFrame with fragment info, None (for compatibility)
        """
        raise NotImplementedError(
            "get_count method must be implemented in subclasses."
        )

    def _comparison_function(
        self,
        smiles: str | None = None,
        fragment: str | None = None,
        mol: Chem.Mol | None = None,
    ) -> bool:
        """Check if the fragment is present in the SMILES string or molecule."""
        raise NotImplementedError(
            "_comparison_function method must be implemented in subclasses."
        )

    def additional_metrics(self):
        """Calculate additional metrics for the fragments."""
        return {}

    def add_score_metrics(
        self,
        smiles_list: list[str],
        scores: list[float],
        additional_columns_df: {} = {},
    ) -> pd.DataFrame:
        """Add score metrics to the DataFrame.

        Args:
            smiles_list (list[str]): List of generated SMILES strings.
            scores (list[float]): List of scores for the generated SMILES.

        Returns:
            pd.DataFrame: DataFrame with score metrics.
        """

        def process_fragment(
            fragment: str, smiles: list[str], scores: list[float]
        ) -> list[float]:
            # ...existing code...
            molecule_scores = [
                (sm, score)
                for score, sm in zip(scores, smiles, strict=False)
                if sm != "None"
            ]
            scores_all = [score for _, score in molecule_scores]
            contains_fragment_dict = {}
            for smiles, score in molecule_scores:
                if smiles not in contains_fragment_dict:
                    contains_fragment_dict[smiles] = self._comparison_function(
                        smiles=smiles, fragment=fragment
                    )

            scores_with_fragment = [
                score
                for smiles, score in molecule_scores
                if contains_fragment_dict[smiles]  # type: ignore
            ]
            scores_without_fragment = [
                score
                for smiles, score in molecule_scores
                if not contains_fragment_dict[smiles]  # type: ignore
            ]
            molecules_with_fragment = [
                smiles
                for smiles, score in molecule_scores
                if contains_fragment_dict[smiles]
            ]  # type: ignore

            if len(scores_with_fragment) == 0:
                return [0.0, 0.0, 0.0, []]
            try:
                median_score = np.median(scores_all)
                median_score_fragment = np.median(scores_with_fragment)

                median_score_not_fragment = np.median(scores_without_fragment)
                smiles_with_fragment = molecules_with_fragment
            except (ValueError, TypeError):
                logging.exception(
                    "Error calculating score metrics for fragment: %s",
                    fragment,
                )
                return [0.0, 0.0, 0.0, []]
            return [
                median_score,
                median_score_fragment,
                median_score_not_fragment,
                smiles_with_fragment,
            ]

        self._fragments_df = self._fragments_df[
            self._fragments_df["Count"] > self._min_count_fragments
        ]  # select fragments with count greater than min_count_fragments
        if self._fragments_df.shape[0] == 0:
            logging.warning(
                "No fragments found with count greater than %d",
                self._min_count_fragments,
            )
            return self._fragments_df
        self._fragments_df[
            [
                "median_score",
                "median_score_fragment",
                "median_score_not_fragment",
                "molecules_with_fragment",
            ]
        ] = self._fragments_df.apply(
            lambda row: pd.Series(
                process_fragment(
                    row["Substructure"],
                    smiles_list,
                    scores,
                )
            ),
            axis=1,
        )
        if self.add_num_atoms:
            num_atoms = []
            for substructure in self._fragments_df["Substructure"]:
                mol = Chem.MolFromSmarts(substructure)
                if mol is not None:
                    num_atoms.append(mol.GetNumAtoms())
                else:
                    num_atoms.append(0)
            self._fragments_df["num_atoms"] = num_atoms
        self._fragments_df["diff_median_score"] = (
            self._fragments_df["median_score_fragment"]
            - self._fragments_df["median_score_not_fragment"]
        )
        # if file exists, append to it
        for col, value in additional_columns_df.items():
            self._fragments_df[col] = value
        self._fragments_df.to_csv(
            f"{self._output_path}/{self._csv_name}_with_score.csv",
            index=False,
            mode="a",
            header=not pd.io.common.file_exists(
                f"{self._output_path}/{self._csv_name}_with_score.csv"
            ),
        )
        return self._fragments_df

    def plot_count_histogram(
        self,
        title: str = "Fragment Count Histogram",
        xlabel: str = "Fragment Count",
        ylabel: str = "Frequency",
        bins: int = 50,
    ) -> None:
        """Plot a histogram of fragment counts.

        Args:
            title (str): Title of the histogram.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            bins (int): Number of bins in the histogram.
        """
        if self._fragments_df is None:
            raise ValueError("Fragments DataFrame is not initialized.")
        fig, ax = plt.subplots()
        ax = self._fragments_df["Count"].plot.hist(
            bins=bins,
            alpha=0.7,
            color="blue",
            edgecolor="black",
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.savefig(f"{self._output_path}/{self._csv_name}_histogram.png")
        plt.close(fig)

    def select_overrepresented_fragments(self):
        """Select overrepresented fragments based on the defined criteria."""
        self._filtered_fragments = self._fragments_df.copy()
        if self._fragments_df is None:
            raise ValueError("Fragments DataFrame is not initialized.")
        for col, value in self.selection_criteria.items():
            if col not in self._filtered_fragments.columns:
                continue
            self._filtered_fragments = self._filtered_fragments[
                self._filtered_fragments[col] > value
            ]
        self._filtered_fragments.to_csv(
            f"{self._output_path}/{self._csv_name}_selected_fragments.csv",
            index=False,
            mode="a",
            header=not pd.io.common.file_exists(
                f"{self._output_path}/{self._csv_name}_selected_fragments.csv"
            ),
        )
        return self._filtered_fragments

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
        start_time = time.time()
        self.get_count(smiles_list)
        time_to_get_count = time.time()
        print(
            f"time to get count: {time_to_get_count - start_time:.2f} seconds"
        )
        unique_fragments = self._fragments_df["Substructure"].to_list()
        self.add_score_metrics(smiles_list, scores, additional_columns_df)
        if self.total_number_of_fragments == 0:
            unicity_ratio = 0.0
        else:
            unicity_ratio = (
                len(unique_fragments) / self.total_number_of_fragments * 100
            )
        for col, value in additional_columns_df.items():
            self._fragments_df[col] = value
        dict_results = {
            "Percentage of Unique Fragments": unicity_ratio,
            "Total Number of Fragments": self.total_number_of_fragments,
            "Number of Unique Fragments": len(unique_fragments),
            "Unique Fragments": unique_fragments,
        }
        dict_results = {**dict_results, **self.additional_metrics()}
        elapsed_time = time.time() - time_to_get_count
        logging.info(
            f"time to get added score metrics {elapsed_time:.2f} seconds"
        )

        return dict_results

    def _from_list_to_count_df(
        self,
        smiles_list: list[str],
        substructure_list: list[str],
        total_number_of_ngrams: int | None = None,
    ) -> tuple[pd.DataFrame, None]:
        ngrams_counter = Counter(substructure_list)
        if total_number_of_ngrams is None:
            total_number_of_ngrams = len(substructure_list)
        self.total_number_of_fragments = total_number_of_ngrams
        ngrams_df = pd.DataFrame(
            {
                "Substructure": list(ngrams_counter.keys()),
                "Count": list(ngrams_counter.values()),
            }
        )
        ngrams_df["Count ratio"] = (
            ngrams_df["Count"] / total_number_of_ngrams
        ) * 100
        ngrams_df = ngrams_df.sort_values(by="Count ratio", ascending=False)
        ngrams_df = ngrams_df.reset_index(drop=True)
        ngrams_df["Number_of_molecules_with_fragment"] = ngrams_df[
            "Substructure"
        ].apply(lambda x: self._count_pattern_occurrences(smiles_list, x))

        ngrams_df["Ratio_of_Molecules_with_Fragment"] = (
            ngrams_df["Number_of_molecules_with_fragment"] / len(smiles_list)
        ) * 100
        if self._output_path:
            try:
                ngrams_df.to_csv(
                    f"{self._output_path}/{self._csv_name}.csv",
                    index=False,
                )
            except OSError:
                logging.exception(
                    "Error saving fragments CSV for output_path: %s",
                    self._output_path,
                )
        self._fragments_df = ngrams_df
        return ngrams_df, None

    def _count_pattern_occurrences(
        self, smiles_list: list[str], pattern: str
    ) -> int:
        """Count occurrences of a pattern in the dataset.

        Args:
            smiles_list: List of SMILES strings.
            pattern: Pattern to search for (interpretation depends on scorer
                type).

        Returns:
            Number of molecules containing/matching the pattern.

        Note:
            The interpretation of 'pattern' depends on the scorer type:
            - Fragment scorers: structural fragment
            - N-gram scorers: string substring
            - Similarity scorers: reference molecule for similarity comparison
        """
        msg = (
            "_count_pattern_occurrences method must be implemented "
            "in subclasses."
        )
        raise NotImplementedError(msg)
