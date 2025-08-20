import logging
import time
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

# Set up logger
logger = logging.getLogger(__name__)


class BaseScore:
    """Handles fragment scoring and analysis for molecular datasets."""

    # Constants
    DEFAULT_MIN_COUNT_FRAGMENTS = 3
    DEFAULT_OVERREP_MIN_PERC = 20
    DEFAULT_CSV_NAME = "Default"
    DEFAULT_HISTOGRAM_BINS = 50
    DEFAULT_SELECTION_CRITERIA = {
        "Count_perc_per_molecule": 1,
        "Count_perc": 1,
        "diff_median_score": -5,
        "median_score_fragment": 0,
    }

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
        self._csv_name = self.DEFAULT_CSV_NAME
        self._fragments_df = None
        self._filtered_fragments = None  # renamed from selected_fragments
        self._min_count_fragments = self.DEFAULT_MIN_COUNT_FRAGMENTS
        self.overrepresented_fragments_min_perc = self.DEFAULT_OVERREP_MIN_PERC
        self.overrepresented_fragments = None
        self.selection_criteria = self.DEFAULT_SELECTION_CRITERIA.copy()

    def _get_output_path(self, filename_suffix: str) -> str:
        """Generate output file path with consistent naming."""
        return f"{self._output_path}/{self._csv_name}_{filename_suffix}.csv"

    def _save_to_csv(
        self,
        dataframe: pd.DataFrame,
        filename_suffix: str,
        append: bool = True,
    ) -> None:
        """Save DataFrame to CSV with consistent formatting."""
        if not self._output_path:
            return

        filepath = self._get_output_path(filename_suffix)
        try:
            dataframe.to_csv(
                filepath,
                index=False,
                mode="a" if append else "w",
                header=not pd.io.common.file_exists(filepath)
                if append
                else True,
            )
        except OSError:
            logger.exception("Error saving CSV file: %s", filepath)

    def _validate_fragments_df(self) -> None:
        """Validate that fragments DataFrame is initialized."""
        if self._fragments_df is None:
            msg = "Fragments DataFrame is not initialized."
            raise ValueError(msg)

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
        msg = "get_count method must be implemented in subclasses."
        raise NotImplementedError(msg)

    def _comparison_function(
        self,
        smiles: str | None = None,
        fragment: str | None = None,
        mol: Chem.Mol | None = None,
    ) -> bool:
        """Check if the fragment is present in the SMILES string or molecule."""
        msg = "_comparison_function method must be implemented in subclasses."
        raise NotImplementedError(msg)

    def additional_metrics(self) -> dict[str, Any]:
        """Calculate additional metrics for the fragments."""
        return {}

    def add_score_metrics(
        self,
        smiles_list: list[str],
        scores: list[float],
        additional_columns_df: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Add score metrics to the DataFrame.

        Args:
            smiles_list (list[str]): List of generated SMILES strings.
            scores (list[float]): List of scores for the generated SMILES.
            additional_columns_df (dict[str, Any] | None): Additional columns
                to add to the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with score metrics.
        """
        if additional_columns_df is None:
            additional_columns_df = {}

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
            for mol_smiles, _score in molecule_scores:
                if mol_smiles not in contains_fragment_dict:
                    contains_fragment_dict[mol_smiles] = (
                        self._comparison_function(
                            smiles=mol_smiles, fragment=fragment
                        )
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
                logger.exception(
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
            logger.warning(
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
        self._validate_fragments_df()
        fig, ax = plt.subplots()
        self._fragments_df["Count"].plot.hist(
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

    def select_overrepresented_fragments(self) -> pd.DataFrame:
        """Select overrepresented fragments based on the defined criteria."""
        self._validate_fragments_df()
        self._filtered_fragments = self._fragments_df.copy()
        for col, value in self.selection_criteria.items():
            if col not in self._filtered_fragments.columns:
                continue
            self._filtered_fragments = self._filtered_fragments[
                self._filtered_fragments[col] > value
            ]
        self._save_to_csv(self._filtered_fragments, "selected_fragments")
        return self._filtered_fragments

    def get_score(
        self,
        smiles_list: list[str],
        scores: list[float],
        additional_columns_df: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get the score for the fragments.

        Args:
            smiles_list (list[str]): List of generated SMILES strings.
            scores (list[float]): List of scores for the generated SMILES.
            additional_columns_df (dict[str, Any] | None): Additional columns
                to add to the DataFrame.

        Returns:
            dict[str, Any]: Dictionary with score metrics.
        """
        if additional_columns_df is None:
            additional_columns_df = {}

        start_time = time.time()
        self.get_count(smiles_list)
        time_to_get_count = time.time()
        logger.info(
            "Time to get count: %.2f seconds", time_to_get_count - start_time
        )
        unique_fragments = self._fragments_df["Substructure"].to_list()
        self.add_score_metrics(smiles_list, scores, additional_columns_df)

        unicity_ratio = (
            0.0
            if self.total_number_of_fragments == 0
            else len(unique_fragments) / self.total_number_of_fragments * 100
        )

        for col, value in additional_columns_df.items():
            self._fragments_df[col] = value

        elapsed_time = time.time() - start_time

        dict_results = {
            "Percentage of Unique Fragments": unicity_ratio,
            "Total Number of Fragments": self.total_number_of_fragments,
            "Number of Unique Fragments": len(unique_fragments),
            "Unique Fragments": unique_fragments,
            "Elapsed Time": elapsed_time,
            "Elapsed Time_to get count": time_to_get_count
        }
        dict_results = {**dict_results, **self.additional_metrics()}

        logger.info(
            "Time to get added score metrics: %.2f seconds", elapsed_time
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
        ngrams_df = ngrams_df[
            ngrams_df["Count"] > self._min_count_fragments
        ]  # select fragments with count greater than min_count_fragments
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
                logger.exception(
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
