from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import HDBSCAN, AgglomerativeClustering

from navidiv.scorer import BaseScore

# Define available similarity metrics and clustering methods
SimilarityMetric = Literal["tanimoto", "dice", "cosine", "euclidean"]
ClusteringMethod = Literal["threshold", "hierarchical", "dbscan"]


class SimilarityCalculator:
    """Handles fingerprint generation and similarity calculations."""

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        fp_size: int = 2048,
    ) -> None:
        """Initialize similarity calculator with fingerprint parameters."""
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.fp_size = fp_size
        self._setup_generator()

    def _setup_generator(self) -> None:
        """Setup fingerprint generator based on type."""
        if self.fingerprint_type == "morgan":
            self.generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius, fpSize=self.fp_size, countSimulation=True
            )
        elif self.fingerprint_type == "rdkit":
            self.generator = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=self.fp_size
            )
        else:
            msg = f"Unsupported fingerprint type: {self.fingerprint_type}"
            raise ValueError(msg)

    def get_fingerprints(self, molecules):
        """Generate fingerprints for a list of molecules."""
        return [self.generator.GetFingerprint(mol) for mol in molecules]

    def calculate_similarity(
        self, fp_list1, fp_list2, metric: SimilarityMetric = "tanimoto"
    ):
        """Calculate similarity between two sets of fingerprints."""
        if metric == "tanimoto":
            return np.array(
                [
                    DataStructs.BulkTanimotoSimilarity(fp1, fp_list2)
                    for fp1 in fp_list1
                ]
            )
        if metric == "dice":
            return np.array(
                [
                    DataStructs.BulkDiceSimilarity(fp1, fp_list2)
                    for fp1 in fp_list1
                ]
            )
        msg = f"Unsupported similarity metric: {metric}"
        raise ValueError(msg)


class ClusteringManager:
    """Handles different clustering approaches."""

    def __init__(self, method: ClusteringMethod = "threshold", **kwargs):
        """Initialize clustering manager with specified method and parameters."""
        self.method = method
        self.params = kwargs

    def get_clusters(self, similarity_matrix: np.ndarray) -> set:
        """Get clusters based on the selected method."""
        if self.method == "threshold":
            return self._threshold_clustering(
                similarity_matrix, self.params.get("threshold", 0.8)
            )
        if self.method == "hierarchical":
            return self._hierarchical_clustering(
                similarity_matrix,
                self.params.get("n_clusters", 5),
                self.params.get("linkage", "average"),
            )
        if self.method == "dbscan":
            return self._dbscan_clustering(
                similarity_matrix,
                self.params.get("eps", 0.2),
                self.params.get("min_samples", 2),
            )
        msg = f"Unsupported clustering method: {self.method}"
        raise ValueError(msg)

    def _threshold_clustering(
        self, similarity: np.ndarray, threshold: float
    ) -> set:
        """Simple threshold-based clustering."""
        clusters = set()
        for i in range(similarity.shape[0]):
            if not np.any(similarity[i, :i] > threshold):
                clusters.add(i)
        return clusters

    def _hierarchical_clustering(
        self, similarity: np.ndarray, n_clusters: int, linkage: str
    ) -> set:
        """Hierarchical clustering using sklearn."""
        # Convert similarity to distance
        raise ValueError(
            "Hirarchical clustering is not supported in this version. "
            "Please use threshold clustering."
        )

    def _dbscan_clustering(
        self, similarity: np.ndarray, eps: float, min_samples: int
    ) -> set:
        """DBSCAN clustering using sklearn."""
        # Convert similarity to distance
        raise ValueError(
            "DBSCAN clustering is not supported in this version. "
            "Please use threshold clustering."
        )


class ClusterSimScorer(BaseScore):
    """Configurable molecular similarity scorer with clustering support."""

    def __init__(
        self,
        threshold: float = 0.8,
        output_path: str | None = None,
        similarity_metric: SimilarityMetric = "tanimoto",
        clustering_method: ClusteringMethod = "threshold",
        fingerprint_type: str = "morgan",
        fingerprint_radius: int = 2,
        fingerprint_size: int = 2048,
        **clustering_params,
    ) -> None:
        """Initialize ClusterSimScorer with configurable parameters.

        Args:
            threshold: Similarity threshold for clustering (used in threshold method).
            output_path: Path to save output files.
            similarity_metric: Similarity metric to use ("tanimoto", "dice").
            clustering_method: Clustering approach ("threshold", "hierarchical", "dbscan").
            fingerprint_type: Type of molecular fingerprint ("morgan", "rdkit").
            fingerprint_radius: Radius for Morgan fingerprints.
            fingerprint_size: Size of fingerprint bit vector.
            **clustering_params: Additional parameters for clustering methods.
        """
        super().__init__(output_path=output_path)
        self._csv_name = "clusters"
        self.threshold = threshold
        self.similarity_metric = similarity_metric
        self._min_count_fragments = 0
        self.fingerprint_type = fingerprint_type

        # Initialize similarity calculator
        self.similarity_calculator = SimilarityCalculator(
            fingerprint_type=fingerprint_type,
            radius=fingerprint_radius,
            fp_size=fingerprint_size,
        )

        # Initialize clustering manager
        clustering_params.setdefault("threshold", threshold)
        self.clustering_manager = ClusteringManager(
            method=clustering_method, **clustering_params
        )
        self._update_csv_name()

    def _update_csv_name(self) -> None:
        """Update the name of the CSV file."""
        self._csv_name = f"clusters_{self.similarity_metric}_{self.fingerprint_type}_{self.clustering_manager.method}_{self.threshold}.csv"

    def get_count(self, smiles_list: list[str]) -> tuple[pd.DataFrame, None]:
        """Calculate clusters and their statistics from SMILES list."""
        # Convert SMILES to molecules and filter valid ones
        self._mol_smiles = [
            mol
            for mol in [
                Chem.MolFromSmiles(smiles)
                for smiles in smiles_list
                if smiles != "None"
            ]
            if mol is not None
        ]

        # Sort by number of atoms for consistent ordering
        self._mol_smiles.sort(key=lambda x: x.GetNumAtoms())

        # Generate SMILES strings from molecules
        self._smiles_list = [Chem.MolToSmiles(mol) for mol in self._mol_smiles]

        # Calculate similarity matrix
        fingerprints = self.similarity_calculator.get_fingerprints(
            self._mol_smiles
        )
        self._similarity_matrix = (
            self.similarity_calculator.calculate_similarity(
                fingerprints, fingerprints, self.similarity_metric
            )
        )

        # Get clusters using the selected method
        clusters = self.clustering_manager.get_clusters(
            self._similarity_matrix
        )
        clusters_smiles = [self._smiles_list[i] for i in clusters]

        # Create fragment dataframe
        fragments, over_represented_fragments = self._from_list_to_count_df(
            self._smiles_list,
            clusters_smiles,
            total_number_of_ngrams=len(self._smiles_list),
        )
        self._fragments_df = fragments
        return fragments, over_represented_fragments

    def _count_pattern_occurrences(
        self, _smiles_list: list[str], reference_smiles: str
    ) -> int:
        """Count molecules similar to a reference molecule."""
        if reference_smiles not in self._smiles_list:
            return 0
        reference_index = self._smiles_list.index(reference_smiles)
        return 1 + len(
            [
                i
                for i in range(self._similarity_matrix.shape[0])
                if self._similarity_matrix[reference_index, i] > self.threshold
            ]
        )

    def _comparison_function(
        self,
        smiles: str | None = None,
        fragment: str | None = None,
    ) -> bool:
        """Check if molecules are similar based on threshold."""
        if not fragment or not smiles:
            return False
        if (
            fragment not in self._smiles_list
            or smiles not in self._smiles_list
        ):
            return False

        reference_index = self._smiles_list.index(fragment)
        smiles_index = self._smiles_list.index(smiles)
        return (
            self._similarity_matrix[reference_index, smiles_index]
            > self.threshold
        )

    def additional_metrics(self) -> dict[str, float]:
        """Calculate additional similarity metrics."""
        # Remove diagonal for statistics
        similarity_no_diag = self._similarity_matrix.copy()
        np.fill_diagonal(similarity_no_diag, 0)

        return {
            "mean_similarity": float(np.mean(similarity_no_diag)),
            "std_similarity": float(np.std(similarity_no_diag)),
        }

    def _get_similar_molecules_vectorized(
        self, reference_pattern: str, smiles_list: list[str]
    ) -> list[str]:
        """Get similar molecules using vectorized operations."""
        if reference_pattern not in self._smiles_list:
            return []

        reference_index = self._smiles_list.index(reference_pattern)
        similar_indices = np.where(
            self._similarity_matrix[reference_index, :] > self.threshold
        )[0]

        return [
            self._smiles_list[idx]
            for idx in similar_indices
            if self._smiles_list[idx] in smiles_list
        ]

    def _process_molecules_data(
        self, molecules_with_fragment: list[str], df_grouped: pd.DataFrame
    ) -> tuple:
        """Process molecules data for a given fragment."""
        if not molecules_with_fragment:
            return np.nan, set(), 0, [], []

        subset_df = df_grouped.loc[
            df_grouped.index.intersection(molecules_with_fragment)
        ]
        mean_score = subset_df["Mean score cluster"].mean()

        # Collect data from subset
        unique_molecules = set()
        step_values = []
        num_mol_values = []

        for _, row in subset_df.iterrows():
            # Process molecules containing fragment
            mol_frag = row["Molecules containing fragment"]
            if isinstance(mol_frag, str):
                try:
                    unique_molecules.update(eval(mol_frag))
                except (ValueError, SyntaxError):
                    pass  # Skip invalid strings
            elif isinstance(mol_frag, set):
                unique_molecules.update(mol_frag)

            # Process step list
            step_list = row["step_list"]
            if isinstance(step_list, str):
                try:
                    step_values.extend(eval(step_list))
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(step_list, list):
                step_values.extend(step_list)

            # Process number of molecules list
            num_mol_list = row["Number of Molecules with Substructure List"]
            if isinstance(num_mol_list, str):
                try:
                    num_mol_values.extend(eval(num_mol_list))
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(num_mol_list, list):
                num_mol_values.extend(num_mol_list)

        return (
            mean_score,
            unique_molecules,
            len(unique_molecules),
            step_values,
            num_mol_values,
        )

    def aggregate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the DataFrame by clusters.

        This is a function to be used on a grouped DataFrame of clusters
        with steps.
        """
        smiles_list = df["Substructure"].tolist()
        self.get_count(smiles_list)
        self._fragments_df = self._fragments_df[
            self._fragments_df["Count"] > self._min_count_fragments
        ]

        # Apply vectorized function for similarity
        self._fragments_df["molecules_with_fragment"] = [
            self._get_similar_molecules_vectorized(substructure, smiles_list)
            for substructure in self._fragments_df["Substructure"]
        ]

        if self.add_num_atoms:
            # Vectorized num_atoms calculation
            num_atoms = []
            for substructure in self._fragments_df["Substructure"]:
                mol = Chem.MolFromSmarts(substructure)
                if mol is not None:
                    num_atoms.append(mol.GetNumAtoms())
                else:
                    num_atoms.append(0)
            self._fragments_df["num_atoms"] = num_atoms

        # Pre-compute DataFrame subsets for faster filtering
        df_grouped = df.set_index("Substructure")

        # Process all fragments efficiently
        results = [
            self._process_molecules_data(molecules_with_fragment, df_grouped)
            for molecules_with_fragment in self._fragments_df[
                "molecules_with_fragment"
            ]
        ]

        # Unpack results and assign to DataFrame
        (
            mean_scores,
            molecules_containing,
            num_molecules,
            steps_list,
            num_molecules_list,
        ) = zip(*results, strict=True)

        self._fragments_df["Mean score cluster"] = mean_scores
        self._fragments_df["Molecules containing fragment"] = (
            molecules_containing
        )
        self._fragments_df["Number of Molecules_with_Fragment"] = num_molecules
        self._fragments_df["Steps"] = steps_list
        self._fragments_df["Number of Molecules_with_Fragment List"] = (
            num_molecules_list
        )

        # Select final columns and add computed metrics
        self._fragments_df = self._fragments_df[
            [
                "Substructure",
                "Molecules containing fragment",
                "Number of Molecules_with_Fragment",
                "Mean score cluster",
                "Steps",
                "Number of Molecules_with_Fragment List",
            ]
        ]

        # Add step metrics using vectorized operations
        self._fragments_df["step min"] = [
            (
                min(x)
                if isinstance(x, list) and x
                else (x if x is not None else 0)
            )
            for x in self._fragments_df["Steps"]
        ]
        self._fragments_df["step count"] = [
            len(x) if isinstance(x, list) else 1
            for x in self._fragments_df["Steps"]
        ]

        return self._fragments_df


def flatten_dataframe_to_unique_column(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten all non-NaN sets from a DataFrame into a single set,
    and return a DataFrame with unique fragments and their count.
    """
    mat = df.to_numpy()
    unique_items_list = []
    for i in range(mat.shape[0]):
        unique_items = set()
        for j in range(mat.shape[1]):
            if isinstance(mat[i, j], str):
                unique_items.update(eval(mat[i, j]))
            elif isinstance(mat[i, j], set):
                unique_items.update(mat[i, j])

        unique_items_list.append(unique_items)

    return pd.DataFrame(
        {
            "unique_fragments": [x for x in unique_items_list],
            "number of molecules": [len(x) for x in unique_items_list],
        }
    )


def flatten_dataframe_to_column(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten all non-NaN sets from a DataFrame into a single set,
    and return a DataFrame with unique fragments and their count.
    """
    mat = df.to_numpy()
    unique_items_list = []
    for i in range(mat.shape[0]):
        unique_items = []
        for j in range(mat.shape[1]):
            if isinstance(mat[i, j], str):
                unique_items.extend(eval(mat[i, j]))
            elif isinstance(mat[i, j], list):
                unique_items.extend(mat[i, j])

        unique_items_list.append(unique_items)

    return pd.DataFrame(
        {
            "unique_fragments": [list(x) for x in unique_items_list],
        }
    )


def aggregate_list_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Aggregate a list column in a DataFrame."""
    df[column_name] = df[column_name].apply(eval)
    df[column_name] = df[column_name].apply(lambda x: list(set(x)))
    return df.groupby(column_name).size().reset_index(name="Count")


# Example usage demonstrating the new configurable approach
def create_scorer_examples() -> tuple[
    ClusterSimScorer, ClusterSimScorer, ClusterSimScorer, ClusterSimScorer
]:
    """Example configurations for different use cases."""
    # Example 1: Default threshold-based clustering with Tanimoto similarity
    default_scorer = ClusterSimScorer(threshold=0.8)

    # Example 2: Hierarchical clustering with Dice similarity
    hierarchical_scorer = ClusterSimScorer(
        similarity_metric="dice",
        clustering_method="hierarchical",
        n_clusters=10,
        linkage="average",
    )

    # Example 3: DBSCAN clustering with RDKit fingerprints
    dbscan_scorer = ClusterSimScorer(
        fingerprint_type="rdkit",
        clustering_method="dbscan",
        eps=0.3,
        min_samples=3,
    )

    # Example 4: Custom Morgan fingerprint parameters
    custom_scorer = ClusterSimScorer(
        fingerprint_type="morgan",
        fingerprint_radius=3,
        fingerprint_size=1024,
        threshold=0.7,
    )

    return default_scorer, hierarchical_scorer, dbscan_scorer, custom_scorer
