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


class ClusterSimScorer(BaseScore):
    """Handles fragment scoring and analysis for molecular datasets."""

    def __init__(
        self,
        threshold: float = 0.8,
        output_path: str | None = None,
    ) -> None:
        """Initialize FragmentScore.

        Args:
            min_count_fragments (int): Minimum count for fragments to be
                considered.
            output_path (str | None): Path to save output files.
        """
        super().__init__(output_path=output_path)
        self._csv_name = "clusters"
        self._similarity = []
        self.threshold = threshold
        self._min_count_fragments = 0

    def get_clusters(self, similarity):
        """Get clusters of similar molecules based on a similarity threshold."""
        # np.fill_diagonal(similarity, 0)
        clusters = set()

        for i in range(similarity.shape[0]):
            if not np.any(similarity[i, :i] > self.threshold):
                clusters.add(i)
        return clusters

    def get_count(self, smiles_list: list[str]) -> tuple[pd.DataFrame, None]:
        """Calculate the percentage of each fragment in the dataset.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            tuple: DataFrame with fragment info, None (for compatibility)
        """
        # if not hasattr(self, "_mol_smiles"):
        self._mol_smiles = [
            Chem.MolFromSmiles(smiles)
            for smiles in smiles_list
            if smiles != "None"
        ]
        self._mol_smiles = [mol for mol in self._mol_smiles if mol is not None]
        # sort the mol_smiles by their number of atoms
        self._mol_smiles.sort(key=lambda x: x.GetNumAtoms())
        self._smiles_list = [
            Chem.MolToSmiles(mol)
            for mol in self._mol_smiles
            if mol is not None
        ]

        search_fps = get_fingerprints(self._mol_smiles)

        self._similarity_to_itself = calculate_similarity(
            search_fps, search_fps
        )
        clusters = self.get_clusters(self._similarity_to_itself)
        clusters_smiles = [self._smiles_list[i] for i in clusters]
        fragments, over_represented_fragments = self._from_list_to_count_df(
            self._smiles_list,
            clusters_smiles,
            total_number_of_ngrams=len(self._smiles_list),
        )
        self._fragments_df = fragments
        return fragments, over_represented_fragments

    def _count_pattern_occurrences(
        self, smiles_list: list[str], reference_smiles: str
    ) -> int:
        """Count molecules similar to a reference molecule.

        Args:
            smiles_list: Not used in similarity scoring but kept for interface
                compatibility.
            reference_smiles: Reference molecule to compare against.

        Returns:
            Number of molecules similar to the reference.
        """
        reference_index = self._smiles_list.index(reference_smiles)
        return 1 + len(
            [
                i
                for i in range(self._similarity_to_itself.shape[0])
                if (
                    self._similarity_to_itself[reference_index, i]
                    > self.threshold
                )
            ]
        )

    def _comparison_function(
        self,
        smiles: str | None = None,
        fragment: str | None = None,
    ) -> bool:
        """Check if the molecule is similar to the reference pattern.

        In cluster similarity scoring, this checks if two molecules are
        similar based on the similarity threshold.
        """
        if fragment in self._smiles_list:
            reference_index = self._smiles_list.index(fragment)
        else:
            return False
        if smiles in self._smiles_list:
            smiles_index = self._smiles_list.index(smiles)
        else:
            return False
        return (
            self._similarity_to_itself[reference_index, smiles_index]
            > self.threshold
        )

    def additional_metrics(self) -> dict[str, float]:
        """Calculate additional metrics for the scorer."""
        np.fill_diagonal(self._similarity_to_itself, 0)
        mean_similarity = np.mean(self._similarity_to_itself)
        std_similarity = np.std(self._similarity_to_itself)
        return {
            "mean_similarity": mean_similarity,
            "std_similarity": std_similarity,
        }

    def _get_similar_molecules_vectorized(
        self, reference_pattern: str, smiles_list: list[str]
    ) -> list[str]:
        """Get similar molecules using vectorized operations."""
        smiles_to_index = {
            smiles: idx for idx, smiles in enumerate(self._smiles_list)
        }
        
        if reference_pattern not in smiles_to_index:
            return []
        
        reference_index = smiles_to_index[reference_pattern]
        similar_indices = np.where(
            self._similarity_to_itself[reference_index, :] > self.threshold
        )[0]
        
        return [
            self._smiles_list[idx] for idx in similar_indices
            if self._smiles_list[idx] in smiles_list
        ]

    def _process_molecules_data(self, molecules_with_fragment: list[str], df_grouped: pd.DataFrame) -> tuple:
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
        
        return mean_score, unique_molecules, len(unique_molecules), step_values, num_mol_values

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
                min(x) if isinstance(x, list) and x
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
