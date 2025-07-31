"""Tests for similarity-based scorers with improved naming."""

import numpy as np
import pytest
from rdkit import Chem

from navidiv.simlarity.cluster_similarity_scorer import ClusterSimScorer
from navidiv.simlarity.orginal_similarity_scorer import OriginalSimScorer


class TestClusterSimilarityScorer:
    """Test the cluster similarity scorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_smiles = [
            "CCO",  # ethanol
            "CCC",  # propane  
            "CCCO",  # propanol (similar to ethanol)
            "c1ccccc1",  # benzene
            "c1ccccc1C",  # toluene (similar to benzene)
            "CCCCCCCC",  # octane (different from all)
        ]
        self.scorer = ClusterSimScorer(threshold=0.7, output_path=None)

    def test_initialization(self):
        """Test scorer initialization."""
        assert self.scorer.threshold == 0.7
        assert self.scorer._csv_name == "clusters"
        assert self.scorer._min_count_fragments == 0

    def test_get_count(self):
        """Test the get_count method."""
        fragments_df, over_represented = self.scorer.get_count(self.test_smiles)
        
        assert fragments_df is not None
        assert len(fragments_df) > 0
        assert "Substructure" in fragments_df.columns
        assert "Count" in fragments_df.columns
        assert "Count ratio" in fragments_df.columns

    def test_count_pattern_occurrences(self):
        """Test the _count_pattern_occurrences method."""
        # First run get_count to initialize similarity matrix
        self.scorer.get_count(self.test_smiles)
        
        # Test counting similar molecules to ethanol
        count = self.scorer._count_pattern_occurrences(
            self.test_smiles, "CCO"
        )
        assert isinstance(count, int)
        assert count >= 1  # At least itself

    def test_comparison_function(self):
        """Test the _comparison_function method."""
        # Initialize scorer
        self.scorer.get_count(self.test_smiles)
        
        # Test similarity comparison
        is_similar = self.scorer._comparison_function(
            smiles="CCO", fragment="CCCO"
        )
        assert isinstance(is_similar, bool)

    def test_additional_metrics(self):
        """Test additional metrics calculation."""
        self.scorer.get_count(self.test_smiles)
        metrics = self.scorer.additional_metrics()
        
        assert "mean_similarity" in metrics
        assert "std_similarity" in metrics
        assert isinstance(metrics["mean_similarity"], (float, np.floating))
        assert isinstance(metrics["std_similarity"], (float, np.floating))

    def test_get_clusters(self):
        """Test cluster detection."""
        # Create a small similarity matrix for testing
        similarity_matrix = np.array([
            [1.0, 0.8, 0.9, 0.2],
            [0.8, 1.0, 0.85, 0.1], 
            [0.9, 0.85, 1.0, 0.15],
            [0.2, 0.1, 0.15, 1.0]
        ])
        
        clusters = self.scorer.get_clusters(similarity_matrix)
        assert isinstance(clusters, set)
        assert len(clusters) >= 1


class TestOriginalSimilarityScorer:
    """Test the original similarity scorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_smiles = [
            "CCO", 
            "c1ccccc1",
            "CCC"
        ]
        self.test_smiles = [
            "CCCO",  # similar to CCO
            "c1ccccc1C",  # similar to benzene
            "CCCC",  # similar to CCC
            "NCCCCN",  # different from all
        ]
        self.scorer = OriginalSimScorer(
            threshold=0.7, 
            output_path=None,
            smiles_list_to_compare_to=self.original_smiles
        )

    def test_initialization(self):
        """Test scorer initialization."""
        assert self.scorer.threshold == 0.7
        assert self.scorer._csv_name == "Original_similarity"
        assert self.scorer.smiles_list_to_compare_to == self.original_smiles

    def test_get_count(self):
        """Test the get_count method."""
        fragments_df, over_represented = self.scorer.get_count(self.test_smiles)
        
        assert fragments_df is not None
        assert len(fragments_df) > 0
        assert "Substructure" in fragments_df.columns
        assert "max_similarity" in fragments_df.columns

    def test_count_pattern_occurrences(self):
        """Test the _count_pattern_occurrences method."""
        # First run get_count to initialize similarity matrix
        self.scorer.get_count(self.test_smiles)
        
        # Test counting similar molecules
        count = self.scorer._count_pattern_occurrences(
            self.test_smiles, "CCCO"
        )
        assert isinstance(count, int)
        assert count >= 1

    def test_get_max_similarity(self):
        """Test maximum similarity calculation."""
        self.scorer.get_count(self.test_smiles)
        
        max_sim = self.scorer._get_max_similarity("CCCO")
        assert isinstance(max_sim, (float, np.floating))
        assert 0.0 <= max_sim <= 1.0

    def test_additional_metrics(self):
        """Test additional metrics calculation."""
        self.scorer.get_count(self.test_smiles)
        metrics = self.scorer.additional_metrics()
        
        assert "mean_distance" in metrics
        assert "std_distance" in metrics
        assert isinstance(metrics["mean_distance"], (float, np.floating))
        assert isinstance(metrics["std_distance"], (float, np.floating))

    def test_comparison_function_disabled(self):
        """Test that comparison function returns False as expected."""
        # Initialize scorer
        self.scorer.get_count(self.test_smiles)
        
        # This should return False as it's disabled in current implementation
        result = self.scorer._comparison_function(
            smiles="CCCO", fragment="CCO"
        )
        assert result is False


class TestSimilarityUtilityFunctions:
    """Test utility functions for similarity scoring."""

    def test_get_fingerprints(self):
        """Test fingerprint generation."""
        from navidiv.simlarity.cluster_similarity_scorer import get_fingerprints
        
        smiles_list = ["CCO", "CCC", "c1ccccc1"]
        molecules = [Chem.MolFromSmiles(s) for s in smiles_list]
        molecules = [m for m in molecules if m is not None]
        
        fingerprints = get_fingerprints(molecules)
        assert len(fingerprints) == len(molecules)
        assert all(fp is not None for fp in fingerprints)

    def test_calculate_similarity(self):
        """Test similarity calculation."""
        from navidiv.simlarity.cluster_similarity_scorer import (
            get_fingerprints, calculate_similarity
        )
        
        smiles1 = ["CCO", "CCC"]
        smiles2 = ["CCCO", "CCCC"]
        
        molecules1 = [Chem.MolFromSmiles(s) for s in smiles1]
        molecules2 = [Chem.MolFromSmiles(s) for s in smiles2]
        
        fps1 = get_fingerprints(molecules1)
        fps2 = get_fingerprints(molecules2)
        
        similarities = calculate_similarity(fps1, fps2)
        
        assert similarities.shape == (len(fps1), len(fps2))
        assert np.all(similarities >= 0.0)
        assert np.all(similarities <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
