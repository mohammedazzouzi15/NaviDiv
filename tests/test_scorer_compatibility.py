"""Tests for general scorer pattern compatibility."""

import pytest
from rdkit import Chem

from navidiv.fragment.fragment_scorer import FragmentScorer
from navidiv.stringbased.Ngram_scorer import NgramScorer
from navidiv.simlarity.cluster_similarity_scorer import ClusterSimScorer


class TestScorerPatternCompatibility:
    """Test that all scorers implement the common interface correctly."""
    
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.test_smiles = [
            "CCO",  # ethanol
            "CCC",  # propane  
            "CCCO",  # propanol
            "c1ccccc1",  # benzene
            "c1ccccc1C",  # toluene
        ]
        
        # Initialize different types of scorers
        self.fragment_scorer = FragmentScorer(
            min_count_fragments=1,
            output_path=None,
            transformation_mode="none"
        )
        
        self.ngram_scorer = NgramScorer(
            ngram_size=3,
            output_path=None
        )
        
        self.similarity_scorer = ClusterSimScorer(
            threshold=0.7,
            output_path=None
        )
        
    def test_all_scorers_have_count_pattern_occurrences(self) -> None:
        """Test that all scorers implement _count_pattern_occurrences."""
        scorers = [
            self.fragment_scorer,
            self.ngram_scorer, 
            self.similarity_scorer
        ]
        
        for scorer in scorers:
            assert hasattr(scorer, '_count_pattern_occurrences')
            
            # Initialize the scorer 
            scorer.get_count(self.test_smiles)
            
            # Test that the method can be called
            if hasattr(scorer, '_fragments_df') and scorer._fragments_df is not None:
                if len(scorer._fragments_df) > 0:
                    first_pattern = scorer._fragments_df['Substructure'].iloc[0]
                    count = scorer._count_pattern_occurrences(
                        self.test_smiles, first_pattern
                    )
                    assert isinstance(count, int)
                    assert count >= 0

    def test_all_scorers_have_comparison_function(self) -> None:
        """Test that all scorers implement _comparison_function."""
        scorers = [
            self.fragment_scorer,
            self.ngram_scorer,
            self.similarity_scorer
        ]
        
        for scorer in scorers:
            assert hasattr(scorer, '_comparison_function')
            
            # Initialize the scorer
            scorer.get_count(self.test_smiles)
            
            # Test with a simple case
            result = scorer._comparison_function(
                smiles="CCO",
                fragment="CCO"  # Same molecule
            )
            assert isinstance(result, bool)

    def test_all_scorers_have_get_count(self) -> None:
        """Test that all scorers implement get_count."""
        scorers = [
            self.fragment_scorer,
            self.ngram_scorer,
            self.similarity_scorer
        ]
        
        for scorer in scorers:
            assert hasattr(scorer, 'get_count')
            
            fragments_df, over_represented = scorer.get_count(self.test_smiles)
            
            assert fragments_df is not None
            assert 'Substructure' in fragments_df.columns
            assert 'Count' in fragments_df.columns

    def test_all_scorers_have_additional_metrics(self) -> None:
        """Test that all scorers implement additional_metrics.""" 
        scorers = [
            self.fragment_scorer,
            self.ngram_scorer,
            self.similarity_scorer
        ]
        
        for scorer in scorers:
            assert hasattr(scorer, 'additional_metrics')
            
            # Initialize the scorer
            scorer.get_count(self.test_smiles)
            
            metrics = scorer.additional_metrics()
            assert isinstance(metrics, dict)

    def test_fragment_scorer_specific_behavior(self) -> None:
        """Test fragment scorer specific behavior."""
        self.fragment_scorer.get_count(self.test_smiles)
        
        # Test with a simple fragment
        count = self.fragment_scorer._count_pattern_occurrences(
            self.test_smiles, "C"  # Carbon atom
        )
        assert count > 0  # Should find carbon in all molecules
        
        # Test comparison function
        result = self.fragment_scorer._comparison_function(
            smiles="CCO",
            fragment="C"
        )
        assert result is True  # CCO contains carbon

    def test_ngram_scorer_specific_behavior(self) -> None:
        """Test n-gram scorer specific behavior."""
        self.ngram_scorer.get_count(self.test_smiles)
        
        # Test with a common substring
        count = self.ngram_scorer._count_pattern_occurrences(
            self.test_smiles, "CC"  # Double carbon
        )
        assert count >= 0
        
        # Test comparison function  
        result = self.ngram_scorer._comparison_function(
            smiles="CCO",
            fragment="CC"
        )
        assert result is True  # CCO contains "CC"

    def test_similarity_scorer_specific_behavior(self) -> None:
        """Test similarity scorer specific behavior."""
        self.similarity_scorer.get_count(self.test_smiles)
        
        # Test with one of the molecules
        count = self.similarity_scorer._count_pattern_occurrences(
            self.test_smiles, "CCO"
        )
        assert count >= 1  # Should find at least itself
        
        # Test comparison function
        result = self.similarity_scorer._comparison_function(
            smiles="CCO",
            fragment="CCO"  # Same molecule
        )
        assert isinstance(result, bool)


class TestScorerConfigurationCompatibility:
    """Test that scorer configurations work correctly."""
    
    def test_fragment_scorer_transformations(self) -> None:
        """Test different fragment transformation modes."""
        transformation_modes = [
            "none",
            "basic_framework", 
            "elemental_wire_frame",
            "basic_wire_frame"
        ]
        
        test_smiles = ["CCO", "c1ccccc1"]
        
        for mode in transformation_modes:
            scorer = FragmentScorer(
                min_count_fragments=1,
                output_path=None,
                transformation_mode=mode
            )
            
            fragments_df, _ = scorer.get_count(test_smiles)
            assert fragments_df is not None
            assert len(fragments_df) >= 0

    def test_similarity_scorer_thresholds(self) -> None:
        """Test different similarity thresholds."""
        thresholds = [0.5, 0.7, 0.9]
        test_smiles = ["CCO", "CCCO", "c1ccccc1"]
        
        for threshold in thresholds:
            scorer = ClusterSimScorer(
                threshold=threshold,
                output_path=None
            )
            
            fragments_df, _ = scorer.get_count(test_smiles)
            assert fragments_df is not None
            assert scorer.threshold == threshold

    def test_ngram_scorer_sizes(self) -> None:
        """Test different n-gram sizes."""
        ngram_sizes = [2, 3, 5]
        test_smiles = ["CCO", "CCCO", "CCCCO"]
        
        for size in ngram_sizes:
            scorer = NgramScorer(
                ngram_size=size,
                output_path=None
            )
            
            fragments_df, _ = scorer.get_count(test_smiles)
            assert fragments_df is not None
            assert scorer.ngram_size == size


if __name__ == "__main__":
    pytest.main([__file__])
