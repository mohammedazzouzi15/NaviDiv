#!/usr/bin/env python3
"""Minimal example to test the clustering capabilities of ClusterSimScorer.

This script demonstrates how to use the ClusterSimScorer with SMILES data
from the test data file and shows different clustering methods.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from navidiv.simlarity.cluster_similarity_scorer import ClusterSimScorer


def load_test_smiles(
    csv_file_path: str, max_molecules: int = 50, step: int = 1
) -> list[str]:
    """Load SMILES from the test data CSV file.

    Args:
        csv_file_path: Path to the CSV file containing SMILES data
        max_molecules: Maximum number of molecules to load for testing

    Returns:
        List of SMILES strings
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")

        if "SMILES" not in df.columns:
            raise ValueError("No 'SMILES' column found in the CSV file")

        # Filter out invalid SMILES and take a subset for testing
        if "step" in df.columns:
            smiles_list = df[df["step"] == step]["SMILES"].dropna().tolist()
        else:
            smiles_list = df[df["SMILES"]].dropna().tolist()
        smiles_list = [s for s in smiles_list if s and s != "None"]

        # Take only the first max_molecules for faster testing
        if len(smiles_list) > max_molecules:
            smiles_list = smiles_list[:max_molecules]
            print(f"Using first {max_molecules} molecules for testing")

        print(f"Found {len(smiles_list)} valid SMILES")
        return smiles_list

    except Exception as e:
        print(f"Error loading test data: {e}")
        return []


def test_basic_clustering():
    """Test basic clustering functionality with default settings."""
    print("\n" + "=" * 60)
    print("BASIC CLUSTERING TEST (Threshold-based)")
    print("=" * 60)

    # Initialize the scorer with default settings
    scorer = ClusterSimScorer(
        threshold=0.8,  # High similarity threshold
        similarity_metric="tanimoto",
        clustering_method="threshold",
        fingerprint_type="morgan",
    )

    # Simple test molecules
    test_smiles = [
        "CCO",  # ethanol
        "CCCO",  # propanol (similar to ethanol)
        "CCCCO",  # butanol (similar to alcohols)
        "c1ccccc1",  # benzene
        "c1ccccc1C",  # toluene (similar to benzene)
        "c1ccccc1CC",  # ethylbenzene (similar to toluene)
        "CCCCCCCCCC",  # decane (different from all)
        "C(=O)O",  # formic acid
        "CC(=O)O",  # acetic acid (similar to formic acid)
    ]

    print(f"Testing with {len(test_smiles)} molecules")
    for i, smiles in enumerate(test_smiles):
        print(f"  {i + 1}: {smiles}")

    # Get clusters
    fragments_df, _ = scorer.get_count(test_smiles)

    print(f"\nFound {len(fragments_df)} cluster representatives:")
    print(fragments_df[["Substructure", "Count", "Count ratio"]])

    # Show similarity matrix (if small enough)
    if hasattr(scorer, "_similarity_matrix"):
        print(f"\nSimilarity matrix shape: {scorer._similarity_matrix.shape}")
        if scorer._similarity_matrix.shape[0] <= 10:
            print("Similarity matrix:")
            print(np.round(scorer._similarity_matrix, 3))

    return scorer, fragments_df


def test_hierarchical_clustering():
    """Test hierarchical clustering method."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLUSTERING TEST")
    print("=" * 60)

    # Initialize with hierarchical clustering
    scorer = ClusterSimScorer(
        threshold=0.7,
        similarity_metric="tanimoto",
        clustering_method="hierarchical",
        n_clusters=3,  # Force 3 clusters
        linkage="average",
    )

    # Test molecules with clear structural groups
    test_smiles = [
        "CCO",
        "CCCO",
        "CCCCO",  # alcohol group
        "c1ccccc1",
        "c1ccccc1C",
        "c1ccccc1CC",  # aromatic group
        "CCCCCCCC",
        "CCCCCCCCC",
        "CCCCCCCCCC",  # alkane group
    ]

    print(f"Testing hierarchical clustering with {len(test_smiles)} molecules")
    fragments_df, _ = scorer.get_count(test_smiles)

    print(f"\nFound {len(fragments_df)} cluster representatives:")
    print(fragments_df[["Substructure", "Count", "Count ratio"]])

    return scorer, fragments_df


def test_dbscan_clustering():
    """Test DBSCAN clustering method."""
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTERING TEST")
    print("=" * 60)

    # Initialize with DBSCAN clustering
    scorer = ClusterSimScorer(
        threshold=0.6,
        similarity_metric="tanimoto",
        clustering_method="dbscan",
        eps=0.4,  # Distance threshold
        min_samples=2,  # Minimum samples per cluster
    )

    # Test molecules
    test_smiles = [
        "CCO",
        "CCCO",
        "CCCCO",
        "CCCCCO",  # similar alcohols
        "c1ccccc1",
        "c1ccccc1C",  # similar aromatics
        "CCCCCCCCCC",  # different alkane
        "C(=O)O",
        "CC(=O)O",  # similar acids
    ]

    print(f"Testing DBSCAN clustering with {len(test_smiles)} molecules")
    fragments_df, _ = scorer.get_count(test_smiles)

    print(f"\nFound {len(fragments_df)} cluster representatives:")
    print(fragments_df[["Substructure", "Count", "Count ratio"]])

    return scorer, fragments_df


def test_with_real_data(clustering_method="threshold", **cluster_args):
    """Test clustering with real data from the CSV file."""
    print("\n" + "=" * 60)
    print("REAL DATA CLUSTERING TEST")
    print("=" * 60)

    # Load SMILES from the test data file
    csv_path = "/media/mohammed/Work/Navi_diversity/tests/test_data/default/default_1_TSNE.csv"
    smiles_list = load_test_smiles(csv_path, max_molecules=800, step=100)

    if not smiles_list:
        print("Could not load test data. Skipping real data test.")
        return None, None

    # Test with different thresholds
    # if any in cluster args is a list

    print(f"\n--- Testing with {cluster_args} ---")

    scorer = ClusterSimScorer(
        similarity_metric="tanimoto",
        clustering_method=clustering_method,
        **cluster_args,
    )

    try:
        fragments_df, _ = scorer.get_count(smiles_list)
        print(
            f"Found {len(fragments_df)} clusters with {clustering_method} method"
        )

        if len(fragments_df) <= 10:  # Only show details if not too many
            print("Top clusters:")
            top_clusters = fragments_df.head().copy()
            # Truncate long SMILES for display
            top_clusters["Substructure_short"] = top_clusters[
                "Substructure"
            ].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
            print(top_clusters[["Substructure_short", "Count", "Count ratio"]])

    except Exception as e:
        print(f"Error with {cluster_args}: {e}")

    return scorer, fragments_df


def show_clustering_stats(scorer, fragments_df):
    """Show additional clustering statistics."""
    print("\n" + "-" * 40)
    print("CLUSTERING STATISTICS")
    print("-" * 40)

    if (
        hasattr(scorer, "_similarity_matrix")
        and scorer._similarity_matrix is not None
    ):
        sim_matrix = scorer._similarity_matrix

        # Get similarity statistics
        mask = np.triu(
            np.ones_like(sim_matrix, dtype=bool), k=1
        )  # Upper triangle
        similarities = sim_matrix[mask]

        print("Similarity statistics:")
        print(f"  Mean similarity: {similarities.mean():.3f}")
        print(f"  Std similarity:  {similarities.std():.3f}")
        print(f"  Min similarity:  {similarities.min():.3f}")
        print(f"  Max similarity:  {similarities.max():.3f}")
        print(f"  Median similarity: {np.median(similarities):.3f}")

        # Count pairs above threshold
        above_threshold = (similarities > scorer.threshold).sum()
        total_pairs = len(similarities)
        print(
            f"  Pairs above threshold ({scorer.threshold}): {above_threshold}/{total_pairs} ({100 * above_threshold / total_pairs:.1f}%)"
        )

    if fragments_df is not None:
        print("\nCluster statistics:")
        print(f"  Number of clusters: {len(fragments_df)}")
        if "Count" in fragments_df.columns:
            print(
                f"  Average cluster size: {fragments_df['Count'].mean():.2f}"
            )
            print(f"  Largest cluster size: {fragments_df['Count'].max()}")
            print(f"  Smallest cluster size: {fragments_df['Count'].min()}")


def main():
    """Run all clustering tests."""
    print("ClusterSimScorer Minimal Testing Example")
    print("=========================================")
    print(
        "This script tests the clustering capabilities of the ClusterSimScorer class."
    )

    try:
        # Test 1: Basic threshold clustering
        # scorer1, df1 = test_basic_clustering()
        # show_clustering_stats(scorer1, df1)

        # Test 2: Hierarchical clustering
        # scorer2, df2 = test_hierarchical_clustering()
        # show_clustering_stats(scorer2, df2)

        # # Test 3: DBSCAN clustering
        # scorer3, df3 = test_dbscan_clustering()
        # show_clustering_stats(scorer3, df3)

        # Test 4: Real data
        # scorer4, df4 = test_with_real_data(
        #     clustering_method="threshold",
        #     threshold=0.25,  # Example threshold
        #     fingerprint_type="morgan",
        #     fingerprint_radius=2,
        # )
        # if scorer4 and df4 is not None:
        #     show_clustering_stats(scorer4, df4)

        # test 5 : real data dbscan
        scorer5, df5 = test_with_real_data(
            clustering_method="dbscan",
            eps=0.6,  # Example epsilon
            min_samples=2,
            fingerprint_type="morgan",
            fingerprint_radius=2,
            min_cluster_size=2
        )
        if scorer5 and df5 is not None:
            show_clustering_stats(scorer5, df5)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
