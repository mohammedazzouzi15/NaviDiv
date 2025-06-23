import time  

import pandas as pd
from pathlib import Path
from navidiv import (
    cluster_similarity_scorer,
)
from navidiv.utils import get_smiles_column


def test_cluster_scorer(df, output_path):
    """Test the cluster similarity scorer."""
    cluster_score = cluster_similarity_scorer.ClusterSimScorer(
        output_path=output_path,
        threshold=0.25,
    )
    start = time.time()  
    scores_cluster = cluster_score.aggregate_df(df)
    cluster_score._fragments_df.to_csv(
        f"{cluster_score._output_path}/grouped_by_aggregate_clusters_with_score.csv",
        index=False,
    )  
    end = time.time()  
    print(
        f"ClusterSimilarityScorer.get_score time: {end - start:.3f} seconds"
    )  
    print("Cluster scores")
    print("scores", scores_cluster)
    return scores_cluster


if __name__ == "__main__":
    csv_file = "/media/mohammed/Work/Navi_diversity/reinvent_runs/runs/test/test4/stage0/scorer_output/clusters/groupby_results_clusters.csv"
    output_path = Path(csv_file).parent
    df = pd.read_csv(csv_file).sample(frac=1).reset_index(drop=True)
    # df = df[df["step"] == 1000]
    print("columns", df.columns)
    smiles_col = get_smiles_column(df)
    df = df.dropna(subset=[smiles_col])

    test_cluster_scorer(df, output_path)

    print("done")
