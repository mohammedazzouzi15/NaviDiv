import logging
import os

import pandas as pd


def get_smiles_column(df):
    for col in ["smiles", "SMILES", "Smiles", "Substructure"]:
        if col in df.columns:
            return col
    raise ValueError("No column containing SMILES strings found.")


def add_mean_of_numeric_columns(df, steps) -> dict:
    """Add mean of numeric columns to the dataframe."""
    dict_output = {}
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_columns:
        dict_output[col] = [
            df[df["step"] == step][col].mean() for step in steps
        ]
    return dict_output


def groupby_results(df):
    df = df[df["Molecules_with_Fragment"] > 0]
    # delete rows where molecules_countaining_fragment is not a list
    df = df[
        df["molecules_countaining_fragment"].apply(
            lambda x: isinstance(eval(x), list)
        )
    ]
    grouped_by_df = df.groupby(["Substructure"]).agg(
        {
            "Count": ["sum", "max"],
            "step": ["max", "min", "count", lambda x: list(x)],
            "median_score_fragment": ["max", "min", "mean", lambda x: list(x)],
            "diff_median_score": ["max", "min", "mean", lambda x: list(x)],
            "Ratio_of Molecules_with_Fragment": [
                "max",
                "first",
                "mean",
                lambda x: list(x),
            ],
            "Molecules_with_Fragment": [
                "max",
                "first",
                "mean",
                "sum",
                lambda x: list(x),
            ],
            "molecules_countaining_fragment": [
                lambda x: set([xsy for xs in x for xsy in eval(xs)]),
                lambda x: len(set([xsy for xs in x for xsy in eval(xs)])),
            ],
        }
    )

    # transform the groubed_by_df to a dataframe
    grouped_by_df = grouped_by_df.reset_index()
    grouped_by_df.columns = [
        "Substructure",
        "Count",
        "Count max",
        "step max",
        "step min",
        "step count",
        "step_list",
        "median_score_fragment_max",
        "median_score_fragment_min",
        "Mean score cluster",
        "median_score_fragment_list",
        "diff_median_score_max",
        "diff_median_score_min",
        "Mean diff score",
        "diff_median_score_list",
        "Count_perc_per_molecule_max",
        "Count_perc_per_molecule_first",
        "Count_perc_per_molecule_mean",
        "Count_perc_per_molecule_list",
        "count_per_molecule_max",
        "count_per_molecule_first",
        "count_per_molecule_mean",
        "Total Number of Molecules with Substructure",
        "Number of Molecules with Substructure List",
        "Molecules containing fragment",
        "Number of Molecules_with_Fragment",
    ]

    grouped_by_df["count_perc_ratio"] = grouped_by_df.apply(
        lambda x: x["Count_perc_per_molecule_max"]
        / x["Count_perc_per_molecule_first"]
        if x["Count_perc_per_molecule_first"] != 0
        else 0,
        axis=1,
    )

    return grouped_by_df


def initialize_scorer(scorer_props: dict):
    """Initialize the scorer based on the provided properties.

    Args:
        scorer_props (dict): Properties of the scorer to be initialized.
            - "scorer_name": Name of the scorer (e.g., "Fragments", "Ngram").
            - "output_path": Path to save output files.
            - "min_count_fragments": Minimum count for fragments to be considered.
            - "ngram_size": Size of n-grams (for Ngram scorer).
            - "scaffold_type": Type of scaffold (for Scaffold scorer).
            - "threshold": Threshold for similarity (for Cluster and Original scorers).
    """
    scorer_name = scorer_props.get("scorer_name")
    if scorer_name == "Fragments Match":
        from navidiv.fragment import fragment_scorer_matching

        scorer = fragment_scorer_matching.FragmentMatchScorer(
            output_path=scorer_props.get("output_path"),
            min_count_fragments=scorer_props.get("min_count_fragments", 1),
        )
    elif scorer_name == "ScaffoldGNN":
        from navidiv.scaffold_gnn.Scaffold_GNN import ScaffoldGNNScorer

        scorer = ScaffoldGNNScorer(
            output_path=scorer_props.get("output_path"),
        )
    elif scorer_name == "Fragments":
        from navidiv.fragment import fragment_scorer

        scorer = fragment_scorer.FragmentScorer(
            output_path=scorer_props.get("output_path"),
            min_count_fragments=scorer_props.get("min_count_fragments"),
        )
    elif scorer_name == "Ngram":
        from navidiv import Ngram_scorer

        scorer = Ngram_scorer.NgramScorer(
            ngram_size=scorer_props.get("ngram_size", 10),
            output_path=scorer_props.get("output_path"),
        )
    elif scorer_name == "Scaffold":
        from navidiv import Scaffold_scorer

        scorer = Scaffold_scorer.Scaffold_scorer(
            output_path=scorer_props.get("output_path"),
            scaffold_type=scorer_props.get("scaffold_type", "csk_bm"),
        )
    elif scorer_name == "Cluster":
        from navidiv import cluster_similarity_scorer

        scorer = cluster_similarity_scorer.ClusterSimScorer(
            output_path=scorer_props.get("output_path"),
            threshold=scorer_props.get("threshold", 0.25),
        )
    elif scorer_name == "RingScorer":
        from navidiv import ring_scorer

        scorer = ring_scorer.RingScorer(
            output_path=scorer_props.get("output_path"),
        )
    elif scorer_name == "FGscorer":
        from navidiv import fg_scorer

        scorer = fg_scorer.FGScorer(
            output_path=scorer_props.get("output_path"),
            # min_count_fragments=scorer_props.get("min_count_fragments", 1),
        )
    elif scorer_name == "Original":
        from navidiv import orginal_similarity_scorer

        # Use original smiles as reference if available
        df_original = pd.read_csv(
            scorer_props.get(
                "reference_csv",
                "/media/mohammed/Work/Navi_diversity/examples/df_original.csv",
            )
        )
        smiles_list_to_compare_to = df_original[
            get_smiles_column(df_original)
        ].tolist()
        scorer = orginal_similarity_scorer.OriginalSimScorer(
            output_path=scorer_props.get("output_path"),
            threshold=scorer_props.get("threshold", 0.3),
            smiles_list_to_compare_to=smiles_list_to_compare_to,
        )
    else:
        logging.error(
            f"Scorer '{scorer_name}' not recognized. Please check the scorer name."
        )
        raise ValueError(
            f"Scorer '{scorer_name}' not recognized. Please check the scorer name."
        )
        return None

    scorer._min_count_fragments = scorer_props.get("min_count_fragments", 1)
    scorer.update_selection_criteria(
        selection_criteria=scorer_props.get("selection_criteria", {}),
    )
    scorer._output_path = scorer._output_path + "/" + scorer._csv_name
    os.makedirs(scorer._output_path, exist_ok=True)
    # clean the output path
    for file in os.listdir(scorer._output_path):
        file_path = os.path.join(scorer._output_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            logging.exception(f"Error deleting file {file_path}: {e}")

    return scorer
