import io
import logging
import os
import streamlit as st

import pandas as pd
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D


def get_smiles_column(df):
    for col in ["smiles", "SMILES", "Smiles"]:
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
    grouped_by_df = df.groupby(["Substructure"]).agg(
        {
            "Count": ["sum", "max"],
            "step": ["max", "min", "count"],
            "median_score_fragment": ["max", "min", "mean"],
            "diff_median_score": ["max", "min", "mean"],
            "Count_perc_per_molecule": ["max", "first", "mean"],
            "count_per_molecule": ["max", "first", "mean", "sum"],
        }
    )
    # transform the groubed_by_df to a dataframe
    grouped_by_df = grouped_by_df.reset_index()
    grouped_by_df.columns = [
        "Substructure",
        "Count",
        "Count_max",
        "step_max",
        "step_min",
        "step_count",
        "median_score_fragment_max",
        "median_score_fragment_min",
        "median_score_fragment_mean",
        "diff_median_score_max",
        "diff_median_score_min",
        "diff_median_score_mean",
        "Count_perc_per_molecule_max",
        "Count_perc_per_molecule_first",
        "Count_perc_per_molecule_mean",
        "count_per_molecule_max",
        "count_per_molecule_first",
        "count_per_molecule_mean",
        "count_per_molecule",
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
    try:
        if scorer_name == "Fragments":
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
        elif scorer_name == "Original":
            from navidiv import orginal_similarity_scorer

            # Use original smiles as reference if available
            df_original = pd.read_csv(
                scorer_props.get(
                    "reference_csv",
                    "/media/mohammed/Work/Navi_diversity/examples/df_original.csv",
                )
            )
            smiles_list_to_compare_to = df_original["smiles"].tolist()
            scorer = orginal_similarity_scorer.OriginalSimScorer(
                output_path=scorer_props.get("output_path"),
                threshold=scorer_props.get("threshold", 0.3),
                smiles_list_to_compare_to=smiles_list_to_compare_to,
            )
        else:
            logging.error(
                f"Scorer '{scorer_name}' not recognized. Please check the scorer name."
            )
            return None
    except Exception as e:
        logging.exception(f"Error initializing scorer '{scorer_name}': {e}")
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
