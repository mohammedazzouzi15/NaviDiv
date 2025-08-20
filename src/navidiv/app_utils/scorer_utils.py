"""Collection of functions to handle scoring of molecules in a dataframe."""

import pandas as pd
import streamlit as st

from navidiv.utils import (
    add_mean_of_numeric_columns,
    groupby_results,
    initialize_scorer,
)


def cumulative_unique_count(list_of_lists):
    seen = set()
    counts = []
    for fragments in list_of_lists:
        seen.update(fragments)
        counts.append(len(seen))
    return counts


def run_scorer_on_dataframe(
    data,
    scorer_name,
    steps,
    scorer_props,
):
    # Dynamically import scorer modules
    scorer = initialize_scorer(scorer_props)
    scores_list = []
    successful_steps = []
    score_status_placeholder = st.sidebar.empty()

    # Initialize session state for stop functionality
    if "stop_scoring" not in st.session_state:
        st.session_state.stop_scoring = False

    # Reset stop flag at the beginning of scoring
    st.session_state.stop_scoring = False

    for step in steps:
        # Check if stop button was clicked
        if st.session_state.stop_scoring:
            st.warning(f"Scoring stopped by user at step {step}")
            break

        if hasattr(scorer, "_mol_smiles"):
            delattr(scorer, "_mol_smiles")
        df_copy = data[data["step"] == step]
        if df_copy.empty:
            continue
        smiles_list = df_copy["SMILES"].tolist()
        scores = df_copy["Score"].tolist()

        try:
            scores_result = scorer.get_score(
                smiles_list=smiles_list,
                scores=scores,
                additional_columns_df={"step": step},
            )
            scores_list.append(scores_result)
            successful_steps.append(step)

            # Calculate progress and time estimates
            progress_percentage = len(successful_steps) / len(steps)
            elapsed_time = scores_result.get("Elapsed Time", 0)

            # Update progress with comprehensive status and stop button
            with score_status_placeholder.container():
                # Add stop button at the top
                if st.button(
                    "🛑 Stop Scoring", key=f"stop_btn_{step}", type="secondary"
                ):
                    st.session_state.stop_scoring = True
                    st.rerun()

                st.progress(progress_percentage)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Step", f"{step}")
                    st.metric(
                        "Progress", f"{len(successful_steps)}/{len(steps)}"
                    )
                with col2:
                    if elapsed_time and elapsed_time != "N/A":
                        remaining_steps = len(steps) - len(successful_steps)
                        estimated_time_left = elapsed_time * remaining_steps
                        st.metric("Elapsed Time/Step", f"{elapsed_time:.2f}s")
                        st.metric(
                            "Est. Remaining", f"{estimated_time_left:.2f}s"
                        )
                    else:
                        st.metric("Elapsed Time/Step", "N/A")
                        st.metric("Est. Remaining", "N/A")

        except Exception as e:
            st.error(f"Scorer error in '{scorer_name}' at step {step}: {e}")
            continue

        scores_result_copy = scores_result.copy()
        if "Unique Fragments" in scores_result_copy:
            scores_result_copy.pop("Unique Fragments")

    # Clear the stop button after completion or stopping
    with score_status_placeholder.container():
        if st.session_state.stop_scoring:
            st.info("Scoring was stopped by user")
        else:
            st.success("Scoring completed successfully")

    # Process results if any steps were completed
    if scores_list:
        df_scores = pd.DataFrame(scores_list)

        if "Unique Fragments" in df_scores.columns:
            df_scores["Cumulative Number of unique Fragments"] = (
                cumulative_unique_count(df_scores["Unique Fragments"].tolist())
            )
            df_scores["Cumulative Number of Fragments"] = df_scores[
                "Total Number of Fragments"
            ].cumsum()
            df_scores["Cumulative Percentage of Unique Fragments"] = (
                df_scores["Cumulative Number of unique Fragments"]
                / df_scores["Cumulative Number of Fragments"]
            ) * 100
            df_scores = df_scores.drop(
                "Unique Fragments", axis=1, errors="ignore"
            )
        dict_mean = add_mean_of_numeric_columns(data, successful_steps)
        for col, mean in dict_mean.items():
            df_scores[col] = mean

        df_scores.to_csv(
            f"{st.session_state.output_path}/{scorer._csv_name}/step_scores_{scorer._csv_name}.csv",
            index=False,
        )
        df_fragments = pd.read_csv(
            f"{st.session_state.output_path}/{scorer._csv_name}/{scorer._csv_name}_with_score.csv",
            index_col=False,
        )
        try:
            groupby_results_df = groupby_results(df_fragments)
            if scorer_name != "Cluster":
                groupby_results_df.to_csv(
                    f"{st.session_state.output_path}/{scorer._csv_name}/groupby_results_{scorer._csv_name}.csv",
                    index=False,
                )
        except Exception as e:
            st.error(f"Error grouping results: {e}")
            groupby_results_df = None
        if scorer_name == "Cluster":
            try:
                print(
                    f"Aggregating clusters for {scorer_name} with {len(groupby_results_df)} groups."
                )
                groupby_results_df.to_csv(
                    f"{st.session_state.output_path}/{scorer._csv_name}/_results_before_aggregated_clusters.csv",
                    index=False,
                )
                scores_cluster = scorer.aggregate_df(groupby_results_df)
                scorer._fragments_df.to_csv(
                    f"{scorer._output_path}/groupby_results_aggregated_clusters.csv",
                    index=False,
                )
                print(f"Cluster aggregation completed for {scorer_name}.")
            except Exception as e:
                st.error(f"Error aggregating clusters: {e}")
                scores_cluster = None
        return df_scores
    return None


def get_scorer_properties_ui(scorer_name):
    """Display UI for scorer properties and return a dict of properties."""
    props = {}
    if scorer_name == "Ngram":
        col1, col2 = st.sidebar.columns(2)
        props["ngram_size"] = col1.number_input(
            "Ngram size",
            min_value=2,
            max_value=20,
            value=10,
            step=1,
            help="Size of the n-grams to consider",
        )
        props["min_count_fragments"] = col2.number_input(
            "Min Molecule Count",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Minimum number of molecules containing a specific sequence to keep for analysis",
        )
        props["output_path"] = st.session_state.output_path

    if scorer_name in ["Scaffold", "Cluster", "Original"]:
        col1, col2 = st.sidebar.columns(2)
        props["output_path"] = st.session_state.output_path
        props["min_count_fragments"] = col1.number_input(
            "Min Molecule Count",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Minimum number of molecules in a cluster to keep for analysis ",
        )
    if scorer_name == "Scaffold":
        props["scaffold_type"] = col2.selectbox(
            "Scaffold type",
            [
                "basic_wire_frame",
                "murcko_framework",
                "elemental_wire_frame",
                "basic_framework",
            ],
            index=0,
        )
    if scorer_name == "Cluster":
        # Create a comprehensive cluster configuration
        cluster_config = {}

        # Similarity metric selection
        similarity_metric = col2.selectbox(
            "Similarity Metric",
            ["tanimoto", "dice"],
            index=0,
            help="Similarity metric for comparing molecules",
        )
        cluster_config["similarity_metric"] = similarity_metric

        # Fingerprint configuration
        fingerprint_type = col1.selectbox(
            "Fingerprint Type",
            ["morgan", "rdkit"],
            index=0,
            help="Type of molecular fingerprint",
        )
        cluster_config["fingerprint_type"] = fingerprint_type

        if fingerprint_type == "morgan":
            fingerprint_radius = col1.number_input(
                "Fingerprint Radius",
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                help="Radius for Morgan fingerprints",
            )
            cluster_config["fingerprint_radius"] = fingerprint_radius

        # Clustering method selection
        clustering_method = col2.selectbox(
            "Clustering Method",
            ["threshold"],
            index=0,
            help="Algorithm for clustering molecules",
        )
        cluster_config["clustering_method"] = clustering_method

        # Method-specific parameters
        clustering_params = {}

        if clustering_method == "threshold":
            threshold = col2.number_input(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                help="Molecules with similarity above this are clustered "
                "together",
            )
            clustering_params["threshold"] = threshold

        elif clustering_method == "hierarchical":
            n_clusters = col2.number_input(
                "Number of Clusters",
                min_value=2,
                max_value=50,
                value=10,
                step=1,
                help="Target number of clusters to create",
            )
            clustering_params["n_clusters"] = n_clusters

            linkage = col2.selectbox(
                "Linkage Method",
                ["average", "complete", "single"],
                index=0,
                help="Linkage criterion for hierarchical clustering",
            )
            clustering_params["linkage"] = linkage

        elif clustering_method == "dbscan":
            eps = col2.number_input(
                "DBSCAN eps",
                min_value=0.01,
                max_value=2.0,
                value=0.5,
                step=0.01,
                help="Maximum distance for molecules to be neighbors",
            )
            clustering_params["eps"] = eps

            min_samples = col2.number_input(
                "Minimum Samples",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                help="Minimum samples in a neighborhood for core points",
            )
            clustering_params["min_samples"] = min_samples

        cluster_config["clustering_params"] = clustering_params
        props["cluster_config"] = cluster_config

    if scorer_name == "Original":
        props["threshold"] = col1.number_input(
            "Original similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
        )
        props["reference_csv"] = col2.text_input(
            "Reference CSV path",
            value="/media/mohammed/Work/Navi_diversity/examples/df_original.csv",
            help="Path to the reference CSV file containing the smiles of the molecules to compare the results to",
        )
    if (
        scorer_name == "Fragments"
        or scorer_name == "Fragments Match"
        or scorer_name == "FGscorer"
        or scorer_name == "RingScorer"
    ):
        col1, col2 = st.sidebar.columns(2)
        props["output_path"] = st.session_state.output_path
        props["min_count_fragments"] = col1.number_input(
            "Min Mol",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Minimum number of molecules containing a fragment to keep for analysis",
        )
        props["transformation_mode"] = col2.selectbox(
            "Transformation mode",
            [
                "none",
                "basic_framework",
                "elemental_wire_frame",
                "basic_wire_frame",
            ],
            index=0,
            help="Transformation mode for fragments",
        )
    return props


def selection_criteria_ui():
    """Example:
    selection_criteria = {
        "Count_perc_per_molecule": 1,
        "Count_perc": 1,
        "diff_median_score": -5,
        "median_score_fragment": 0,
    }
    """
    col_selection_criteria = st.sidebar.columns(3)
    with col_selection_criteria[0]:
        count_perc_per_molecule = st.sidebar.number_input(
            "Count percentage per molecule",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=1.0,
        )
    with col_selection_criteria[1]:
        count_perc = st.sidebar.number_input(
            "Median score for molecule containing fragment",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=1.0,
        )
    with col_selection_criteria[2]:
        diff_median_score = st.sidebar.number_input(
            "Difference median score",
            min_value=-100.0,
            max_value=100.0,
            value=-5.0,
            step=1.0,
        )
    selection_criteria = {
        "Count_perc_per_molecule": count_perc_per_molecule,
        "Count_perc": count_perc,
        "diff_median_score": diff_median_score,
    }
    return selection_criteria
