import argparse
import os

import numpy as np
import pandas as pd

from navidiv.utils import (
    add_mean_of_numeric_columns,
    groupby_results,
    initialize_scorer,
)

SCORERS = [
    "Ngram",
    "Scaffold",
    "Cluster",
    "Original",
    "RingScorer",
    "FGscorer",
    # "Fragments Match",
    "Fragmets_basic",
    "Fragments_default",
    # "ScaffoldGNN",
]


def cumulative_unique_count(list_of_lists):
    seen = set()
    counts = []
    for fragments in list_of_lists:
        seen.update(fragments)
        counts.append(len(seen))
    return counts


def get_default_props(scorer_name, output_path):
    props = {
        "output_path": output_path,
        "scorer_name": scorer_name,
    }

    if scorer_name == "Ngram":
        props["scorer_name"] = "Ngram"
        props["ngram_size"] = 10
        props["min_count_fragments"] = 5
    if scorer_name in ["Scaffold", "Cluster", "Original", "Fragments Match"]:
        props["min_count_fragments"] = 0
    if scorer_name == "Scaffold":
        props["scaffold_type"] = "basic_framework"
    if scorer_name == "Cluster":
        props["threshold"] = 0.25
    if scorer_name == "Fragmets_basic":
        props["tranfomation_mode"] = "basic_wire_frame"
        props["scorer_name"] = "Fragments"
        props["min_count_fragments"] = 2

    if scorer_name == "Fragments_default":
        props["tranfomation_mode"] = "none"
        props["scorer_name"] = "Fragments"
        props["min_count_fragments"] = 2
    if scorer_name == "Original":
        props["threshold"] = 0.3
        props["reference_csv"] = (
            "/media/mohammed/Work/Navi_diversity/data/formed/Molecules_found_by_thanapat.csv"
        )
    return props


def run_scorer(steps, df, scorer, scorer_name, output_path):
    scores_list = []
    succesfull_steps = []
    for step in steps:
        df_step = df[df["step"] == step]
        if df_step.empty:
            continue
        smiles_list = df_step["SMILES"].tolist()
        scores = df_step["Score"].tolist()
        print(
            f"Running {scorer_name} on step {step} with {len(smiles_list)} molecules."
        )
        try:
            scores_result = scorer.get_score(
                smiles_list=smiles_list,
                scores=scores,
                additional_columns_df={"step": step},
            )
            scores_list.append(scores_result)
            succesfull_steps.append(step)
        except Exception as e:
            print(f"Error running {scorer_name} on step {step}: {e}")

    if scores_list:
        df_scores = pd.DataFrame(scores_list)
        if "Unique Fragments" in df_scores.columns:
            df_scores["Cumulative Number of unique Fragments"] = (
                cumulative_unique_count(df_scores["Unique Fragments"].tolist())
            )
            df_scores = df_scores.drop(
                "Unique Fragments", axis=1, errors="ignore"
            )
        if "Total Number of Fragments" in df_scores.columns:
            df_scores["Cumulative Number of Fragments"] = df_scores[
                "Total Number of Fragments"
            ].cumsum()
        if "Cumulative Number of unique Fragments" in df_scores.columns:
            df_scores["Cumulative Percentage of Unique Fragments"] = (
                df_scores["Cumulative Number of unique Fragments"]
                / df_scores["Cumulative Number of Fragments"]
            ) * 100

        dict_mean = add_mean_of_numeric_columns(df, succesfull_steps)
        for col, mean in dict_mean.items():
            df_scores[col] = mean

        df_scores.to_csv(
            f"{output_path}/{scorer._csv_name}/step_scores_{scorer._csv_name}.csv",
            index=False,
        )
        df_fragments = pd.read_csv(
            f"{output_path}/{scorer._csv_name}/{scorer._csv_name}_with_score.csv",
            index_col=False,
        )
        try:
            groupby_results_df = groupby_results(df_fragments)
            groupby_results_df.to_csv(
                f"{output_path}/{scorer._csv_name}/groupby_results_{scorer._csv_name}.csv",
                index=False,
            )
        except Exception as e:
            print(f"Error grouping results for {scorer_name}.")
            print(f"Exception: {e}")
            groupby_results_df = None
        if scorer_name == "Cluster":
            try:
                scores_cluster = scorer.aggregate_df(groupby_results_df)
                scorer._fragments_df.to_csv(
                    f"{output_path}/{scorer._csv_name}/groupby_aggregated_clusters_with_score.csv",
                    index=False,
                )
            except Exception as e:
                print(f"Error aggregating clusters for {scorer_name}.")
                print(f"Exception: {e}")
        return df_scores
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run all scorers with default settings."
    )
    parser.add_argument(
        "--df_path", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_path", default="output", help="Directory to save results"
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    df = pd.read_csv(args.df_path)
    steps = df["step"].unique().tolist()
    steps.sort()
    print(f"Found {len(steps)} unique steps in the dataset.")
    max_number_of_steps = 100
    if len(steps) > max_number_of_steps:
        indices = np.linspace(
            0, len(steps) - 1, max_number_of_steps, dtype=int
        )
        steps = [steps[i] for i in indices]

    for scorer_name in SCORERS:
        print(f"Running scorer: {scorer_name}")
        props = get_default_props(scorer_name, args.output_path)
        scorer = initialize_scorer(props)
        run_scorer(
            steps=steps,
            df=df,
            scorer=scorer,
            scorer_name=scorer_name,
            output_path=args.output_path,
        )
        print(f"Finished running scorer: {scorer_name}")


if __name__ == "__main__":
    main()
