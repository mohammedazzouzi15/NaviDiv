import pandas as pd

csv_file = "/media/mohammed/Work/Navi_diversity/examples/results/ngrams_10_selected_fragments.csv"
df = pd.read_csv(csv_file)
print("columns", df.columns)
def groupby_results(df):
    grouped_by_df = df.groupby(["Substructure"]).agg(
        {
            "Count": "sum",
            "step": ["max", "min","count"],
            "median_score_fragment": ["max", "min", "mean"],
            "diff_median_score": ["max", "min", "mean"],
        }
    )
    #transform the groubed_by_df to a dataframe
    grouped_by_df = grouped_by_df.reset_index()
    grouped_by_df.columns = [
        "Substructure",
        "Count",
        "step_max",
        "step_min",
        "step_count",
        "median_score_fragment_max",
        "median_score_fragment_min",
        "median_score_fragment_mean",
        "diff_median_score_max",
        "diff_median_score_min",
        "diff_median_score_mean",
    ]
    
    return grouped_by_df
grouped_by_df = groupby_results(df)
