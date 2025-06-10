from navidiv.utils import (
    add_mean_of_numeric_columns,
    groupby_results,
    initialize_scorer,
)
import pandas as pd
output_path = "/media/mohammed/Work/Navi_diversity/reinvent_runs/runs/test/test4/stage0/scorer_output"
name = "Fragments_Match"
csv_file = "/media/mohammed/Work/Navi_diversity/reinvent_runs/runs/test/test4/stage0/scorer_output/Fragments_Match/Fragments_Match_with_score.csv"
df = pd.read_csv(csv_file)
print("columns", df.columns)

grouped_by_df = groupby_results(df)
grouped_by_df.to_csv(
                f"{output_path}/{name}/groupby_results_{name}.csv",
                index=False,
            )