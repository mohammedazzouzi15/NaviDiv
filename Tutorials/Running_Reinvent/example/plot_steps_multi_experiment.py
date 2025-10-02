import glob
import os
import pickle  # Add this import

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Result:
    def __init__(
        self, filename, data: pd.DataFrame, experiment_type, scorer_type
    ):
        self.filename = filename
        self.data = data
        self.experiment_type = experiment_type
        self.scorer_type = scorer_type
        self.ax = None  # Placeholder for the axis if needed
        self.number_of_csvs = 1

    def __repr__(self):
        return f"Result(filename={self.filename}, experiment_type={self.experiment_type}), scorer_type={self.scorer_type}"

    def update_data(self, new_data):
        """Update the data attribute with new data."""
        # average the new data with existing data using the step column
        # concatenate them
        self.data = (
            pd.concat([self.data, new_data])
            .groupby("step")
            .mean()
            .reset_index()
        )
        self.number_of_csvs += 1


class Results_list:
    def __init__(self, results):
        self.results = {}
        for result in results:
            if isinstance(result, Result):
                self.results.update({f"{results}": result})

    def add_result(self, result):
        """Add a new Result object to the results list."""
        if isinstance(result, Result):
            if f"{result}" not in self.results:
                self.results[f"{result}"] = result
            else:
                self.results[f"{result}"].update_data(result.data)
        else:
            raise ValueError("Only Result objects can be added.")

    def read_results_from_files(self, folders):
        """Read Result objects from a list of file paths."""
        for folder in folders:
            files = glob.glob(
                os.path.join(f"{folder}/*/*/step*.csv"), recursive=True
            )
            for file in files:
                results = Result(
                    filename=os.path.basename(file),
                    data=pd.read_csv(file),
                    experiment_type=os.path.basename(os.path.abspath(folder)),
                    scorer_type=os.path.basename(os.path.dirname(file)),
                )
                self.add_result(results)

    def generate_summary(self):
        """Generate a summary of the results."""
        summary = []
        for result in self.results:
            summary.append(
                {
                    "filename": result.filename,
                    "experiment_type": result.experiment_type,
                    "scorer_type": result.scorer_type,
                    "data_shape": result.data.shape,
                    "number_of_csvs": result.number_of_csvs,
                }
            )
        return pd.DataFrame(summary)

    def __repr__(self):
        return f"Results_list(results={self.results})"


def add_repeated_columns_plot(
    df,
    axes,
    axis_dict,
    axis_counter,
    scores_labels_dict,
    col_x,
    folder_name,
    column_name,
    ylim=(0, 1),
):
    if column_name in df.columns:
        if column_name in axis_dict:
            ax = axis_dict[column_name]
        else:
            axis_counter += 1
            ax = axes[axis_counter - 1]
            axis_dict[column_name] = ax
        if f"{folder_name}" not in scores_labels_dict:
            scores_labels_dict[f"{folder_name}"] = True
            sns.lineplot(
                ax=ax,
                x=col_x,
                y=column_name,
                data=df,
                label=f"{folder_name}",
            )
            ax.set_xlim([0, 1000])  # Adjust x-axis limits as needed
            ax.set_ylim(ylim)  # Adjust y-axis limits as needed
    return axis_counter, axis_dict, scores_labels_dict


def plot_steps(folders, col_x, col_y, save_pickle_path=None):
    """Plot col_x vs col_y from all 'step*.csv' files in given folders.
    Each CSV gets its own subplot (axis).
    Each folder's traces are colored and labeled by folder name.
    Also plots the evolution of the 'score' column (if present) on a secondary y-axis.
    """
    # Collect all step*.csv files per folder
    all_files = []
    results_dict = {}
    for folder in folders:
        files = glob.glob(
            os.path.join(f"{folder}/*/*/step*.csv"), recursive=True
        )
        for file in files:
            results = Result(
                filename=os.path.basename(file),
                data=pd.read_csv(file),
                experiment_type=os.path.basename(os.path.abspath(folder)),
                scorer_type=os.path.basename(os.path.dirname(file)),
            )
            if f"{results}" not in results_dict:
                results_dict[f"{results}"] = results
            else:
                data_to_append = pd.read_csv(file)
                results_dict[f"{results}"].update_data(data_to_append)

        all_files.append((os.path.basename(os.path.abspath(folder)), files))
    axis_dict = {}
    axis_counter = 0
    scores_labels_dict = {}  # To store labels for scores
    prior_labels_dict = {}
    max_csvs = max(len(files) for _, files in all_files)
    print(f"Max CSVs found: {max_csvs}")
    fig, axes = plt.subplots(
        max_csvs // 2 + 10, 2, figsize=(16, 2 * max_csvs), squeeze=False
    )
    axes = axes.flatten()
    for key, result in results_dict.items():
        # print(f"Experiment Type: {key}, Results: {result}")
        df = result.data
        folder_name = result.experiment_type
        axis_counter, axis_dict, scores_labels_dict = (
            add_repeated_columns_plot(
                df,
                axes,
                axis_dict,
                axis_counter,
                scores_labels_dict,
                col_x,
                folder_name,
                "Score",
            )
        )
        axis_counter, axis_dict, prior_labels_dict = add_repeated_columns_plot(
            df,
            axes,
            axis_dict,
            axis_counter,
            prior_labels_dict,
            col_x,
            folder_name,
            "Prior",
            ylim=(20, 50),
        )
        if col_x not in df.columns:
            continue

        y_cols_present = [col for col in col_y if col in df.columns]
        for y_cols_present in col_y:
            if y_cols_present not in df.columns:
                continue
            if (
                f"{os.path.basename(result.filename)}_{y_cols_present}"
                in axis_dict
            ):
                ax = axis_dict[
                    f"{os.path.basename(result.filename)}_{y_cols_present}"
                ]
            else:
                axis_counter += 1
                ax = axes[axis_counter - 1]
                axis_dict[
                    f"{os.path.basename(result.filename)}_{y_cols_present}"
                ] = ax

            # Plot the first col_y on left y-axis, second (if present) on right y-axis

            sns.lineplot(
                ax=ax,
                x=col_x,
                y=y_cols_present,
                data=df,
                label=f"{folder_name}",
            )

            ax.set_title(f"{os.path.basename(result.filename)}")
            ax.set_xlabel(col_x)
        ax.set_xlim([0, 1000])  # Adjust x-axis limits as needed

    # Add legends and layout
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    print(f"saveing figure to {save_pickle_path.replace('.pkl', '.png')}")
    plt.savefig(save_pickle_path.replace(".pkl", ".png"), dpi=300)
    # Save figure as pickle for later modification if requested
    if save_pickle_path:
        with open(save_pickle_path, "wb") as f:
            pickle.dump(fig, f)
    return axes  # Return axes for further customization if needed


if __name__ == "__main__":
    # Example folders and columns
    main_folders = [
        "/media/mohammed/Work/Navi_diversity/example_other_cases/runs/test_case_1",
    ]
    experiment_types = [
        "All_constraints",
        "All_weak_constraints",
        "1_default",
        "fragement_only",
        "ngram_only",
        # "scaffold_only",
        "similarity_only",
    ]
    example_folders = [
        os.path.join(main_folder, exp_type)
        for main_folder in main_folders
        for exp_type in experiment_types
    ]
    print(f"number of folders: {len(example_folders)}")
    os.makedirs("figures/raw", exist_ok=True)

    axes = plot_steps(
        example_folders,
        "step",
        [
            "Appeared more than 10 times",
            "mean_distance",
            "mean_similarity",
            "Percentage of Unique Fragments",
        ],
        save_pickle_path="figures/raw/steps_plot_multi_experiment.pkl",  # Save as pickle
    )
