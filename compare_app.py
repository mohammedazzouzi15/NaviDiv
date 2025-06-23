import glob
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from navidiv.app_utils import file_name_registry

file_name_registry = file_name_registry.initiate_file_name_registry()
import plotly.express as px
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
                    filename=Path(file).name,
                    data=pd.read_csv(file),
                    experiment_type=Path(file).parent.parent.parent.name,
                    scorer_type=Path(file).parent.name,
                )
                self.add_result(results)

    def generate_summary(self):
        """Generate a summary of the results."""
        summary = []
        for key, result in self.results.items():
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

    def filter_results(
        self,
        experiment_type: str | None = None,
        scorer_type: str | None = None,
    ) -> dict:
        """Filter results by experiment type and/or scorer type.

        Args:
            experiment_type (Optional[str], optional): The type of experiment to filter by. Defaults to None.
            scorer_type (Optional[str], optional): The type of scorer to filter by. Defaults to None.

        Returns:
            dict: A dictionary of filtered results.
        """
        return {
            key: result
            for key, result in self.results.items()
            if (
                experiment_type is None
                or result.experiment_type == experiment_type
            )
            and (scorer_type is None or result.scorer_type == scorer_type)
        }

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


def plot_step_results(results_list):
    """Plot multiple lines for selected results from Results_list."""

    if not results_list.results:
        st.warning("No results to display.")
        return

    # Generate summary DataFrame for selection
    summary_df = results_list.generate_summary()

    # Initialize session state for selected results
    if "selected_results" not in st.session_state:
        st.session_state.selected_results = []

    # Create two columns: one for data selection and one for the figure
    col_data, col_plot = st.columns([1, 2])

    with col_data:
        st.write("Filter results:")
        selected_experiment = st.selectbox(
            "Experiment Type",
            options=["All"] + summary_df["experiment_type"].unique().tolist(),
            index=0,
        )
        selected_scorer = st.selectbox(
            "Scorer Type",
            options=["All"] + summary_df["scorer_type"].unique().tolist(),
            index=0,
        )
        if selected_experiment == "All" and selected_scorer == "All":
            filtered_df = summary_df
        else:
            filtered_df = summary_df[
                (
                    summary_df["experiment_type"] == selected_experiment
                    if selected_experiment != "All"
                    else True
                )
                & (
                    summary_df["scorer_type"] == selected_scorer
                    if selected_scorer != "All"
                    else True
                )
            ]

        def get_name(result):
            if selected_experiment != "All":
                name = f"{result.scorer_type}"
            elif selected_scorer != "All":
                name = f"{result.experiment_type}"
            else:
                # If both filters are "All", use the result key as the name
                name = f"{result.experiment_type} ({result.scorer_type})"
            return name

        def format_func(df, idx):
            result_key = list(results_list.results.keys())[idx]
            result = results_list.results[result_key]
            return get_name(result)

        st.write("Select results to plot:")
        selected_results = st.multiselect(
            "Results",
            options=filtered_df.index,
            default=filtered_df.index[:5],  # Default to first 5 results
            format_func=lambda idx: format_func(filtered_df, idx),
        )

        # Add a selection for plot type
        plot_type = st.radio(
            "Select Plot Type:", options=["Line Plot", "Scatter Plot"], index=0
        )

        # Update session state with the current selection
        if (
            "selected_results" not in st.session_state
            or st.session_state["selected_results"] != selected_results
        ):
            st.session_state["selected_results"] = selected_results

        if not selected_results:
            st.info("Please select at least one result to plot.")
            return

        # Select columns for X and Y axes
        columns = list(results_list.results.values())[0].data.columns.tolist()
        col_selection = st.columns(2)
        with col_selection[0]:
            x_column = st.selectbox(
                "X-axis column",
                columns,
                key="x_column_steps",
                index=columns.index("step"),
            )
        with col_selection[1]:
            y_column = st.selectbox(
                "Y-axis column", columns, key="y_column_steps", index=1
            )

    with col_plot:
        # Plot selected results
        fig = px.line()
        for idx in selected_results:
            result_key = list(results_list.results.keys())[idx]
            result = results_list.results[result_key]
            filtered_data = result.data
            name = get_name(result)
            if (
                x_column in filtered_data.columns
                and y_column in filtered_data.columns
            ):
                fig.add_scatter(
                    x=filtered_data[x_column],
                    y=filtered_data[y_column],
                    mode="lines" if plot_type == "Line Plot" else "markers",
                    name=name,
                )

        fig.update_layout(
            xaxis_title=file_name_registry.get_display_name(x_column),
            yaxis_title=file_name_registry.get_display_name(y_column),
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the Streamlit app."""
    # Initialize Streamlit app
    st.set_page_config(page_title="Compare Results App", layout="wide")

    # Input: User specifies folders
    folder_input = st.text_area("Enter folder paths (one per line):")
    folder_paths = [
        path.strip() for path in folder_input.split("\n") if path.strip()
    ]

    if st.button("Generate Summary"):
        st.session_state.generate_summary = True
    if st.session_state.get("generate_summary", False):
        if folder_paths:
            results_list = Results_list([])
            results_list.read_results_from_files(folder_paths)
            summary_df = results_list.generate_summary()
            if not summary_df.empty:
                st.write("Summary DataFrame:")
                st.dataframe(summary_df)
                plot_step_results(results_list)
            else:
                st.info("No valid results found in the specified folders.")
        else:
            st.error("Please enter at least one folder path.")


if __name__ == "__main__":
    main()
