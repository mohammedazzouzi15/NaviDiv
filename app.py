import glob
import os
import subprocess
from pathlib import Path

import plotly.io as pio
import streamlit as st

from navidiv.app_utils.file_name_registry import initiate_file_name_registry
from navidiv.app_utils.plot_generated_molecules import (
    get_data_from_file,
    plot_generated_molecules,
)
from navidiv.app_utils.plot_results import plot_results, plot_step_results
from navidiv.utils import get_smiles_column

# Create a global registry instance (or inject as needed)
file_name_registry = initiate_file_name_registry()


def initialize_app():
    """Initialize the Streamlit app settings."""
    st.set_page_config(layout="wide", initial_sidebar_state="auto")
    pio.templates.default = "plotly"
    #st.title("Analyse Generated Molecules")
    return True



def do_tsne(file_path):
    """Run t-SNE analysis on the provided CSV file.

    Args:
        file_path (str): Path to the CSV file containing molecular data.

    Returns:
        bool: True if t-SNE analysis was successful, False otherwise.
    """
    step_increment = st.sidebar.number_input(
        "step increment", min_value=1, max_value=100, value=10, step=1
    )
    if st.sidebar.button("Run t-SNE"):
        cmd = [
            "python3",
            "src/navidiv/get_tsne.py",
            "--df_path",
            file_path,
            "--step",
            str(step_increment),
        ]
        try:
            subprocess.run(cmd, check=True)
            st.success("t-SNE analysis complete.")
            st.session_state.file_path = file_path.replace(".csv", "_TSNE.csv")
            return True
        except Exception as e:
            st.error(f"Error running t-SNE: {e}")
    return False


def run_all_scorers(file_path):
    """Run all scorers with default settings on the provided CSV file.

    Args:
        file_path (str): Path to the CSV file containing molecular data.

    Returns:
        bool: True if scoring was successful, False otherwise.
    """
    st.session_state.output_path = st.sidebar.text_input(
        "Output path",
        value=Path(file_path).parent / "scorer_output",
        key="scorer_output_path",
    )
    if st.sidebar.button("Run All Scorers"):
        cmd = [
            "python3",
            "src/navidiv/run_all_scorers.py",
            "--df_path",
            file_path,
            "--output_path",
            st.session_state.scorer_output_path,
        ]
        try:
            subprocess.Popen(cmd)
            st.text(
                "Running all scorers. This may take a while depending on the size of the dataset."
            )
            return True
        except Exception as e:
            st.error(f"Error running all scorers: {e}")
    return False


def on_change_file_path():
    st.session_state.file_path = st.session_state.file_path_input




# Streamlit app
def main() -> None:
    """Main entry point for the Streamlit app."""
    initialised = initialize_app()

    if initialised:
        with st.container():
            col_loading = st.columns(2)
            with col_loading[0]:
                val = st.text_input(
                    "Enter the path to your CSV file",
                    key="file_path_input",
                    on_change=on_change_file_path,
                    label_visibility="collapsed",
                )
            with col_loading[1]:
                if st.button("Load CSV file"):
                    if not val:
                        st.error("Please enter a valid file path.")
                    else:
                        st.session_state.file_path = val
                        st.success(f"File loaded: {st.session_state.file_path}")
        st.divider()
        col1, col2, col3 = st.columns([2, 2, 1])
        if val:
            with col1:
                do_tsne(st.session_state.file_path)
                run_all_scorers(st.session_state.file_path)

                filtered_data, x_column_2, y_column_2, hue_column_2 = (
                    get_data_from_file(st.session_state.file_path)
                )
                # --- Tabs for all data vs. fragment/substructure ---
                tab_all, tab_frag = st.tabs(
                    [
                        "All Molecules",
                        "Molecules Containing Fragment/Substructure",
                    ]
                )
                with tab_all:
                    st.markdown(
                        "**All Molecules:** This tab shows all data points from the loaded file."
                    )
                    # Show display names for selected columns
                    plot_generated_molecules(
                        filtered_data,
                        symbol_column=None,
                        x_column=x_column_2,
                        y_column=y_column_2,
                        hue_column=hue_column_2,
                        key="molecules_all",
                    )
                with tab_frag:
                    st.markdown(
                        "**Molecules Containing Fragment/Substructure:** This tab only shows molecules containing the selected fragment or substructure."
                    )
                    if (
                        hasattr(
                            st.session_state,
                            "list_of_molecules_containing_fragment",
                        )
                        and st.session_state.list_of_molecules_containing_fragment
                    ):
                        filtered_data["Molecules containing fragment"] = (
                            filtered_data[
                                get_smiles_column(filtered_data)
                            ].apply(
                                lambda x: x
                                in st.session_state.list_of_molecules_containing_fragment
                            )
                        )
                        plot_generated_molecules(
                            filtered_data,
                            symbol_column="Molecules containing fragment",
                            x_column=x_column_2,
                            y_column=y_column_2,
                            hue_column=hue_column_2,
                            key="molecules_frag",
                        )
                    else:
                        st.info(
                            "No fragment/substructure selection available."
                        )
        else:
            st.info("Load the CSV to start the analysis. ")
            st.info(
                "Ensure the CSV file contains a column with SMILES strings."
            )
            st.info(
                "For the scorer analysis ensure that the CSV contains a column with the name step and one with the column Score."
            )
        st.divider()
        with col2:
            tab_per_fragment, tab_per_step = st.tabs(
                ["Per Fragment Results", "Per Step Results"]
            )
            with tab_per_fragment:
                if hasattr(st.session_state, "output_path"):
                    st.write(
                        "Displays the occurance of fragments in the generated dataset"
                    )
                    if not os.path.exists(st.session_state.output_path):
                        os.makedirs(st.session_state.output_path)
                    csv_files = [
                        Path(files).relative_to(st.session_state.output_path)
                        for files in glob.glob(
                            f"{st.session_state.output_path}/*/group*.csv"
                        )
                    ]

                    file_path_results = st.selectbox(
                        "Select a CSV file from folder",
                        csv_files,
                        format_func=lambda x: file_name_registry.get_display_name(
                            x.parent.name
                        )
                        if isinstance(x, Path)
                        else x,
                    )
                    if file_path_results:
                        plot_results(
                            f"{st.session_state.output_path}/{file_path_results}",
                            col3,
                        )
            with tab_per_step:
                if hasattr(st.session_state, "output_path"):
                    st.markdown(
                        "Displays the Average evolution of the metrics per generation step."
                    )
                    if not os.path.exists(st.session_state.output_path):
                        os.makedirs(st.session_state.output_path)
                    csv_files = [
                        Path(files).relative_to(st.session_state.output_path)
                        for files in glob.glob(
                            f"{st.session_state.output_path}/*/step_*.csv"
                        )
                    ]

                    file_path_results = st.selectbox(
                        "Select a CSV file from folder",
                        csv_files,
                        key="file_path_results",
                        format_func=lambda x: file_name_registry.get_display_name(
                            x.parent.name
                        )
                        if isinstance(x, Path)
                        else x,
                    )
                    if file_path_results:
                        plot_step_results(
                            f"{st.session_state.output_path}/{file_path_results}"
                        )


if __name__ == "__main__":
    main()

