"""Streamlit web application for molecular diversity analysis.

This module provides a user-friendly interface for molecular diversity analysis
using various scoring functions including frequency-based, similarity-based,
and cluster-based approaches.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import plotly.io as pio
import streamlit as st
from rdkit import RDLogger

# Check if navidiv is installed, if not install it in development mode
try:
    import navidiv  # noqa: F401
except ImportError:
    st.error("NaviDiv package not found. Installing in development mode...")
    try:
        # Get the current directory (where the app is running from)
        current_dir = Path(__file__).parent
        
        # Install the package in development mode
        # Using trusted input (sys.executable and hardcoded command)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", str(current_dir)
        ], capture_output=True, text=True, check=True)
        
        st.success("NaviDiv package installed successfully!")
        st.info(
            "Please restart the Streamlit app to use the newly installed "
            "package."
        )
        st.stop()
        
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install NaviDiv package: {e}")
        st.error(f"Error output: {e.stderr}")
        st.info(
            "Please manually install the package by running: "
            "`pip install -e .` in the project directory"
        )
        st.stop()
    except (OSError, RuntimeError) as e:
        st.error(f"Installation error: {e}")
        st.stop()

from navidiv.app_utils.action_func import do_tsne, run_all_scorers, run_scorer
from navidiv.app_utils.description import (
    create_scoring_info_section,
)
from navidiv.app_utils.file_name_registry import initiate_file_name_registry
from navidiv.app_utils.plot_generated_molecules import (
    get_data_from_file,
    plot_generated_molecules,
)
from navidiv.app_utils.plot_results import plot_results, plot_step_results
from navidiv.utils import get_smiles_column

RDLogger.logger().setLevel(RDLogger.ERROR)

# Create a global registry instance (or inject as needed)
file_name_registry = initiate_file_name_registry()


def initialize_app() -> bool:
    """Initialize the Streamlit app settings."""
    st.set_page_config(
        page_title="NaviDiv - Molecular Diversity Analysis",
        layout="wide",
        initial_sidebar_state="auto",
        page_icon="🧬",
    )
    pio.templates.default = "plotly"

    # Add main title and description
    st.title("🧬 NaviDiv - Molecular Diversity Analysis")
    st.markdown("""
    **A comprehensive tool for analyzing molecular diversity in datasets.**

    Upload your CSV file containing SMILES strings to start exploring
    molecular diversity through various scoring methods, t-SNE visualization,
    and fragment analysis.
    """)
    st.divider()
    return True


def on_change_file_path() -> None:
    """Handle file path input changes."""
    st.session_state.file_path = st.session_state.file_path_input


def load_file_section() -> str:
    """Handle file loading section and return the file path."""
    st.markdown("### 📁 Load Your Dataset")

    with st.container():
        col_loading = st.columns([3, 1])
        with col_loading[0]:
            val = st.text_input(
                "📄 Enter path to your CSV file containing SMILES data",
                key="file_path_input",
                on_change=on_change_file_path,
                placeholder="Tutorials/Using_The_app/example/default_1_TSNE.csv",
                value="Tutorials/Using_The_app/example/default_1_TSNE.csv",
                help="CSV should contain SMILES strings and optionally "
                "'step' and 'Score' columns for analysis",
            )
        with col_loading[1]:
            if st.button(
                "📂 Load File",
                type="primary",
                help="Load and validate the CSV file",
            ):
                if not val:
                    st.error("❌ Please enter a valid file path.")
                else:
                    try:
                        # Basic validation
                        file_path = Path(val)
                        if not file_path.exists():
                            st.error(f"❌ File not found: {val}")
                        elif file_path.suffix.lower() != ".csv":
                            st.warning("⚠️ File should be a CSV (.csv)")
                        else:
                            st.session_state.file_path = val
                            st.success(f"✅ File loaded: {file_path.name}")
                    except OSError as e:
                        st.error(f"❌ Error loading file: {e}")

    return val

def sidebar_analysis(file_path):
        # Analysis buttons in sidebar
    do_tsne(file_path)
    run_all_scorers(file_path)

    run_scorer(file_path)



def create_analysis_tools_section(file_path: str) -> None:
    """Create the analysis tools section."""
    st.markdown("### 🔬Chemical space:")

    # Main visualization
    try:
        filtered_data, x_column_2, y_column_2, hue_column_2 = (
            get_data_from_file(file_path)
        )

        # Tabs for different views
        tab_all, tab_frag = st.tabs(
            [
                "🧬 All Molecules",
                "🎯 Fragment Analysis",
            ]
        )

        with tab_all:
            st.markdown(
                "**All Molecules View:** Comprehensive visualization of "
                "all molecules in your dataset."
            )
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
                "**Fragment Analysis:** Focused view on molecules containing "
                "specific structural fragments."
            )
            if (
                hasattr(
                    st.session_state, "list_of_molecules_containing_fragment"
                )
                and st.session_state.list_of_molecules_containing_fragment
            ):
                filtered_data["Molecules containing fragment"] = filtered_data[
                    get_smiles_column(filtered_data)
                ].apply(
                    lambda x: x
                    in st.session_state.list_of_molecules_containing_fragment
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
                    "🔍 No fragment selection available. "
                    "Run fragment analysis first."
                )

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.info("💡 Please ensure your CSV contains valid SMILES strings.")


def create_results_section(col3: Any) -> None:
    """Create the results analysis section."""
    st.markdown("### 📊 Analysis Results")

    tab_per_fragment, tab_per_step = st.tabs(
        ["🧩 Per Fragment", "📈 Per Step"]
    )

    with tab_per_fragment:
        if hasattr(st.session_state, "output_path"):
            st.markdown(
                "**Fragment Occurrence Analysis:** Shows how frequently "
                "different molecular fragments appear in your dataset."
            )

            output_path = Path(st.session_state.output_path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)

            csv_files = list(output_path.glob("*/group*.csv"))
            csv_files = [f.relative_to(output_path) for f in csv_files]

            if csv_files:
                file_path_results = st.selectbox(
                    "Select Fragment Analysis Results",
                    csv_files,
                    format_func=lambda x: file_name_registry.get_display_name(
                        x.parent.name
                    )
                    if isinstance(x, Path)
                    else x,
                    help="Choose which fragment analysis results to display",
                )
                if file_path_results:
                    plot_results(f"{output_path}/{file_path_results}", col3)
            else:
                st.info(
                    "🔄 No fragment results available yet. "
                    "Run 'All Scorers' to generate analysis."
                )

    with tab_per_step:
        if hasattr(st.session_state, "output_path"):
            st.markdown(
                "**Evolution Analysis:** Displays the evolution of diversity "
                "metrics across generation steps."
            )

            output_path = Path(st.session_state.output_path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)

            csv_files = list(output_path.glob("*/step_*.csv"))
            csv_files = [f.relative_to(output_path) for f in csv_files]

            if csv_files:
                file_path_results = st.selectbox(
                    "Select Step Evolution Results",
                    csv_files,
                    key="file_path_results",
                    format_func=lambda x: file_name_registry.get_display_name(
                        x.parent.name
                    )
                    if isinstance(x, Path)
                    else x,
                    help="Choose which step-wise analysis results to display",
                )
                if file_path_results:
                    plot_step_results(f"{output_path}/{file_path_results}")
            else:
                st.info(
                    "🔄 No step results available yet. "
                    "Run 'All Scorers' to generate analysis."
                )


def main() -> None:
    """Main entry point for the Streamlit app."""
    initialised = initialize_app()

    if not initialised:
        return
    # Add information section
    create_scoring_info_section()

    # File loading section
    val = load_file_section()
    st.divider()

    # Main analysis layout
    col1, col2, col3 = st.columns([2, 2, 1])

    if val and hasattr(st.session_state, "file_path"):
        # Left column - Analysis tools
        sidebar_analysis(st.session_state.file_path)

        with col1:
            create_analysis_tools_section(st.session_state.file_path)
    else:
        with col1:
            st.info("👆 **Getting Started:** Load your CSV file above.")
            st.markdown("""
            **Requirements:**
            - 📊 CSV file with SMILES strings
            - 📈 Optional: 'step' and 'Score' columns for evolution analysis
            - 🧪 Recommended: At least 100+ molecules for diversity analysis
            """)

    # Right columns - Results
    with col2:
        create_results_section(col3)


if __name__ == "__main__":
    main()
