"""Streamlit web application for molecular diversity analysis.

This module provides a user-friendly interface for molecular diversity analysis
using various scoring functions including frequency-based, similarity-based,
and cluster-based approaches.
"""
import subprocess
from pathlib import Path
from typing import Any

import plotly.io as pio
import streamlit as st
from rdkit import RDLogger

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


def get_scorer_descriptions() -> dict[str, dict[str, str]]:
    """Get descriptions for different scoring functions."""
    return {
        "Ngram": {
            "title": "N-gram String Analysis",
            "description": "Analyzes molecular diversity using string-based "
                          "n-gram patterns from SMILES representations. "
                          "Identifies common substrings that may represent "
                          "chemical motifs.",
            "use_case": "Useful for identifying string-based patterns and "
                       "recurring sequences in molecular representations."
        },
        "Scaffold": {
            "title": "Scaffold Diversity Analysis",
            "description": "Extracts and analyzes molecular scaffolds using "
                          "Murcko framework decomposition. Focuses on the "
                          "core ring systems and connecting bonds.",
            "use_case": "Essential for understanding the structural diversity "
                       "of core molecular frameworks in your dataset."
        },
        "Cluster": {
            "title": "Molecular Clustering",
            "description": "Groups molecules based on structural similarity "
                          "using molecular fingerprints and Tanimoto "
                          "similarity metrics.",
            "use_case": "Identifies clusters of structurally similar "
                       "molecules and analyzes cluster diversity."
        },
        "Original": {
            "title": "Reference Dataset Comparison",
            "description": "Compares generated molecules against a reference "
                          "dataset to identify novel vs. known structures.",
            "use_case": "Evaluates how well your generated molecules match "
                       "or diverge from known molecular space."
        },
        "RingScorer": {
            "title": "Ring System Analysis",
            "description": "Focuses specifically on ring systems and cyclic "
                          "structures within molecules.",
            "use_case": "Analyzes the diversity of ring systems, which are "
                       "crucial for drug-like properties."
        },
        "FGscorer": {
            "title": "Functional Group Analysis",
            "description": "Identifies and analyzes functional groups "
                          "present in the molecular dataset.",
            "use_case": "Evaluates the chemical functionality diversity "
                       "through functional group distribution."
        },
        "Fragmets_basic": {
            "title": "Basic Fragment Analysis",
            "description": "Performs fragment analysis using basic wire "
                          "frame transformation (atoms → carbon, "
                          "preserve bonds).",
            "use_case": "Analyzes structural patterns while focusing on "
                       "connectivity rather than atom types."
        },
        "Fragments_default": {
            "title": "Default Fragment Analysis",
            "description": "Standard fragment analysis without chemical "
                          "transformations, preserving original atom types "
                          "and bonds.",
            "use_case": "Comprehensive fragment analysis maintaining full "
                       "chemical information."
        }
    }


def initialize_app() -> bool:
    """Initialize the Streamlit app settings."""
    st.set_page_config(
        page_title="NaviDiv - Molecular Diversity Analysis",
        layout="wide",
        initial_sidebar_state="auto",
        page_icon="🧬"
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


def create_scoring_info_section() -> None:
    """Create an expandable information section about scoring functions."""
    with st.expander(
        "ℹ️ **About Molecular Diversity Scoring Functions**",
        expanded=False
    ):
        st.markdown("""
        ### Overview
        This tool provides multiple scoring functions to analyze different
        aspects of molecular diversity:
        """)

        scorer_descriptions = get_scorer_descriptions()

        for scorer_name, info in scorer_descriptions.items():
            st.markdown(f"""
            **{info['title']}** (`{scorer_name}`)

            {info['description']}

            *{info['use_case']}*

            ---
            """)

        st.markdown("""
        ### How to Use
        1. **Load your CSV file** containing SMILES strings and optionally
           'step' and 'Score' columns
        2. **Run t-SNE** to create 2D visualizations of molecular diversity
        3. **Run All Scorers** to perform comprehensive diversity analysis
        4. **Explore results** in the Per Fragment and Per Step tabs

        ### Requirements
        - CSV file must contain a column with SMILES strings
        - For scorer analysis: include 'step' and 'Score' columns
        - For t-SNE: ensure sufficient molecular diversity in your dataset
        """)


def do_tsne(file_path: str) -> bool:
    """Run t-SNE analysis on the provided CSV file.

    Args:
        file_path (str): Path to the CSV file containing molecular data.

    Returns:
        bool: True if t-SNE analysis was successful, False otherwise.
    """
    st.sidebar.markdown("### 📊 t-SNE Visualization Settings")

    step_increment = st.sidebar.number_input(
        "Step Increment",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Defines how many generation steps to skip between t-SNE "
             "calculations. Higher values = faster processing but lower "
             "temporal resolution."
    )

    with st.sidebar.container():
        st.markdown("""
        **About t-SNE:**

        t-SNE creates 2D visualizations of molecular diversity by reducing
        high-dimensional chemical space to an interactive plot. Points that
        are close together represent structurally similar molecules.
        """)

    if st.sidebar.button(
        "🔬 Run t-SNE Analysis",
        help="Generates t-SNE visualization of molecular diversity. "
             "Creates a new CSV with coordinates.",
        type="secondary"
    ):
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
            st.success("✅ t-SNE analysis completed successfully!")
            st.session_state.file_path = file_path.replace(".csv", "_TSNE.csv")
            st.info(f"📄 Updated dataset: {st.session_state.file_path}")
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error running t-SNE analysis: {e}")
        except OSError as e:
            st.error(f"❌ File system error: {e}")
    return False


def run_all_scorers(file_path: str) -> bool:
    """Run all scorers with default settings on the provided CSV file.

    Args:
        file_path (str): Path to the CSV file containing molecular data.

    Returns:
        bool: True if scoring was successful, False otherwise.
    """
    st.sidebar.markdown("### 🎯 Diversity Analysis Settings")

    st.session_state.output_path = st.sidebar.text_input(
        "Output Directory",
        value=Path(file_path).parent / "scorer_output",
        key="scorer_output_path",
        help="Directory where analysis results will be saved. "
             "Creates subdirectories for each scorer type."
    )

    with st.sidebar.container():
        st.markdown("**Available Scoring Methods:**")
        scorer_descriptions = get_scorer_descriptions()

        # Create a nice overview of what will be run
        selected_scorers = [
            "Frequency", "tSNE", "Similarity", "Activity", "UMap",
            "Ngram", "Scaffold", "Cluster", "Original", "RingScorer",
            "FGscorer", "Fragmets_basic", "Fragments_default"
        ]

        # Show scorer descriptions in an expandable section
        if selected_scorers:
            with st.expander("📋 View Selected Scorers", expanded=False):
                for scorer in selected_scorers:
                    if scorer in scorer_descriptions:
                        st.markdown(
                            f"• **{scorer}**: "
                            f"{scorer_descriptions[scorer]['title']}"
                        )

    if st.sidebar.button(
        "🚀 Run All Scorers",
        help="Executes all diversity scoring methods on your dataset. "
             "This may take several minutes for large datasets.",
        type="primary"
    ):
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
            st.success("✅ Diversity analysis started successfully!")
            st.info("⏳ Running all scorers. This may take a while "
                   "depending on the size of the dataset.")
            st.info("📊 Results will appear in the tabs once complete.")
            return True
        except subprocess.SubprocessError as e:
            st.error(f"❌ Error running all scorers: {e}")
        except OSError as e:
            st.error(f"❌ File system error: {e}")
    return False


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
                placeholder="/media/mohammed/Work/Navi_diversity/tests/test_data/default/default_1_TSNE.csv",
                help="CSV should contain SMILES strings and optionally "
                     "'step' and 'Score' columns for analysis"
            )
        with col_loading[1]:
            if st.button(
                "📂 Load File",
                type="primary",
                help="Load and validate the CSV file"
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


def create_analysis_tools_section(file_path: str) -> None:
    """Create the analysis tools section."""
    st.markdown("### 🔬 Analysis Tools")

    # Analysis buttons in sidebar
    do_tsne(file_path)
    run_all_scorers(file_path)

    # Main visualization
    try:
        filtered_data, x_column_2, y_column_2, hue_column_2 = (
            get_data_from_file(file_path)
        )

        # Tabs for different views
        tab_all, tab_frag = st.tabs([
            "🧬 All Molecules",
            "🎯 Fragment Analysis",
        ])

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
                filtered_data["Molecules containing fragment"] = (
                    filtered_data[get_smiles_column(filtered_data)].apply(
                        lambda x: x in
                        st.session_state.list_of_molecules_containing_fragment
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
                st.info("🔍 No fragment selection available. "
                       "Run fragment analysis first.")

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.info("💡 Please ensure your CSV contains valid SMILES strings.")


def create_results_section(col3: Any) -> None:
    """Create the results analysis section."""
    st.markdown("### 📊 Analysis Results")

    tab_per_fragment, tab_per_step = st.tabs([
        "🧩 Per Fragment", "📈 Per Step"
    ])

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
                    ) if isinstance(x, Path) else x,
                    help="Choose which fragment analysis results to display"
                )
                if file_path_results:
                    plot_results(f"{output_path}/{file_path_results}", col3)
            else:
                st.info("🔄 No fragment results available yet. "
                       "Run 'All Scorers' to generate analysis.")

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
                    ) if isinstance(x, Path) else x,
                    help="Choose which step-wise analysis results to display"
                )
                if file_path_results:
                    plot_step_results(f"{output_path}/{file_path_results}")
            else:
                st.info("🔄 No step results available yet. "
                       "Run 'All Scorers' to generate analysis.")


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
