"""Streamlit web application for molecular diversity analysis.

This module provides a user-friendly interface for molecular diversity analysis
using various scoring functions including frequency-based, similarity-based,
and cluster-based approaches.
"""

from pathlib import Path
from typing import Any

import plotly.io as pio
import streamlit as st
from rdkit import RDLogger


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


def cleanup_temp_files() -> None:
    """Clean up temporary uploaded files."""
    if (hasattr(st.session_state, "file_path") and
        hasattr(st.session_state, "is_uploaded_file") and
        st.session_state.is_uploaded_file and
        st.session_state.file_path):
        try:
            temp_path = Path(st.session_state.file_path)
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass  # File might already be cleaned up


def _get_example_csv_files() -> list[str]:
    """Get list of example CSV files from project directories."""
    default_dir = Path()
    example_dirs = ["Tutorials", "data", "examples"]
    csv_files = []
    
    for dir_name in example_dirs:
        dir_path = default_dir / dir_name
        if dir_path.exists():
            csv_files.extend([
                str(f.relative_to(default_dir))
                for f in dir_path.rglob("*.csv")
            ])
    
    return sorted(csv_files)


def _handle_file_upload() -> str | None:
    """Handle file upload and return file path."""
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="CSV should contain SMILES strings and optionally "
             "'step' and 'Score' columns for analysis",
        key="file_uploader"
    )

    if uploaded_file is not None:
        import tempfile

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            suffix=".csv"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        # Store the temporary file path in session state
        st.session_state.file_path = temp_file_path
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.is_uploaded_file = True

        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Show file info
        file_size = len(uploaded_file.getvalue())
        st.info(f"📊 File size: {file_size:,} bytes")

        return temp_file_path
    
    return None


def _handle_path_input() -> str | None:
    """Handle manual path input and return file path."""
    val = None
    with st.container():
        col_loading = st.columns([3, 1])
        with col_loading[0]:
            path_input = st.text_input(
                "📄 Enter path to your CSV file containing SMILES data",
                key="file_path_input",
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
                if not path_input:
                    st.error("❌ Please enter a valid file path.")
                else:
                    try:
                        # Basic validation
                        file_path = Path(path_input)
                        if not file_path.exists():
                            st.error(f"❌ File not found: {path_input}")
                        elif file_path.suffix.lower() != ".csv":
                            st.warning("⚠️ File should be a CSV (.csv)")
                        else:
                            st.session_state.file_path = path_input
                            st.session_state.is_uploaded_file = False
                            st.success(f"✅ File loaded: {file_path.name}")
                            val = path_input
                    except OSError as e:
                        st.error(f"❌ Error loading file: {e}")
    return val


def _display_current_file_status() -> bool:
    """Display current file status and return whether file is loaded."""
    file_loaded = (hasattr(st.session_state, "file_path") and
                   st.session_state.file_path)

    if file_loaded:
        col1, col2 = st.columns([3, 1])
        with col1:
            file_name = (st.session_state.get("uploaded_file_name",
                        Path(st.session_state.file_path).name))
            st.success(f"✅ **Current file:** {file_name}")
        with col2:
            if st.button("🔄 Change File", help="Load a different file"):
                # Clear current file to expand the section
                keys_to_clear = [
                    "file_path", "uploaded_file_name", "is_uploaded_file"
                ]
                for key in keys_to_clear:
                    if hasattr(st.session_state, key):
                        delattr(st.session_state, key)
                st.rerun()

    return file_loaded


def load_file_section() -> str:
    """Handle file loading section and return the file path."""
    # Display current file status and get load state
    file_loaded = _display_current_file_status()

    # Create expander that collapses when file is loaded
    with st.expander(
        "📁 Load Your Dataset",
        expanded=not file_loaded
    ):
        # Create tabs for different loading methods
        tab_upload, tab_path = st.tabs(["📤 Upload File", "📁 File Path"])

        val = None

    with tab_upload:
        st.markdown("**Upload a CSV file from your computer:**")

        # Option 1: Browse local project files
        st.markdown("**Browse project example files:**")
        csv_files = _get_example_csv_files()

        if csv_files:
            selected_file = st.selectbox(
                "Select an example CSV file:",
                options=["", *csv_files],
                help="Choose from available CSV files in the project"
            )

            if selected_file:
                full_path = str(Path() / selected_file)
                st.session_state.file_path = full_path
                st.session_state.is_uploaded_file = False
                st.success(f"✅ File selected: {selected_file}")
                val = full_path

        st.markdown("**Or upload your own CSV file:**")
        upload_result = _handle_file_upload()
        if upload_result:
            val = upload_result

    with tab_path:
        st.markdown("**Or enter a file path manually:**")
        path_result = _handle_path_input()
        if path_result:
            val = path_result

    # Return the current file path from session state if available
    if hasattr(st.session_state, "file_path") and st.session_state.file_path:
        return st.session_state.file_path

    return val or ""

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
