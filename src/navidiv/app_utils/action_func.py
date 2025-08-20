import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from navidiv.app_utils.description import get_scorer_descriptions
from navidiv.app_utils.scorer_utils import (
    get_scorer_properties_ui,
    run_scorer_on_dataframe,
)


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
        "temporal resolution.",
    )

    with st.sidebar.container():
        with st.expander("ℹ️ **About t-SNE:**", expanded=False):
            st.markdown("""
        
            t-SNE creates 2D visualizations of molecular diversity by reducing
            high-dimensional chemical space to an interactive plot. Points that
            are close together represent structurally similar molecules.
            """)

    if st.sidebar.button(
        "🔬 Run t-SNE Analysis",
        help="Generates t-SNE visualization of molecular diversity. "
        "Creates a new CSV with coordinates.",
        type="secondary",
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
    st.sidebar.markdown("### Run all scorers")

    st.session_state.output_path = st.sidebar.text_input(
        "Output Directory",
        value=Path(file_path).parent / "scorer_output",
        key="scorer_output_path",
        help="Directory where analysis results will be saved. "
        "Creates subdirectories for each scorer type.",
    )

    with st.sidebar.container():
        st.markdown("**Available Scoring Methods:**")
        scorer_descriptions = get_scorer_descriptions()

        # Create a nice overview of what will be run
        selected_scorers = [
            "Frequency",
            "tSNE",
            "Similarity",
            "Activity",
            "UMap",
            "Ngram",
            "Scaffold",
            "Cluster",
            "Original",
            "RingScorer",
            "FGscorer",
            "Fragments_basic",
            "Fragments_default",
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
        type="primary",
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
            st.info(
                "⏳ Running all scorers. This may take a while "
                "depending on the size of the dataset."
            )
            st.info("📊 Results will appear in the tabs once complete.")
            return True
        except subprocess.SubprocessError as e:
            st.error(f"❌ Error running all scorers: {e}")
        except OSError as e:
            st.error(f"❌ File system error: {e}")
    return False


def run_scorer(file_path):
    # --- Scorer UI ---
    data = pd.read_csv(file_path, index_col=False)
    data = data.dropna(axis=1, how="all")
    st.sidebar.markdown("## Run Scorer on DataFrame ")
    scorer_options = [
        "Ngram",
        "Scaffold",
        "Cluster",
        "Original",
        "Fragments",
        "Fragments Match",
        "RingScorer",
        "FGscorer",
    ]
    scorer_name = st.sidebar.selectbox(
        "Scorer", scorer_options, key="scorer_select"
    )
    step_col = "step" if "step" in data.columns else None
    steps = []
    if step_col:
        min_step = int(data[step_col].min())
        max_step = int(data[step_col].max())
        col1,col2 = st.sidebar.columns(2)
        with col1:
            steps = col1.slider(
                "Step range",
                min_value=min_step,
                max_value=max_step,
                value=(min_step, max_step),
                step=1,
            )
        with col2:
            steps_increment = col2.number_input(
                "Step increment",
                min_value=1,
                max_value=max_step - min_step,
                value=10,
                step=1,
            )

        steps = list(range(steps[0], steps[1] + 1, steps_increment))
    # --- Scorer properties UI ---
    scorer_props = get_scorer_properties_ui(scorer_name)
    scorer_props["selection_criteria"] = {}  # selection_criteria_ui()
    scorer_props["scorer_name"] = scorer_name

    run_scorer = st.sidebar.button("Run scorer")
    if run_scorer and step_col:
        run_scorer_on_dataframe(
            data,
            scorer_name,
            steps,
            scorer_props,
        )
    elif run_scorer and not step_col:
        st.warning("No 'step' column found in data. Cannot run scorer.")
