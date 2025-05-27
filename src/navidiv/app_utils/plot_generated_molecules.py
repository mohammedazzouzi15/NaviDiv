"""Module for plotting results from a CSV file in Streamlit."""

import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem

from navidiv.app_utils.data_filter import data_filter
from navidiv.app_utils.molecules_drawing import draw_molecule
from navidiv.app_utils.scorer_utils import (
    get_scorer_properties_ui,
    run_scorer_on_dataframe,
    selection_criteria_ui,
)
from navidiv.utils import (
    get_smiles_column,
)


def plot_generated_molecules(file_path):
    """Plot a dataframe using Streamlit and Seaborn."""
    try:
        data = pd.read_csv(file_path, index_col=False)
        st.write("### Data Preview")
        st.dataframe(data.head(2))
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    filtered_data = data_filter(data)
    filtered_data.reset_index(drop=True, inplace=True)

    # --- Scorer UI ---

    st.sidebar.markdown("## Run Scorer on DataFrame ")
    scorer_options = ["Ngram", "Scaffold", "Cluster", "Original", "Fragments"]
    scorer_name = st.sidebar.selectbox(
        "Select scorer", scorer_options, key="scorer_select"
    )
    step_col = "step" if "step" in filtered_data.columns else None
    steps = []
    if step_col:
        min_step = int(filtered_data[step_col].min())
        max_step = int(filtered_data[step_col].max())
        steps = st.sidebar.slider(
            "Select step range",
            min_value=min_step,
            max_value=max_step,
            value=(min_step, max_step),
            step=1,
        )
        steps_increment = st.sidebar.number_input(
            "Select step increment",
            min_value=1,
            max_value=max_step - min_step,
            value=10,
            step=1,
        )
        st.session_state.output_path = st.sidebar.text_input(
            "Output path for scorer results (optional)",
            value="examples/results",
        )
        steps = list(range(steps[0], steps[1] + 1, steps_increment))
    # --- Scorer properties UI ---
    scorer_props = get_scorer_properties_ui(scorer_name)
    scorer_props["selection_criteria"] = selection_criteria_ui()
    scorer_props["scorer_name"] = scorer_name

    run_scorer = st.sidebar.button("Run scorer")
    if run_scorer and step_col:
        run_scorer_on_dataframe(
            filtered_data,
            scorer_name,
            steps,
            scorer_props,
        )
    elif run_scorer and not step_col:
        st.warning("No 'step' column found in data. Cannot run scorer.")

    # Select columns for x-axis and y-axis
    st.write("### Select Columns for Plotting")
    columns = filtered_data.columns.tolist()
    col_columns_selection = st.columns(3)
    with col_columns_selection[0]:
        x_column_2 = st.selectbox("Select X-axis column", columns)
    with col_columns_selection[1]:
        y_column_2 = st.selectbox("Select Y-axis column", columns)
    with col_columns_selection[2]:
        hue_column_2 = st.selectbox("Select Hue column (optional)", columns)

    # Plot the data
    if x_column_2 and y_column_2:
        fig_2 = px.scatter(
            filtered_data,
            x=x_column_2,
            y=y_column_2,
            color=hue_column_2 if hue_column_2 else None,
            hover_data=[filtered_data.index],
            render_mode="webgl",
        )
        selected_points_2 = st.plotly_chart(
            fig_2,
            use_container_width=True,
            height=200,
            key="iris",
            on_select="rerun",
        )
        if len(selected_points_2.selection.points) > 0:
            st.session_state.hover_indexs = [
                selected_points_2.selection.points[x]["customdata"]["0"]
                for x in range(len(selected_points_2.selection.points))
            ]
            st.dataframe(
                filtered_data.iloc[st.session_state.hover_indexs][
                    list(set([x_column_2, y_column_2, hue_column_2]))
                ]
            )
            smiles_column = get_smiles_column(filtered_data)
            img = draw_molecule(
                Chem.MolFromSmiles(
                    filtered_data.iloc[st.session_state.hover_indexs][
                        smiles_column
                    ].to_numpy()[0]
                )
            )
            if img is not None:
                st.image(
                    img,
                    caption=(
                        f" {filtered_data.iloc[st.session_state.hover_indexs][smiles_column].to_numpy()[0]}"
                    ),
                )
