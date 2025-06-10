"""Module for plotting results from a CSV file in Streamlit."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Add this import at the top if not present
import streamlit as st
from rdkit import Chem

from navidiv.app_utils.data_filter import data_filter
from navidiv.app_utils.molecules_drawing import draw_molecule
from navidiv.app_utils.scorer_utils import (
    get_scorer_properties_ui,
    run_scorer_on_dataframe,
)
from navidiv.utils import (
    get_smiles_column,
)


def plot_generated_molecules_from_file(file_path):
    """Plot a dataframe using Streamlit and Seaborn."""
    try:
        data = pd.read_csv(file_path, index_col=False)
        data = data.dropna(
            axis=1, how="all"
        )  # Drop columns with all NaN values
        st.write("#### Data Preview")
        st.dataframe(data.head(2))
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    filtered_data = data_filter(data)
    filtered_data.reset_index(drop=True, inplace=True)

    # --- Scorer UI ---

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
    step_col = "step" if "step" in filtered_data.columns else None
    steps = []
    if step_col:
        min_step = int(filtered_data[step_col].min())
        max_step = int(filtered_data[step_col].max())
        steps = st.sidebar.slider(
            "Step range",
            min_value=min_step,
            max_value=max_step,
            value=(min_step, max_step),
            step=1,
        )
        steps_increment = st.sidebar.number_input(
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
        x_column_2 = st.selectbox("X-axis column", columns)
    with col_columns_selection[1]:
        y_column_2 = st.selectbox("Y-axis column", columns)
    with col_columns_selection[2]:
        hue_column_2 = st.selectbox("Hue column", columns)

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
        fig_2.update_layout(
            coloraxis_colorbar=dict(
                title=dict(
                    text=hue_column_2,  # or your custom title
                    side="right",  # 'right', 'top', 'bottom'
                    font=dict(size=14),
                ),
                # You can also adjust x/y to move the colorbar itself
                # x=1.05,  # move colorbar horizontally
                # y=0.5,   # move colorbar vertically
            )
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
    return filtered_data


def plot_generated_molecules(filtered_data, key="second", symbol_column=None):
    """Plot a dataframe using Streamlit and Seaborn."""
    # Select columns for x-axis and y-axis
    columns = filtered_data.columns.tolist()
    col_columns_selection = st.columns(3)
    with col_columns_selection[0]:
        x_column_2 = st.selectbox(
            "X-axis column", columns, key=f"x_column_{key}"
        )
    with col_columns_selection[1]:
        y_column_2 = st.selectbox(
            "Y-axis column", columns, key=f"y_column_{key}"
        )
    with col_columns_selection[2]:
        hue_column_2 = st.selectbox(
            "Hue column", columns, key=f"hue_column_{key}"
        )

    # Plot the data
    if x_column_2 and y_column_2:
        # Example: select background points by a condition (e.g., a boolean mask or a subset)
        # Here, let's say you want all points where symbol_column is None or a specific value as background
        if symbol_column and symbol_column in filtered_data.columns:
            foreground_mask = filtered_data[symbol_column]
            background_mask = ~foreground_mask
        else:
            # If no symbol_column, just plot all as foreground
            background_mask = pd.Series([False] * len(filtered_data))
            foreground_mask = ~background_mask

        # Create background trace (grey, low opacity)
        background_trace = px.scatter(
            filtered_data[background_mask],
            x=x_column_2,
            y=y_column_2,
            color_discrete_sequence=["grey"],  # Use grey for background
            # color="grey",  # No color for background
            hover_data=[filtered_data[background_mask].index],
            render_mode="webgl",
        )

        # Create foreground trace (colored by symbol_column or hue_column_2)
        foreground_trace = px.scatter(
            filtered_data[foreground_mask],
            x=x_column_2,
            y=y_column_2,
            color=hue_column_2 if hue_column_2 else None,
            hover_data=[filtered_data[foreground_mask].index],
            render_mode="webgl",
        )
        # Combine both traces into a single figure
        fig_2 = go.Figure()
        fig_2.add_trace(background_trace.data[0])
        fig_2.add_trace(foreground_trace.data[0])
        fig_2.data[1].marker.size = 16  # Only foreground trace
        fig_2.update_layout(
            # title=f"Plot of {x_column_2} vs {y_column_2}",
            xaxis_title=x_column_2,
            yaxis_title=y_column_2,
        )

        fig_2.update_layout(
            coloraxis_colorbar=dict(
                title=dict(
                    text=hue_column_2,  # or your custom title
                    side="right",  # 'right', 'top', 'bottom'
                    font=dict(size=14),
                ),
                # You can also adjust x/y to move the colorbar itself
                # x=1.05,  # move colorbar horizontally
                # y=0.5,   # move colorbar vertically
            )
        )
        selected_points_2 = st.plotly_chart(
            fig_2,
            use_container_width=True,
            height=200,
            key=f"iris_{key}",
            on_select="rerun",
        )
        if len(selected_points_2.selection.points) > 0:
            st.session_state.hover_indexs = [
                selected_points_2.selection.points[x]["customdata"]["0"]
                for x in range(len(selected_points_2.selection.points))
            ]
            if any(
                x in filtered_data[foreground_mask].index
                for x in st.session_state.hover_indexs
            ):
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
