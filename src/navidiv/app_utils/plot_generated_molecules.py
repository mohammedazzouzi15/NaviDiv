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

from .file_name_registry import initiate_file_name_registry

# Create a global registry instance (or inject as needed)
file_name_registry = initiate_file_name_registry()


def get_data_from_file(file_path):
    """Plot a dataframe using Streamlit and Seaborn."""
    try:
        data = pd.read_csv(file_path, index_col=False)
        data = data.dropna(
            axis=1, how="all"
        )  # Drop columns with all NaN values
        # st.write("#### Data Preview")
        # st.dataframe(data.head(2))
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    filtered_data = data_filter(data)
    filtered_data.reset_index(drop=True, inplace=True)


    # Select columns for x-axis and y-axis
    # st.write("#### Select Columns for Plotting")
    columns = filtered_data.columns.tolist()
    display_columns = [
        file_name_registry.get_display_name(col) for col in columns
    ]
    col_columns_selection = st.columns(3)
    with col_columns_selection[0]:
        x_column_2_display = st.selectbox(
            "X-axis column", display_columns, index=len(display_columns) - 1
        )
        x_column_2 = columns[display_columns.index(x_column_2_display)]
    with col_columns_selection[1]:
        y_column_2_display = st.selectbox(
            "Y-axis column", display_columns, index=len(display_columns) - 2
        )
        y_column_2 = columns[display_columns.index(y_column_2_display)]
    with col_columns_selection[2]:
        hue_column_2_display = st.selectbox(
            "Hue column", display_columns, index=len(display_columns) - 3
        )
        hue_column_2 = columns[display_columns.index(hue_column_2_display)]
    return filtered_data, x_column_2, y_column_2, hue_column_2

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


def plot_generated_molecules(
    filtered_data,
    key="second",
    symbol_column=None,
    x_column=None,
    y_column=None,
    hue_column=None,
):
    """Plot a dataframe using Streamlit and Seaborn."""
    # Plot the data
    if x_column and y_column:
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
            x=x_column,
            y=y_column,
            color_discrete_sequence=["grey"],  # Use grey for background
            hover_data=[filtered_data[background_mask].index],
            render_mode="webgl",
        )

        # Create foreground trace (colored by symbol_column or hue_column_2)
        foreground_trace = px.scatter(
            filtered_data[foreground_mask],
            x=x_column,
            y=y_column,
            color=hue_column if hue_column else None,
            hover_data=[filtered_data[foreground_mask].index],
            render_mode="webgl",
        )
        # Combine both traces into a single figure
        fig_2 = go.Figure()
        fig_2.add_trace(background_trace.data[0])
        fig_2.add_trace(foreground_trace.data[0])
        fig_2.data[0].hovertemplate = ""
        fig_2.data[
            0
        ].hoverinfo = "skip"  # Skip hover info for background trace
        fig_2.data[0].marker.size = 6  # Only background trace
        fig_2.data[1].marker.size = 12  # Only foreground trace
        fig_2.update_layout(
            xaxis_title=file_name_registry.get_display_name(x_column),
            yaxis_title=file_name_registry.get_display_name(y_column),
            hovermode="closest",
        )

        fig_2.update_layout(
            coloraxis_colorbar=dict(
                title=dict(
                    text=file_name_registry.get_display_name(
                        hue_column
                    ),  # or your custom title
                    side="right",  # 'right', 'top', 'bottom'
                    font=dict(size=14),
                ),
            )
        )
        selected_points_2 = st.plotly_chart(
            fig_2,
            use_container_width=True,
            height=200,
            key=key,
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
                        list(set([x_column, y_column, hue_column]))
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
