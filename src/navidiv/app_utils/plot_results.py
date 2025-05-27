"""function to plot results from a CSV file using Streamlit and Plotly"""

import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem

from navidiv.app_utils.data_filter import data_filter
from navidiv.app_utils.molecules_drawing import draw_molecule


def plot_results(file_path):
    try:
        st.session_state.df = pd.read_csv(file_path, index_col=False)
        st.session_state.df["index"] = st.session_state.df.index
        st.write("### Data Preview")
        st.dataframe(st.session_state.df.head(2))
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return
    if len(st.session_state.df) == 0:
        st.warning("No data to display.")
        return
    filtered_data = data_filter(st.session_state.df, key="results")
    st.subheader("Interactive Plot")
    col_not_object = [
        col
        for col in filtered_data.columns
        if filtered_data[col].dtype != "object"
    ]

    col_not_object = col_not_object + [
        col
        for col in filtered_data.columns
        if isinstance(filtered_data[col].values[0], str)
        and len(set(filtered_data[col].values)) < 100
    ]  # avoid color scale for large number of unique values

    columns = filtered_data.columns.tolist()
    col_columns_selection = st.columns(3)
    with col_columns_selection[0]:
        x_column = st.selectbox(
            "Select X-axis column _", columns, index=len(columns) - 2
        )
    with col_columns_selection[1]:
        y_column = st.selectbox("Select Y-axis column _", columns)
    with col_columns_selection[2]:
        hue_column = st.selectbox(
            "Select Hue column (optional) _", col_not_object
        )
    filtered_data.reset_index(drop=True, inplace=True)
    fig = px.scatter(
        filtered_data,
        x=x_column,
        y=y_column,
        color=hue_column if hue_column else None,
        hover_data=[filtered_data.index],
    )
    selected_points = st.plotly_chart(
        fig,
        use_container_width=True,
        height=200,
        key="iris_2",
        on_select="rerun",
    )
    if len(selected_points.selection.points) > 0:
        st.session_state.hover_indexs = [
            selected_points.selection.points[x]["customdata"]["0"]
            for x in range(len(selected_points.selection.points))
        ]
        st.dataframe(
            filtered_data.iloc[st.session_state.hover_indexs][
                list(set(["Substructure", x_column, y_column, hue_column]))
            ]
        )
        smiles_column = "Substructure"  # get_smiles_column(filtered_data)
        if smiles_column in filtered_data.columns:
            img = draw_molecule(
                Chem.MolFromSmiles(
                    filtered_data.iloc[st.session_state.hover_indexs][
                        smiles_column
                    ].values[0]
                )
            )
            if img is not None:
                st.image(
                    img,
                    caption=f" {filtered_data.iloc[st.session_state.hover_indexs][smiles_column].values[0]}",
                )
