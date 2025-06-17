"""Functions to filter data based on various criteria."""

import pandas as pd
import streamlit as st


def data_filter(data, key="ori"):
    # --- Data Filtering UI ---
    #st.markdown("## Data Filtering")
    # Allow multiple columns to be selected for filtering
    filter_columns = st.multiselect(
        "Select columns to filter (optional)",
        options=data.columns.tolist(),
        default=[],
        key=f"filter_columns_{key}",
    )
    filtered_data = data
    # Apply filters for each selected column
    for col in filter_columns:
        col_dtype = data[col].dtype
        if pd.api.types.is_numeric_dtype(col_dtype):
            min_val = float(filtered_data[col].min())
            max_val = float(filtered_data[col].max())
            filter_range = st.slider(
                f"Filter {col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                step=(max_val - min_val) / 100 if max_val > min_val else 1.0,
                key=f"filter_range_{key}_{col}",
            )
            filtered_data = filtered_data[
                (filtered_data[col] >= filter_range[0])
                & (filtered_data[col] <= filter_range[1])
            ]
        elif (
            pd.api.types.is_categorical_dtype(col_dtype)
            or data[col].nunique() < 50
        ):
            categories = filtered_data[col].unique().tolist()
            selected_categories = st.multiselect(
                f"Select {col} values",
                options=categories,
                default=categories,
                key=f"filter_categories_{key}_{col}",
            )
            filtered_data = filtered_data[filtered_data[col].isin(selected_categories)]
        else:
            st.sidebar.info(f"Filtering for column '{col}' type is not supported.")
    st.write("number of rows", len(filtered_data))
    return filtered_data
