"""Functions to filter data based on various criteria."""

import pandas as pd
import streamlit as st
from .file_name_registry import FileNameRegistry, initiate_file_name_registry

# Create a global registry instance (or inject as needed)
file_name_registry = initiate_file_name_registry()


def data_filter(data, key="ori",columns_option=None):
    # --- Data Filtering UI ---
    # Add expander to show/hide data filtering section
    show_filter = st.checkbox(
        "Show data filtering options", value=False, key=f"show_filter_{key}"
    )
    if columns_option is None:
        columns_option = data.columns.tolist()
    filter_columns = []
    if show_filter:
        st.markdown("#### Data Filtering")
        filter_columns = st.multiselect(
            "Select columns to filter (optional)",
            options=columns_option,
            format_func=lambda x: file_name_registry.get_display_name(x),
            key=f"filter_columns_{key}",
        )
        # Map display names back to internal names for filtering

    else:
        # When hidden, use last filter_columns from session state
        filter_columns = st.session_state.get(f"filter_columns_{key}", [])

    filtered_data = data
    # Apply filters for each selected column
    for col in filter_columns:
        col_dtype = data[col].dtype
        # Use display name in UI labels
        display_col = file_name_registry.get_display_name(col)
        if pd.api.types.is_numeric_dtype(col_dtype):
            min_val = float(filtered_data[col].min())
            max_val = float(filtered_data[col].max())
            if show_filter:
                filter_range = st.slider(
                    f"Filter {display_col} range",
                    min_value=min_val,
                    max_value=max_val,
                    value=st.session_state.get(
                        f"filter_range_{key}_{col}", (min_val, max_val)
                    ),
                    step=(max_val - min_val) / 100
                    if max_val > min_val
                    else 1.0,
                    key=f"filter_range_{key}_{col}",
                )
            else:
                filter_range = st.session_state.get(
                    f"filter_range_{key}_{col}", (min_val, max_val)
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
            if show_filter:
                selected_categories = st.multiselect(
                    f"Select {display_col} values",
                    options=categories,
                    default=st.session_state.get(
                        f"filter_categories_{key}_{col}", categories
                    ),
                    key=f"filter_categories_{key}_{col}",
                )
            else:
                selected_categories = st.session_state.get(
                    f"filter_categories_{key}_{col}", categories
                )
            filtered_data = filtered_data[
                filtered_data[col].isin(selected_categories)
            ]
        else:
            st.sidebar.info(
                f"Filtering for column '{display_col}' type is not supported."
            )
    st.write("number of rows", len(filtered_data))
    return filtered_data
