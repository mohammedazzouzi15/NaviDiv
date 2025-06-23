import streamlit as st
import pandas as pd
import os

def read_results_from_folder(folder_path: str) -> pd.DataFrame:
    """Reads CSV files from the specified folder and concatenates them into a single DataFrame."""
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a summary of the DataFrame."""
    return df.describe()

st.title("Folder Results Summary App")

# Let user input folder paths (Streamlit does not support folder picker directly)
folders = st.text_area(
    "Enter folder paths (one per line):",
    placeholder="/path/to/folder1\n/path/to/folder2"
)
folder_list = [f.strip() for f in folders.splitlines() if f.strip()]

if st.button("Read and Summarize"):
    all_data = []
    for folder in folder_list:
        if os.path.isdir(folder):
            df = read_results_from_folder(folder)
            if not df.empty:
                all_data.append(df)
            else:
                st.warning(f"No CSV files found in {folder}")
        else:
            st.error(f"Folder not found: {folder}")
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        summary_df = summarize_data(combined_df)
        st.dataframe(summary_df)
    else:
        st.info("No data to display.")