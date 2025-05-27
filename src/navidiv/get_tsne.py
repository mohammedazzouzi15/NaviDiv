import time

import numpy as np
from sklearn.manifold import TSNE
import logging

# from cuml.manifold import TSNE
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def get_fingerprints(molecules):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=5, fpSize=2048, countSimulation=True
    )
    fingerprints = [mfpgen.GetFingerprint(mol) for mol in molecules]

    # Convert fingerprints to numpy array
    fingerprints_array = np.array(
        [np.array(fp).astype(int) for fp in fingerprints]
    )
    return fingerprints_array, fingerprints


def perform_tsne(fingerprints_array):
    """Perform t-SNE dimensionality reduction, with options to save or load the model."""

    # Perform t-SNE using sklearn
    if fingerprints_array.shape[0] > 10000:
        raise ValueError(
            "t-SNE is not supported for more than 10000 samples. Please reduce the number of samples."
        )
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        n_iter=1000,
        learning_rate=10,
        random_state=42,
        method="barnes_hut",
    )
    tsne_results = tsne.fit_transform(fingerprints_array)

    return tsne_results


def plot_TSNE(
    df,
    output_path,
    color_scale,
):
    """Generate and save a t-SNE plot."""
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df[df["File"] == "original_data"],
        x="TSNEx",
        y="TSNEy",
        color="grey",
        alpha=0.6,
        label="Original Data",
    )
    sns.scatterplot(
        data=df[df["File"] != "original_data"],
        x="TSNEx",
        y="TSNEy",
        hue=color_scale,
        # palette="viridis",
        alpha=0.6,
        legend="brief",
    )

    # Hide the axes and the frame
    plt.axis("off")
    plt.gca().set_frame_on(False)

    plt.legend([], [], frameon=False)
    plt.savefig(output_path)
    plt.close()


def get_smiles_column(df):
    for col in ["smiles", "SMILES", "Smiles"]:
        if col in df.columns:
            return col
    raise ValueError("No column containing SMILES strings found.")


if __name__ == "__main__":
    # run_name = "170425_DAP2_Substructure_high"
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine .txt files into a DataFrame."
    )
    parser.add_argument(
        "--df_path",
        type=str,
        default="170425_DAP2_Substructure_high",
        help="Name of the run to process.",
    )
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()
    step = args.step
    df_path = args.df_path
    output_csv = df_path.replace(".csv", "_TSNE.csv")

    df = pd.read_csv(df_path).sample(frac=1, random_state=1)
    if "step" in df.columns:
        df = df[df["step"] % step == 0]
        logging.info(
            f"Filtered DataFrame to include only rows where step is a multiple of {step}."
        )
        logging.info(f"Filtered DataFrame shape: {df.shape}")

    smiles_col = get_smiles_column(df)
    df.drop_duplicates(subset=[smiles_col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=[smiles_col], inplace=True)
    df.drop_duplicates(subset=[smiles_col], inplace=True)

    # Run t-SNE
    df["mols"] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
    df.dropna(subset=["mols"], inplace=True)
    fingerprints_array, fingerprints = get_fingerprints(df["mols"])
    # Perform t-SNE
    tsne_results = perform_tsne(fingerprints_array)
    # Add t-SNE results to the DataFrame
    df["TSNEx"] = tsne_results[:, 0]
    df["TSNEy"] = tsne_results[:, 1]
    df.drop(columns=["mols"], inplace=True)
    # Save t-SNE results
    tsne_output_csv = output_csv
    df.to_csv(tsne_output_csv, index=False)
