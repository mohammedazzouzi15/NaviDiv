import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def get_fingerprints(molecules):
    """Generate Morgan fingerprints for a list of molecules."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=5, fpSize=2048, countSimulation=True
    )
    fingerprints = [mfpgen.GetFingerprint(mol) for mol in molecules]
    return fingerprints


def calculate_similarity(fp_list1, fp_list2):
    """Calculate the Tanimoto similarity between two sets of fingerprints."""
    return np.array(
        [DataStructs.BulkTanimotoSimilarity(fp1, fp_list2) for fp1 in fp_list1]
    )


def plot_similarity_histogram(
    molecules1, molecules2, output_file, title="Similarity Histogram"
):
    """Plot a histogram of Tanimoto similarities between two sets of molecules."""
    # Generate fingerprints
    fps1 = get_fingerprints(molecules1)
    fps2 = get_fingerprints(molecules2)

    # Calculate similarities
    similarities = calculate_similarity(fps1, fps2)
    print(f"Calculated {len(similarities.flatten())} similarities.")

    # Create a DataFrame for plotting
    df = pd.DataFrame(similarities.flatten(), columns=["Similarity"])

    # Plot the histogram
    ax = df.plot.hist(bins=50, alpha=0.7, title=title)
    ax.set_xlabel("Tanimoto Similarity")
    ax.set_ylabel("Frequency")

    # Save the plot
    ax.get_figure().savefig(output_file)
    print(f"Histogram saved to {output_file}")
    print(f"mean: {df['Similarity'].mean()}")


def get_smiles_column(df):
    for col in ["smiles", "SMILES", "Smiles"]:
        if col in df.columns:
            return col
    raise ValueError("No column containing SMILES strings found.")


def main():
    # Example usage

    csv_path = "/media/mohammed/Work/Navi_diversity/tests/clusters/scorer_output/clusters/groupby_results_aggregated_clusters.csv"

    df_smiles = pd.read_csv(csv_path)  # .sample(10000, random_state=42)
    print(df_smiles.columns)
    col_mols = "Molecules containing fragment"
    col_num_mols = "Number of Molecules_with_Fragment"
    df_smiles[col_mols] = df_smiles[
        col_mols
    ].apply(
        lambda x: eval(x.replace("{", "[").replace("}", "]"))
        if isinstance(x, str)
        else x
    )
    df_smiles = df_smiles.sort_values(
        by=[col_num_mols], ascending=False
    )
    smiles1 = df_smiles[col_mols].iloc[0]
    smiles2 = [df_smiles["Substructure"].iloc[0]]
    print(smiles1)
    print(smiles2)

    print(f"Using {len(smiles1)} SMILES from the dataset.")

    molecules1 = [
        Chem.MolFromSmiles(smiles)
        for smiles in smiles1
        if Chem.MolFromSmiles(smiles)
    ]
    molecules2 = [
        Chem.MolFromSmiles(smiles)
        for smiles in smiles2
        if Chem.MolFromSmiles(smiles)
    ]
    
    print(f"Using {len(molecules1)} valid molecules from the dataset.")

    output_file = "similarity_histogram.png"
    plot_similarity_histogram(molecules1, molecules2, output_file)


if __name__ == "__main__":
    main()
