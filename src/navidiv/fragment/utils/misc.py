"""Assess the over-representation of patterns in a CSV file of molecules.

Considers a list of smiles as a diverse set of good molecules,
Fragments the molecules in the csv, and generates a csv file of smarts patterns.
Then, it counts the occurrences of each smarts pattern in the molecules and compares with the occurence in the reference set.
if a pattern is over-represented in the csv file to assess as compared to the reference set, we add it to the list of over-represented patterns.
"""

import concurrent.futures
import re
import time
from collections import Counter

import pandas as pd
from omegaconf import DictConfig
from rdkit import Chem
from rdkit.Chem import FilterCatalog

from OPTMOLGEN.fragment_utils.Mac_frag import MacFrag
from OPTMOLGEN.fragment_utils.reduce_rings import fragment_mols_fused_ring


def run_frag_analysis(cfg: DictConfig):
    wd = cfg.wd + "/" + cfg.name
    min_score = cfg.analyse_substructures.min_score
    name = cfg.name
    df_smiles = pd.read_csv(cfg.wd + "/" + name + "/" + name + "_1.csv")
    df_smiles = df_smiles[df_smiles["Score"] > min_score]
    df_ref = pd.read_csv(cfg.analyse_substructures.ref_csv_file)
    if cfg.analyse_substructures.load_patterns:
        df_smarts = pd.read_csv(cfg.wd + "/" + name + "/" + "smarts_frag.csv")
    else:
        df_smarts = find_common_smarts_patterns(
            df_smiles,
            output_path=cfg.wd + "/" + name + "/" + "smarts_frag.csv",
            min_counter=7,
            sample_size=100,
        )
    df_counts_in_data = count_patterns_in_data(df_smiles, df_smarts)
    df_count_in_orginal = count_patterns_in_data(df_ref, df_smarts)
    df_counts_in_data.to_csv(f"{wd}/fragment_counts_in_data.csv", index=False)
    df_count_in_orginal.to_csv(
        f"{wd}/fragment_counts_in_orginal.csv", index=False
    )
    # compare the df counts
    df_counts_in_data = df_counts_in_data.merge(
        df_count_in_orginal,
        on="Fragment",
        how="left",
        suffixes=("", "_in_orginal"),
    )
    # replace NaN with 0
    df_counts_in_data.fillna(0, inplace=True)
    df_counts_in_data["Over_represented"] = (
        df_counts_in_data["Portion Count"]
        > df_counts_in_data["Portion Count_in_orginal"]
    )
    df_counts_in_data["under_represented"] = (
        df_counts_in_data["Portion Count"]
        < df_counts_in_data["Portion Count_in_orginal"]
    )
    df_counts_in_data.to_csv(f"{wd}/fragment_counts_in_data.csv", index=False)
    df_over_represented = df_counts_in_data[
        (df_counts_in_data["Over_represented"])
        & (
            df_counts_in_data["Portion Count"]
            > cfg.analyse_substructures.limit_portion
        )
        & (
            df_counts_in_data["Portion Count_in_orginal"]
            < cfg.analyse_substructures.limit_portion
        )
    ]
    df_over_represented.to_csv(f"{wd}/over_represented.csv", index=False)
    smarts_list = df_over_represented["Fragment"].tolist()
    smarts_list = [
        smart
        for smart in smarts_list
        # if len(smart) > 4 * cfg.analyse_substructures.min_length_fragment
    ]
    smarts_list = filter_smarts(
        smarts_list,
        cfg.analyse_substructures.min_length_fragment,
        cfg.analyse_substructures.min_length_fragment_ring,
    )
    with open(f"{wd}/smarts_list.txt", "w") as f:
        for item in smarts_list:
            f.write("%s\n" % item)
    return df_over_represented


def filter_smarts(smarts_list, min_length_fragment, min_length_fragment_ring):
    mols = [Chem.MolFromSmarts(smart) for smart in smarts_list]
    filtered_smarts = []
    for smart in smarts_list:
        mol = Chem.MolFromSmarts(smart)
        if mol is None:
            continue
        # Sanitize the molecule
        # Sanitize the molecule
        Chem.SanitizeMol(mol)

        # Compute ring information
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        if mol.GetRingInfo().NumRings() > 0:
            if mol.GetNumAtoms() > min_length_fragment_ring:
                filtered_smarts.append(smart)
            continue
        if mol.GetNumAtoms() > min_length_fragment:
            filtered_smarts.append(smart)
    return filtered_smarts


def count_patterns_in_data(df_smiles, df_smarts):
    catalog = FilterCatalog.FilterCatalog()

    for smarts in df_smarts["Fragment"]:
        # create RDKit molecule
        if re.findall(r"\[\d+#0\]", smarts):
            smarts = re.sub(r"\[\d+#0\]", "*", smarts)
        m = Chem.MolFromSmarts(smarts)
        # replace connection with wildcard
        # create SMARTS matcher
        sm = FilterCatalog.SmartsMatcher(m)
        # add entry to FilterCatalog
        entry = FilterCatalog.FilterCatalogEntry(smarts, sm)
        catalog.AddEntry(entry)
    frag_list = []
    df_smiles.dropna(subset=["SMILES"], inplace=True)
    for smiles in df_smiles["SMILES"]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        matches = catalog.GetMatches(mol)
        if len(matches) > 0:
            for match in matches:
                frag_list.append(match.GetDescription())
    frag_counter = Counter(frag_list)
    if len(frag_counter) == 0:
        return pd.DataFrame(columns=["Fragment", "Count ", "Portion Count"])
    df = pd.DataFrame.from_dict(frag_counter, orient="index").reset_index()
    df.columns = ["Fragment", "Count"]
    df["Portion Count"] = df["Count"] / len(df_smiles)
    # df.to_csv("fragment_counts_good_formed.csv", index=False)
    return df


def find_common_smarts_patterns(
    df_smiles,
    smiles_column="SMILES",
    min_counter=1,
    output_path="fragment_counts.csv",
    sample_size=100,
):
    number_of_fragment = 0
    number_of_fragment_old = -1
    fragment_counts_df = pd.DataFrame(columns=["Fragment", "Count", "smiles"])
    while (
        number_of_fragment > number_of_fragment_old
        and number_of_fragment < 100
    ):
        number_of_fragment_old = number_of_fragment
        fragment_counts_df_to_merge = find_common_smarts_patterns_sample(
            df_smiles, smiles_column, min_counter, sample_size=100
        )
        fragment_counts_df = pd.concat(
            [fragment_counts_df, fragment_counts_df_to_merge]
        )
        fragment_counts_df.drop_duplicates(subset=["Fragment"], inplace=True)
        number_of_fragment = len(fragment_counts_df)
        fragment_counts_df.to_csv(output_path, index=False)

    fragment_counts_df["Portion Count"] = fragment_counts_df["Count"] / len(
        df_smiles
    )
    fragment_counts_df = fragment_counts_df[
        fragment_counts_df["Count"] > min_counter
    ]
    return fragment_counts_df


def process_fragment(fragment):
    mol = Chem.MolFromSmarts(fragment)
    if mol is not None and mol.GetNumAtoms() < 40 and mol.GetNumAtoms() > 10:
        return fragment_mols_fused_ring(mol)
    return []


def check_all_atoms_in_rings(mol):
    for atom in mol.GetAtoms():
        if not atom.IsInRing():
            return False
    return True


def delete_non_ring_atoms(mol):
    """Delete non-ring atoms from the molecule"""
    atoms_to_delete = []
    for atom in mol.GetAtoms():
        if not atom.IsInRing():
            atoms_to_delete.append(atom.GetIdx())
    mol_rw = Chem.RWMol(mol)
    mol_rw.BeginBatchEdit()

    for atom in atoms_to_delete:
        mol_rw.RemoveAtom(atom)
    mol_rw.CommitBatchEdit()
    mol = Chem.Mol(mol_rw)
    return mol


def find_common_smarts_patterns_sample(
    df_smiles, smiles_column="SMILES", min_counter=1, sample_size=100
):
    """Find the most common SMARTS patterns in a dataset.

    Args:
        csv_file (str): Path to the CSV file.
        smiles_column (str): Column name containing SMILES strings.
        top_n (int): Number of top common SMARTS patterns to return.

    Returns:
        list: List of tuples containing the most common SMARTS patterns and their counts.
    """
    start_time = time.time()
    # Read the CSV file
    df = df_smiles.sample(sample_size)
    df = df.drop_duplicates(subset=[smiles_column])
    # params for MacFrag
    maxBlocks = 100
    maxSR = 10
    minFragAtoms = 3
    # Convert SMILES to RDKit molecules
    molecules = [Chem.MolFromSmiles(smiles) for smiles in df[smiles_column]]

    # Generate fragments for each molecule
    fragments = []
    fragment_rings = []
    for mol in molecules:
        if mol is None:
            continue
        smarts_list = [
            Chem.MolToSmarts(mol_frag)
            for mol_frag in MacFrag(
                mol,
                maxBlocks=maxBlocks,
                maxSR=maxSR,
                asMols=True,
                minFragAtoms=minFragAtoms,
            )
            # if len(mol_frag.GetRingInfo().AtomRings()) > 1
        ]
        fragment_ring_only = [
            delete_non_ring_atoms(Chem.MolFromSmarts(fragment))
            for fragment in smarts_list
        ]
        smarts_list.extend(
            [Chem.MolToSmarts(fragment) for fragment in fragment_ring_only]
        )
        smart_list_rings = [
            smarts
            for smarts in smarts_list
            if check_all_atoms_in_rings(Chem.MolFromSmarts(smarts))
        ]
        smarts_list_new = []
        for smarts in smarts_list:
            if "." in smarts:
                smarts = smarts.split(".")
                smarts_list_new.extend(smarts)
                continue
            smarts_list_new.append(smarts)
        fragments.extend(smarts_list_new)

        fragment_rings.extend(smart_list_rings)
    time_for_initial_frag = time.time() - start_time
    print(f"Time for initial fragments: {time_for_initial_frag}")
    new_frag = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_fragment, fragment_rings))
    for result in results:
        new_frag.extend(result)
    fragments.extend(
        [
            Chem.MolToSmarts(frag)
            for frag in new_frag
            if frag is not None  # and len(frag.GetRingInfo().AtomRings()) > 1
        ]
    )

    time_for_ring_frag = time.time() - time_for_initial_frag - start_time
    print(f"Time for ring fragments: {time_for_ring_frag} ")
    # Count the frequency of each fragment
    fragment_counts = Counter(fragments)

    # save fragment counts to csv
    fragment_counts_df = pd.DataFrame(
        fragment_counts.items(), columns=["Fragment", "Count"]
    )
    fragment_counts_df["smiles"] = fragment_counts_df["Fragment"].apply(
        lambda x: Chem.MolToSmiles(Chem.MolFromSmarts(x))
    )

    return fragment_counts_df
