
import itertools
import random
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import rdFMCS


def check_similarity(mol1, mol2) -> float:
    """Check the similarity using Tanimoto similarity."""
    try:
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)
    except (AttributeError, ValueError):
        logging.exception("[check_similarity] Error calculating similarity")
        return 0.0


def check_smarts(smart: str) -> bool:
    """Check if the SMILES string is valid."""
    try:
        Chem.Kekulize(
            Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmarts(smart)))
        )
    except (ValueError, RuntimeError):
        logging.exception("[check_smarts] Invalid SMARTS")
        return False
    else:
        return True


class CompareElementsOutsideRings(rdFMCS.MCSAtomCompare):
    """Custom atom comparison for MCS that compares elements outside rings."""
    def __call__(self, p, mol1, atom1, mol2, atom2) -> bool:
        """Compare two atoms for MCS, considering ring membership and chirality."""
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        if (a1.GetAtomicNum() != a2.GetAtomicNum()) and not (
            a1.IsInRing() and a2.IsInRing()
        ):
            return False
        if p.MatchChiralTag and not self.CheckAtomChirality(
            p, mol1, atom1, mol2, atom2
        ):
            return False
        if p.RingMatchesRingOnly:
            return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
        return True


def clean_substrucuture_list(substructure_list: set, config) -> set:
    """Remove redundant or invalid substructures from the set."""
    substructures_to_remove = set()
    for smart1, smart2 in itertools.combinations(substructure_list, 2):
        try:
            mol1 = Chem.MolFromSmiles(
                Chem.MolToSmiles(Chem.MolFromSmarts(smart1))
            )
            Chem.Kekulize(mol1)
        except (ValueError, RuntimeError):
            logging.exception(
                "[clean_substrucuture_list] Invalid substructure: %s", smart1
            )
            substructures_to_remove.add(smart1)
            continue
        try:
            mol2 = Chem.MolFromSmiles(
                Chem.MolToSmiles(Chem.MolFromSmarts(smart2))
            )
            Chem.Kekulize(mol2)
        except (ValueError, RuntimeError):
            logging.exception(
                "[clean_substrucuture_list] Invalid substructure: %s", smart2
            )
            substructures_to_remove.add(smart2)
            continue
        try:
            if check_similarity(mol1, mol2) > config.similarity_threshold:
                substructures_to_remove.add(smart2)
        except (AttributeError, ValueError):
            logging.exception(
                "[clean_substrucuture_list] Error comparing %s and %s",
                smart1, smart2
            )
            continue
    substructure_list -= substructures_to_remove
    return substructure_list


def find_most_common_substructure(config):
    """Find the most common substructure in the dataset."""
    # Load the CSV file
    wd = config.wd + "/" + config.name
    min_score = config.analyse_substructures.min_score
    name = config.name
    data = pd.read_csv(config.wd + "/" + name + "/" + name + "_1.csv")
    data = data[data["Score"] > min_score]
    data = data.drop_duplicates(subset=["SMILES"])
    data = data.reset_index(drop=True)
    logging.info("Number of molecules: %d", len(data))
    params = rdFMCS.MCSParameters()
    params.AtomTyper = CompareElementsOutsideRings()
    params.BondTyper = rdFMCS.BondCompare.CompareOrder
    params.BondCompareParameters.RingMatchesRingOnly = (
        config.get_most_common_substructure.ring_matches_ring_only
    )
    params.BondCompareParameters.CompleteRingsOnly = (
        config.get_most_common_substructure.complete_rings_only
    )

    # Extract SMILES strings and convert to RDKit molecules
    smiles_column = None
    for col in ["SMILES", "smiles", "Smiles"]:
        if col in data.columns:
            smiles_column = col
            break
    if not smiles_column:
        raise ValueError("No valid SMILES column found in the dataset.")

    smiles_list = data[smiles_column].dropna().tolist()
    molecules = [
        Chem.MolFromSmiles(smiles)
        for smiles in smiles_list
        if Chem.MolFromSmiles(smiles) is not None
    ]

    if not molecules:
        logging.warning("No valid molecules found in the dataset.")
        return None

    substructures = set()
    converged = False

    while not converged:
        initial_size = len(substructures)
        for _ in range(
            min(config.get_most_common_substructure.iterations, len(molecules))
        ):
            sampled_molecules = random.sample(
                molecules,
                min(
                    config.get_most_common_substructure.sample_size,
                    len(molecules),
                ),
            )
            mcs_result = rdFMCS.FindMCS(sampled_molecules, params)
            mcs_mol = mcs_result.queryMol
            if (
                mcs_mol is None
                or mcs_mol.GetNumAtoms()
                < config.get_most_common_substructure.min_atoms
            ):
                continue
            logging.info("Size of substructures: %d", len(substructures))
            substructures.add(mcs_result.smartsString)
            if (
                len(substructures)
                > config.get_most_common_substructure.max_substructures
            ):
                break

        if (
            initial_size == len(substructures)
            or len(substructures)
            > config.get_most_common_substructure.max_substructures
        ):
            converged = True

    # substructures = clean_substrucuture_list(substructures)
    df_substructures = pd.DataFrame(
        {
            "Substructure": list(substructures),
            "Count": [0] * len(substructures),
        }
    )

    for smarts in substructures:
        count = sum(
            1
            for mol in molecules
            if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))
        )
        average_score = (
            sum(
                data.loc[
                    data[smiles_column].isin(
                        [
                            Chem.MolToSmiles(mol)
                            for mol in molecules
                            if mol.HasSubstructMatch(
                                Chem.MolFromSmarts(smarts)
                            )
                        ]
                    ),
                    "Score",
                ]
            )
            / count
        )
        df_substructures.loc[
            df_substructures["Substructure"] == smarts, "Count"
        ] = count
        df_substructures.loc[
            df_substructures["Substructure"] == smarts, "Average_Score"
        ] = average_score

    df_substructures["count_perc"] = (
        df_substructures["Count"] / len(molecules)
    ) * 100
    df_substructures["smiles"] = [
        Chem.MolToSmiles(Chem.MolFromSmarts(smart))
        for smart in df_substructures["Substructure"]
    ]
    df_substructures["num_atoms"] = [
        Chem.MolFromSmarts(smart).GetNumAtoms()
        for smart in df_substructures["Substructure"]
    ]
    df_substructures["num_rings"] = [
        Chem.MolFromSmarts(smart).GetRingInfo().NumRings()
        for smart in df_substructures["Substructure"]
    ]
    logging.info("Substructure counts:")
    logging.info("%s", df_substructures)
    df_substructures["stage_name"] = name
    df_substructures["score"] = min_score
    df_substructures.to_csv(f"{wd}/df_substructures.csv", index=False)
    smarts_list = df_substructures[
        df_substructures["Count"]
        > config.get_most_common_substructure.count_threshold
    ]["Substructure"].tolist()

    with Path(f"{wd}/smarts_list.txt").open("w") as f:
        for item in smarts_list:
            f.write("%s\n" % item)
    return df_substructures


if __name__ == "__main__":
    config = OmegaConf.load(
        "/media/mohammed/Work/OPT_MOL_GEN/configs/find_common_substructure.yaml"
    )
    df_substructures = find_most_common_substructure(config)
    df_substructures.to_csv(config.output_file)
