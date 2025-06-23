from typing import List
import chemprop
import numpy as np
from reinvent.scoring.utils import suppress_output
import pandas as pd
import excit_obj_func
# from .excit_obj_func import energy_score


def load_chemprop_models(checkpoint_dirs: List[str], rdkit_2d_normalized: List[bool]):
    """
    Loads ChemProp models from the specified checkpoint directories.

    Args:
        checkpoint_dirs (List[str]): List of paths to ChemProp model checkpoint directories.
        rdkit_2d_normalized (List[bool]): List of booleans indicating whether to use RDKit 2D normalized features.

    Returns:
        List[Tuple]: A list of tuples containing the loaded models and their arguments.
    """
    chemprop_params = []

    for checkpoint_dir, rdkit_2d_norm in zip(checkpoint_dirs, rdkit_2d_normalized):
        args = [
            "--checkpoint_dir",
            checkpoint_dir,
            "--test_path",
            "/dev/null",
            "--preds_path",
            "/dev/null",
        ]

        if rdkit_2d_norm:
            args.extend(
                ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"]
            )

        with suppress_output():
            chemprop_args = chemprop.args.PredictArgs().parse_args(args)
            chemprop_model = chemprop.train.load_model(args=chemprop_args)
            chemprop_params.append((chemprop_model, chemprop_args))

    return chemprop_params


def predict_with_chemprop(models: List[tuple], smiles: List[str]) -> np.array:
    """
    Uses loaded ChemProp models to make predictions on a list of SMILES strings.

    Args:
        models (List[tuple]): List of tuples containing ChemProp models and their arguments.
        smiles (List[str]): List of SMILES strings to predict.

    Returns:
        np.array: Array of predictions for the input SMILES strings.
    """
    smiles_list = [[s] for s in smiles]
    scores = []

    for model, args in models:
        with suppress_output():
            preds = chemprop.train.make_predictions(
                model_objects=model,
                smiles=smiles_list,
                args=args,
                return_invalid_smiles=True,
                return_uncertainty=False,
            )
        scores.append(preds)

    return np.array(scores)


def read_txt_file(file_path):
    """
    Reads a text file and returns its content as a list of lines.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: List of lines from the text file.
    """
    df = pd.read_csv(file_path)
    return df

def get_smiles_column(df):
    """
    Returns the name of the column containing SMILES strings in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to check.

    Returns:
        str: Name of the column containing SMILES strings.
    """
    if "SMILES" in df.columns:
        return "SMILES"
    elif "smiles" in df.columns:
        return "smiles"
    else:
        raise ValueError("No SMILES column found in the DataFrame.")

# Example usage:
# checkpoint_dirs = ["path/to/checkpoint1", "path/to/checkpoint2"]
# rdkit_2d_normalized = [False, True]
# smiles = ["CCO", "CCN", "CCC"]
# models = load_chemprop_models(checkpoint_dirs, rdkit_2d_normalized)
# predictions = predict_with_chemprop(models, smiles)
# print(predictions)
if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--df_path",
        type=str,
        help="path of to the txt file.",
    )
    # Example usage
    checkpoint_dirs = [
        "/media/mohammed/Work/SF_generative/model/formed_chemprop/fold_0/model_0/",
        "/media/mohammed/Work/SF_generative/model/formed_t1t2/fold_0/model_0/",
    ]
    df_path = argparser.parse_args().df_path
    # df_path = "/media/mohammed/Work/SF_generative/reinvent_runs/150425_first_2_/stage0/150425_SF_2_DAP20.txt"
    df = read_txt_file(df_path)
    smiles_column = get_smiles_column(df)
    smiles = df[smiles_column].tolist()
    rdkit_2d_normalized = [True, True]
    # smiles = ["CCO", "CCN", "CCC"]
    models = []
    for checkpoint_dir in checkpoint_dirs:
        models.extend(load_chemprop_models([checkpoint_dir], rdkit_2d_normalized))
    predictions = predict_with_chemprop(models, smiles)
    predictions = np.concatenate(predictions, axis=1)
    df_predictsions = pd.DataFrame(predictions, columns=["S1", "T1", "T1_2", "T2"])
    df_predictsions["smiles"] = smiles
    #df_predictsions["NLL"] = df["NLL"].tolist()
    df_predictsions["T1"] = pd.to_numeric(df_predictsions["T1"], errors="coerce")
    df_predictsions["T2"] = pd.to_numeric(df_predictsions["T2"], errors="coerce")
    df_predictsions["S1"] = pd.to_numeric(df_predictsions["S1"], errors="coerce")
    df_predictsions["T1_2"] = pd.to_numeric(df_predictsions["T1_2"], errors="coerce")
    df_predictsions.dropna(inplace=True)
    df_predictsions["energy_score"] = df_predictsions.apply(
        lambda row: excit_obj_func.energy_score(row["T1"], row["S1"]), axis=1
    )

    df_predictsions["ES1T1"] = df_predictsions["S1"] - 2 * df_predictsions["T1"]
    df_predictsions["ET2T1"] = df_predictsions["T2"] - 2 * df_predictsions["T1"]
    
    if "txt" in df_path:
        df_predictsions.to_csv(df_path.replace(".txt", "_predictions.csv"), index=False)
    if "csv" in df_path:
        df_predictsions.to_csv(df_path.replace(".csv", "_predictions.csv"), index=False)
