"""Extract rings, functional groups and scaffolds from molecules dataset"""

from pathlib import Path

import pandas as pd

from  navidiv.diversity.diversity import diversity_all


def get_diversity_metric_from_csv(
    csv_file, output_path_orig, max_size_intdiv=10000
):
    df_orig = pd.read_csv(csv_file).sample(frac=1)
    df_orig = df_orig.dropna(subset=["smiles"])
    df_orig = df_orig.drop_duplicates(subset=["smiles"])
    summary_lists = []
    for stage, df in df_orig.groupby("stage"):
        if df.shape[0] < 10:
            continue
        output_path = output_path_orig + "/" + str(int(stage))
        Path(output_path).mkdir(parents=True, exist_ok=True)
        smiles = df["smiles"].to_list()

        Richness = diversity_all(smiles=smiles, mode="Richness")
        print("Richness", Richness)
        RS_size = diversity_all(
            smiles=smiles, mode="RS_size", path_fo_csv=output_path
        )
        print("RS_size", RS_size)
        FG = diversity_all(smiles=smiles, mode="FG", path_fo_csv=output_path)
        print("FG", FG)
        RS = diversity_all(smiles=smiles, mode="RS", path_fo_csv=output_path)
        print("RS", RS)
        BM = diversity_all(smiles=smiles, mode="BM", path_fo_csv=output_path)
        print("BM", BM)
        if len(smiles) > max_size_intdiv:
            # randomly select 10000 smiles
            import numpy as np

            random_smiles = np.random.choice(
                smiles, max_size_intdiv, replace=False
            )
            IntDiv = diversity_all(
                smiles=random_smiles, mode="IntDiv", path_fo_csv=output_path
            )
        else:
            IntDiv = diversity_all(smiles=smiles, mode="IntDiv")
        print("Int_div", IntDiv)
        summary_list = [stage, Richness, RS_size, FG, RS, BM, IntDiv]
        summary_lists.append(summary_list)

    merged_df_RS_size = merge_csvs(
        [
            output_path_orig + "/" + str(int(stage)) + "/RG_size.csv"
            for stage in df_orig["stage"].unique()
        ]
    )
    merged_df_RS_size.drop_duplicates().to_csv(
        output_path_orig + "/RS_size.csv", index=False
    )
    merged_df_FG = merge_csvs(
        [
            output_path_orig + "/" + str(int(stage)) + "/functional_groups.csv"
            for stage in df_orig["stage"].unique()
        ]
    )
    merged_df_FG.drop_duplicates().to_csv(
        output_path_orig + "/FG.csv", index=False
    )
    merged_df_RS = merge_csvs(
        [
            output_path_orig + "/" + str(int(stage)) + "/ring_systems.csv"
            for stage in df_orig["stage"].unique()
        ]
    )
    merged_df_RS.drop_duplicates().to_csv(
        output_path_orig + "/RS.csv", index=False
    )
    merged_df_BM = merge_csvs(
        [
            output_path_orig + "/" + str(int(stage)) + "/scaffolds.csv"
            for stage in df_orig["stage"].unique()
        ]
    )
    merged_df_BM.drop_duplicates().to_csv(
        output_path_orig + "/BM.csv", index=False
    )
    smiles = df_orig["smiles"].to_list()
    if len(smiles) > max_size_intdiv:
        # randomly select 10000 smiles
        import numpy as np

        random_smiles = np.random.choice(
            smiles, max_size_intdiv, replace=False
        )
        intdiv_total = diversity_all(
            smiles=random_smiles, mode="IntDiv", path_fo_csv=output_path
        )
    else:
        intdiv_total = diversity_all(smiles=smiles, mode="IntDiv")

    summary_lists.append(
        [
            "Total",
            len(df_orig["smiles"].unique()),
            merged_df_RS_size.shape[0],
            merged_df_FG.shape[0],
            merged_df_RS.shape[0],
            merged_df_BM.shape[0],
            intdiv_total,
        ]
    )
    summary_df = pd.DataFrame(
        summary_lists,
        columns=["stage", "Richness", "RS_size", "FG", "RS", "BM", "IntDiv"],
    )
    summary_df.to_csv(output_path_orig + "/summary.csv", index=False)


def merge_csvs(csv_files):
    dfs = []
    for csv_file in csv_files:
        if not Path(csv_file).exists():
            print(f"File {csv_file} does not exist")
            continue
        df = pd.read_csv(csv_file)
        dfs.append(df)
    merged_df = pd.concat(dfs)
    return merged_df
