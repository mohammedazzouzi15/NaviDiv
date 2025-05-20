import sys

import numpy as np
import pandas as pd
import six
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.modules["sklearn.externals.six"] = six
import networkx as nx
from navidiv.diversity.utils import (
    GetRingSystems,
    get_size_of_fused_ring,
    identify_functional_groups,
)
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from tqdm import tqdm


def dist_array(smiles=None, mols=None):
    if mols == None:
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mols = [x for x in mols if x is not None]
        l = len(mols)
    else:
        l = len(mols)
    """
    You can replace the Tanimoto distances of ECFPs with other molecular distance metrics!
    """
    sims = np.zeros((l, l))
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=6, fpSize=4096
    )
    fps = [
        morgan_gen.GetFingerprint(x) if x is not None else None for x in mols
    ]
    fps = [fp for fp in fps if fp is not None]
    for i in tqdm(range(l), disable=(l < 2000)):
        sims[i, i] = 1
        for j in range(i + 1, l):
            sims[i, j] = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            sims[j, i] = sims[i, j]
    dists = 1 - sims
    return dists


def diversity_all(
    smiles=None,
    mols=None,
    dists=None,
    mode="HamDiv",
    args=None,
    disable=False,
    path_fo_csv="",
):
    if mode == "Richness":
        if smiles != None:
            return len(set(smiles))
        smiles = set()
        for mol in mols:
            smiles.add(Chem.MolToSmiles(mol))
        return len(smiles)
    if mode == "FG":
        func_groups = set()
        for i in range(len(mols) if mols is not None else len(smiles)):
            smi = Chem.MolToSmiles(mols[i]) if smiles is None else smiles[i]
            grps = identify_functional_groups(smi)
            if grps is None:
                continue
            func_groups.update(grps)
        df_func_groups = pd.DataFrame(
            func_groups, columns=["Functional Group"]
        )
        if path_fo_csv != "":
            df_func_groups.to_csv(
                f"{path_fo_csv}/functional_groups.csv", index=False
            )
            return len(func_groups)
        return func_groups

    if mode == "RS_size":
        ring_sys = set()
        for i in range(len(mols) if mols is not None else len(smiles)):
            smi = Chem.MolToSmiles(mols[i]) if smiles is None else smiles[i]
            grps = get_size_of_fused_ring(smi)
            if grps is None:
                continue
            ring_sys.add(grps)
        df_ring_sys = pd.DataFrame(ring_sys, columns=["Ring System"])
        if path_fo_csv != "":
            df_ring_sys.to_csv(f"{path_fo_csv}/RG_size.csv", index=False)
        return len(ring_sys)
    if mode == "RS":
        ring_sys = set()
        for i in range(len(mols) if mols is not None else len(smiles)):
            smi = Chem.MolToSmiles(mols[i]) if smiles is None else smiles[i]
            grps = GetRingSystems(smi)
            if grps is None:
                continue
            ring_sys.update(grps)
        df_ring_sys = pd.DataFrame(ring_sys, columns=["Ring System"])
        if path_fo_csv != "":
            df_ring_sys.to_csv(f"{path_fo_csv}/ring_systems.csv", index=False)
            return len(ring_sys)
        return ring_sys

    if mode == "BM":
        scaffolds = set()
        for i in range(len(mols) if mols is not None else len(smiles)):
            if mols is not None:
                scaf = MurckoScaffold.GetScaffoldForMol(mols[i])
            else:
                mol = Chem.MolFromSmiles(smiles[i])
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if scaf is None:
                continue
            scaf_smi = Chem.MolToSmiles(scaf)
            scaffolds.update([scaf_smi])
        df_scaffolds = pd.DataFrame(scaffolds, columns=["Scaffold"])
        if path_fo_csv != "":
            df_scaffolds.to_csv(f"{path_fo_csv}/scaffolds.csv", index=False)
            return len(scaffolds)
        return scaffolds

    if type(dists) is np.ndarray:
        l = len(dists)
    elif mols == None:
        l = len(smiles)
        assert l >= 2
        dists = dist_array(smiles)
    else:
        l = len(mols)
        assert l >= 2
        dists = dist_array(smiles, mols)

    if mode == "IntDiv":
        if l == 1:
            return 0
        return np.sum(dists) / l / (l - 1)
    if mode == "SumDiv":
        if l == 1:
            return 0
        return np.sum(dists) / (l - 1)
    if mode == "Diam":
        if l == 1:
            return 0
        d_max = 0
        for i in range(l):
            for j in range(i + 1, l):
                d_max = max(d_max, dists[i, j])
        return d_max
    if mode == "SumDiam":
        if l == 1:
            return 0
        sum_d_max = 0
        for i in range(l):
            d_max_i = 0
            for j in range(l):
                if j != i and d_max_i < dists[i, j]:
                    d_max_i = dists[i, j]
            sum_d_max += d_max_i
        return sum_d_max
    if mode == "Bot":
        if l == 1:
            return 0
        d_min = 1
        for i in range(l):
            for j in range(i + 1, l):
                d_min = min(d_min, dists[i, j])
        return d_min
    if mode == "SumBot":
        if l == 1:
            return 0
        sum_d_min = 0
        for i in range(l):
            d_min_i = 1
            for j in range(l):
                if j != i and d_min_i > dists[i, j]:
                    d_min_i = dists[i, j]
            sum_d_min += d_min_i
        return sum_d_min
    if mode == "DPP":
        return np.linalg.det(1 - dists)
    if mode.split("-")[0] == "NCircles":
        threshold = float(mode.split("-")[1])
        circs_sum = []
        for k in tqdm(range(1), disable=disable):
            circs = np.zeros(l)
            rs = np.arange(l)
            # random.shuffle(rs)
            for i in rs:
                circs_i = 1
                for j in range(l):
                    if j != i and circs[j] == 1 and dists[i, j] <= threshold:
                        circs_i = 0
                        break
                circs[i] = circs_i
            circs_sum.append(np.sum(circs))
        return np.max(np.array(circs_sum))
    if mode == "HamDiv":
        total = HamDiv(dists=dists)
        return total

    raise Exception("Undefined mode")


def HamDiv(smiles=None, mols=None, dists=None, method="greedy_tsp"):
    l = (
        dists.shape[0]
        if dists is not None
        else len(mols)
        if mols is not None
        else len(smiles)
    )
    if l == 1:
        return 0
    dists = dist_array(smiles) if dists is None else dists

    remove = np.zeros(l)
    for i in range(l):
        for j in range(i + 1, l):
            if dists[i, j] == 0:
                remove[i] = 1
    remove = np.argwhere(remove == 1)
    dists = np.delete(dists, remove, axis=0)
    dists = np.delete(dists, remove, axis=1)

    G = nx.from_numpy_array(dists)

    if method == "exact_dp":
        tsp, total = solve_tsp_dynamic_programming(dists)
    elif method == "christofides":
        tsp = nx.approximation.christofides(G, weight="weight")
    elif method == "greedy_tsp":
        tsp = nx.approximation.greedy_tsp(G, weight="weight")
    elif method == "simulated_annealing_tsp":
        tsp = nx.approximation.simulated_annealing_tsp(
            G, init_cycle="greedy", weight="weight"
        )
    elif method == "threshold_accepting_tsp":
        tsp = nx.approximation.threshold_accepting_tsp(
            G, init_cycle="greedy", weight="weight"
        )
    elif method == "local_search":
        tsp, total = solve_tsp_local_search(dists, max_processing_time=300)
    else:
        Exception("Undefined method")

    if method not in ["exact_dp", "local_search"]:
        total = 0
        for i in range(1, len(tsp)):
            total += dists[tsp[i - 1], tsp[i]]

    return total
