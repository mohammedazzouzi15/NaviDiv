# Test cases for the scaffold functions in navidiv/Scaffold_scorer.py

from rdkit import Chem
from rdkit.Chem import Draw

from navidiv.scaffold import (
    Scaffold_GNN,
    Scaffold_scorer,
)
from navidiv.scaffold.Scaffold_scorer import get_scaffold


def plot_scaffolds(scaffolds_dict, filename="scaffolds.png"):
    """Plot the generated scaffolds and save to a file.

    Args:
        scaffolds_dict (dict): Dictionary of {case: scaffold_smiles}.
        filename (str): Output image filename.
    """
    mols = []
    legends = []
    for case, smi in scaffolds_dict.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            legends.append(case)
    if mols:
        img = Draw.MolsToGridImage(
            mols, legends=legends, molsPerRow=3, subImgSize=(250, 250)
        )
        img.save(filename)
        print(f"Scaffolds plot saved to {filename}")


def test_get_scaffold_gnn_scorer():
    """Test the Scaffold GNN scorer."""
    scaffold_gnn_score = Scaffold_GNN.ScaffoldGNNScorer(
        output_path="/media/mohammed/Work/Navi_diversity/tests/resultstests/",
    )
    scaffold_gnn_score._min_count_fragments = 0
    smiles = "O=C1NC(=O)N(c2ccccc2)C1=C1c2cc(Cl)ccc2C(c2ccccc2Cl)=N1"
    scaffolds_dict = {
        "initial": smiles,
    }
    for threshold in [0.8, 0.85, 0.9, 0.95, 0.99]:
        scaffold_gnn_score._dict_scaffolds = {}
        scaffold_gnn_score.threshold = threshold
        scaff = scaffold_gnn_score.get_scaffold(smiles)
        case = f"threshold_{scaffold_gnn_score.threshold}"
        assert scaff is not None
        assert isinstance(scaff, Chem.Mol), f"Failed for case: {case}"
        scaffolds_dict[case] = Chem.MolToSmiles(scaff, isomericSmiles=True)
    plot_scaffolds(scaffolds_dict, filename="tests/scaffolds_test_GNN.png")


def test_get_scaffold_gnn_scorer_target():
    """Test the Scaffold GNN scorer."""
    scaffold_gnn_score = Scaffold_GNN.ScaffoldGNNScorer(
        output_path="/media/mohammed/Work/Navi_diversity/tests/resultstests/",
    )
    scaffold_gnn_score._min_count_fragments = 0
    scaffold_gnn_score.threshold = 0.9
    smiles = "O=C1NC(=O)N(c2ccccc2)C1=C1c2cc(Cl)ccc2C(c2ccccc2Cl)=N1"
    scaffolds_dict = {}
    while True:
        scaffold_gnn_score._dict_scaffolds = {}
        scaff = scaffold_gnn_score.get_scaffold(smiles)
        assert scaff is not None

        score = scaffold_gnn_score._dict_targets.get(smiles, None)
        if score is not None:
            case = r"E_{S1}" + f": {score[0]:.2f}"
        else:
            case = "no_target"
        scaffolds_dict[case] = smiles
        if Chem.MolToSmiles(scaff, isomericSmiles=True) == smiles:
            break
        smiles = Chem.MolToSmiles(scaff, isomericSmiles=True)
    plot_scaffolds(scaffolds_dict, filename="tests/scaffolds_test_GNN_rep.png")


def test_get_scaffold_returns_murcko_scaffold():
    # Simple benzene ring

    smiles = "O=C1NC(=O)N(c2ccccc2)C1=C1c2cc(Cl)ccc2C(c2ccccc2Cl)=N1"
    cases = [
        "scaffold",
        "murcko_framework",
        "basic_wire_frame",
        "elemental_wire_frame",
        "basic_framework",
    ]
    scaffolds_dict = {
        "initial": smiles,
    }
    for case in cases:
        scaffold_score = Scaffold_scorer.Scaffold_scorer(
            output_path="/media/mohammed/Work/Navi_diversity/tests/resultstests/",
            scaffold_type=case,
        )
        scaff = scaffold_score.get_scaffold(smiles)
        assert scaff is not None
        assert isinstance(scaff, Chem.Mol), f"Failed for case: {case}"
        scaffolds_dict[case] = Chem.MolToSmiles(scaff, isomericSmiles=True)
    plot_scaffolds(scaffolds_dict, filename="tests/scaffolds_test.png")


def test_get_scaffold_invalid_smiles_returns_none():
    scaff = get_scaffold("not_a_smiles", cases="murcko")
    assert scaff is None


if __name__ == "__main__":
    # test_get_scaffold_invalid_smiles_returns_none()
    test_get_scaffold_returns_murcko_scaffold()
    # test_get_scaffold_gnn_scorer()
    test_get_scaffold_gnn_scorer_target()
    print("All tests passed for scaffold functions.")
