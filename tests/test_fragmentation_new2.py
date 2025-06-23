


import logging
import time  

from rdkit import Chem  
from rdkit.Chem import Draw

from navidiv.fragment import (
    fragment_scorer,
)

logging.basicConfig(level=logging.INFO)


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


def test_fragment_scorer():
    """Test the fragment scorer."""
    smiles = "O=C1NC(=O)N(c2ccccc2)C1=C1c2cc(Cl)ccc2C(c2ccccc2Cl)=N1"
    for transformation_mode in [
        "basic_framework",
        "basic_wire_frame",
        "none",
        "elemental_wire_frame",
    ]:
        frag_score = fragment_scorer.FragmentScorer(
            min_count_fragments=1,
            output_path="/media/mohammed/Work/Navi_diversity/tests/resultstests/",
        )
        frag_score.update_transformation_mode(transformation_mode)
        start = time.time()
        fragement = frag_score.get_fragments([smiles])
        fragemnt_dict = {"initial": smiles}
        end = time.time()
        print(f"FragmentScorer.get_fragments time: {end - start:.3f} seconds")
        for frag in fragement:
            num_atoms = Chem.MolFromSmiles(frag).GetNumAtoms()
            fragemnt_dict[f"num_atoms {num_atoms}"] = frag

        plot_scaffolds(
            fragemnt_dict,
            filename=f"tests/fragment_test_{transformation_mode}.png",
        )


if __name__ == "__main__":
    test_fragment_scorer()
    print("Fragment scorer test completed successfully.")
