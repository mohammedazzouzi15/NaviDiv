"""Classical scaffold soring using RDKit.
This module provides functionality to generate scaffolds from smiles strings list
and score them based on their presence in the dataset.

Example usage:
```python
from navidiv.Scaffold_scorer import Scaffold_scorer
from navidiv.utils import plot_scaffolds
from navidiv.utils import plot_scaffolds
from navidiv.utils import get_smiles_column
from rdkit import Chem
scaffold_scorer = Scaffold_scorer(
    output_path="/path/to/output",
    scaffold_type="bajorath",
)
scaffolds = scaffold_scorer.get_scaffolds(smiles_list)
from navidiv.utils import plot_scaffolds
plot_scaffolds(scaffolds, filename="scaffolds.png")
```
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from navidiv.scorer import BaseScore

PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
REPL = Chem.MolFromSmarts("[*]")


def get_scaffold(smile, cases: str = "all"):
    """Get the scaffold of a molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        cases (str): Cases to consider for scaffold generation.
            "bajorath": Use Bajorath's method.
            "real_bm": Use real Bajorath's method.
            "csk": Use CSK method.
            "csk_bm": Use CSK with Bajorath's method.
            "murcko": Use Murcko's method.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)  # important for canonization of CSK!
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        match cases:
            case "bajorath":
                scaff = AllChem.DeleteSubstructs(scaff, PATT)
            case "real_bm":
                scaff = AllChem.ReplaceSubstructs(scaff, PATT, REPL)[0]
            case "csk":
                scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
            case "csk_bm":
                scaff = AllChem.DeleteSubstructs(scaff, PATT)
                scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)

        return scaff
    except Exception as e:
        print(f"Error generating scaffold for {smile}: {e}")
        return None


class Scaffold_scorer(BaseScore):
    """Handles fragment scoring and analysis for molecular datasets."""

    def __init__(
        self,
        output_path: str | None = None,
        scaffold_type: str = "bajorath",
    ) -> None:
        """Initialize FragmentScore.

        Args:
            min_count_fragments (int): Minimum count for fragments to be
                considered.
            output_path (str | None): Path to save output files.
        """
        super().__init__(output_path=output_path)
        self.scaffold_type = scaffold_type
        self._csv_name = f"scaffold_{scaffold_type}"
        self._min_count_fragments = 1
        self._dict_scaffolds = {}

    def get_scaffolds(self, smiles_list: list[str]) -> list[str]:
        """Get scaffolds for a list of SMILES strings.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            list[str]: List of scaffold SMILES strings.
        """
        scaffolds = []
        for smi in smiles_list:
            if smi == "None":
                continue
            grps = self.get_scaffold(smi)
            if grps is None:
                continue
            scaffolds.append(Chem.MolToSmiles(grps, isomericSmiles=True))
        return scaffolds

    def get_count(self, smiles_list: list[str]) -> tuple[pd.DataFrame, None]:
        """Calculate the percentage of each fragment in the dataset.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            tuple: DataFrame with fragment info, None (for compatibility)
        """
        self._scaffolds = self.get_scaffolds(smiles_list)

        fragments, over_represented_fragments = self._from_list_to_count_df(
            smiles_list,
            self._scaffolds,
        )
        self._fragments_df = fragments
        return fragments, over_represented_fragments

    def _count_substructure_in_smiles(self, smiles_list, scaffold):
        """Check if molecule has the same scaffold"""
        if not hasattr(self, "_scaffolds"):
            raise ValueError(
                "Scaffolds have not been calculated. Please run get_count first."
            )
        return len([i for i in self._scaffolds if scaffold == i])

    def _comparison_function(
        self,
        smiles: str | None = None,
        fragment: str | None = None,
        mol=None,
    ) -> bool:
        """Check if the fragment is present in the SMILES string or molecule."""
        if not hasattr(self, "_scaffolds"):
            raise ValueError(
                "Scaffolds have not been calculated. Please run get_count first."
            )
        if smiles is None:
            return False

        if fragment is None:
            return False
        smiles_scaffold = self.get_scaffold(smiles)
        if smiles_scaffold is None:
            return False
        smiles_scaffold = Chem.MolToSmiles(
            smiles_scaffold, isomericSmiles=True
        )
        return smiles_scaffold == fragment

    def get_scaffold(self, smile: str) -> Chem.Mol | None:
        """Get the scaffold of a molecule.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            cases (str): Cases to consider for scaffold generation.
                "bajorath": Use Bajorath's method.
                "real_bm": Use real Bajorath's method.
                "csk": Use CSK method.
                "csk_bm": Use CSK with Bajorath's method.
                "murcko": Use Murcko's method.
        """
        if smile in self._dict_scaffolds:
            return self._dict_scaffolds[smile]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)  # important for canonization of CSK!
        try:
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            match self.scaffold_type:
                case "bajorath":
                    scaff = AllChem.DeleteSubstructs(scaff, PATT)
                case "real_bm":
                    scaff = AllChem.ReplaceSubstructs(scaff, PATT, REPL)[0]
                case "csk":
                    scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
                case "csk_bm":
                    scaff = AllChem.DeleteSubstructs(scaff, PATT)
                    scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
            self._dict_scaffolds[smile] = scaff
            return scaff
        except Exception as e:
            print(f"Error generating scaffold for {smile}: {e}")
            return None
