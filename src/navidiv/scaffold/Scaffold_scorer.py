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

from collections import Counter

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


class ScaffoldModeOption:
    SCAFFOLD = "scaffold"
    MURCKO_FRAMEWORK = "murcko_framework"
    BASIC_WIRE_FRAME = "basic_wire_frame"
    ELEMENTAL_WIRE_FRAME = "elemental_wire_frame"
    BASIC_FRAMEWORK = "basic_framework"


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
        self.scaffold_mode_setting = scaffold_type
        self._csv_name = f"scaffold_{scaffold_type}"
        self._min_count_fragments = 0
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
        self._scaffolds = scaffolds
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
            smile (str): Input SMILES string.

        Returns:
            Chem.Mol | None: RDKit Mol object of the scaffold or None if
            the SMILES is invalid.
        """
        if smile in self._dict_scaffolds:
            return self._dict_scaffolds[smile]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)  # important for canonization of CSK!
        if self.scaffold_mode_setting == ScaffoldModeOption.MURCKO_FRAMEWORK:
            return MurckoScaffold.GetScaffoldForMol(mol)
        if self.scaffold_mode_setting == ScaffoldModeOption.SCAFFOLD:
            # For demonstration, use Murcko as base, but keep non-single bonded atoms attached to rings/linkers
            return MurckoScaffold.GetScaffoldForMol(mol)
        if self.scaffold_mode_setting == ScaffoldModeOption.BASIC_WIRE_FRAME:
            # Anonymize atoms and set all bonds to single
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                return MurckoScaffold.MakeScaffoldGeneric(scaffold)
            except Exception as e:
                print(f"Error generating scaffold for {smile}: {e}")
                return scaffold
        if (
            self.scaffold_mode_setting
            == ScaffoldModeOption.ELEMENTAL_WIRE_FRAME
        ):
            # Set all bonds to single, keep atom types
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            rw_scaffold = Chem.RWMol(scaffold)  # Make editable
            for bond in rw_scaffold.GetBonds():
                bond.SetBondType(Chem.BondType.SINGLE)
            scaffold_single = rw_scaffold.GetMol()
            Chem.SanitizeMol(scaffold_single)
            return scaffold_single
        if self.scaffold_mode_setting == ScaffoldModeOption.BASIC_FRAMEWORK:
            # Set all atoms to carbon, keep bond orders
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            for atom in scaffold.GetAtoms():
                atom.SetAtomicNum(6)
            return scaffold
        raise ValueError("Unknown scaffold mode")

    def additional_metrics(self):
        """Calculate additional metrics for the scorer."""
        count_scaffolds = Counter(self._scaffolds)
        ngrams_above_10 = {
            ngram: count
            for ngram, count in count_scaffolds.items()
            if count > 10
        }
        ngrams_above_5 = {
            ngram: count
            for ngram, count in count_scaffolds.items()
            if count > 5
        }
        ngrams_below_2 = {
            ngram: count
            for ngram, count in count_scaffolds.items()
            if count < 2
        }

        return {
            "Appeared more than 10 times": len(ngrams_above_10),
            "Appeared more than 5 times": len(ngrams_above_5),
            "Appeared once": len(ngrams_below_2),
        }
