
from rdkit import Chem

from navidiv.diversity.utils import (
    identify_functional_groups,
)
from navidiv.fragment.fragment_scorer import FragmentScorer


class FGScorer(FragmentScorer):
    """Handles fragment scoring and analysis for molecular datasets."""

    def __init__(
        self,
        output_path: str | None = None,
    ) -> None:
        """Initialize FragmentScore.

        Args:
            min_count_fragments (int): Minimum count for fragments to be
                considered.
            output_path (str | None): Path to save output files.
        """
        super().__init__(output_path=output_path)
        self._csv_name = "functional_groups"

    def _get_fragment(self, mol: Chem.Mol) -> list[str]:
        """Extract functional groups from a molecule.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            list[str]: List of functional group SMILES.
        """
        return identify_functional_groups(mol)
