"""GNN Scaffold Scorer for molecular datasets.
This module implements a scaffold scorer using a GNN model to analyze molecular structures.
The scorer identifies scaffolds based on molecular embeddings and applies a threshold to filter fragments.
this way we can identify the scaffold of a molecule and use it to score the molecule.
the Scaffold generated here should be more adapted to the molecular properties
than the classical Murcko scaffold.
"""

from rdkit import Chem

from navidiv.scaffold.Scaffold_scorer import Scaffold_scorer
from navidiv.scaffold.utils import (
    ModelLoader,
    annotate_molecule,
    get_scaffold,
)


class Params:
    """Parameters for ScaffoldGNNScorer."""

    def __init__(self, checkpoint: str, threshold: float) -> None:
        """Initialize Params for ScaffoldGNNScorer.

        Args:
            checkpoint (str): Path to the model checkpoint.
            threshold (float): Threshold for fragment scoring.
        """
        self.checkpoint = checkpoint
        self.threshold = threshold


class ScaffoldGNNScorer(Scaffold_scorer):
    """Handles fragment scoring and analysis for molecular datasets."""

    def __init__(
        self,
        params: Params | None = None,
        output_path: str | None = None,
    ) -> None:
        """Initialize FragmentScore.

        Args:
            params (Params | None): Parameters for the scorer, including
                checkpoint and threshold.
            output_path (str | None): Path to save output files.
        """
        scaffold_type = "GNN"
        super().__init__(
            output_path=output_path,
            scaffold_type=scaffold_type,
        )
        self._dict_scaffolds = {}
        self._dict_targets = {}

        self.number_of_endpoints = 1
        self.smiles_type = "rdkit_smiles"
        if params is None:
            params = Params(
                checkpoint="/media/mohammed/Work/FORMED_PROP/training_with_charge_embedding/training_with_charge_embedding/training/random/S1_exc/PainnEmbed/epoch=20-val_mae=0.15-val_mse=0.00.ckpt",
                threshold=0.8,
            )
        self.checkpoints = params.checkpoint
        self.model_loader = ModelLoader()
        self.model_loader.load_model(self.checkpoints)
        self.threshold = params.threshold

    def get_scaffold(self, smiles: str) -> Chem.Mol | None:
        """Get the scaffold for a given SMILES string"""
        if smiles in self._dict_scaffolds:
            return self._dict_scaffolds[smiles]
        embeddings, target = self.model_loader.get_embeddings(smiles)
        molecule = Chem.MolFromSmiles(smiles)
        # Normalize embeddings for color mapping
        if embeddings is None or len(embeddings) == 0:
            return None
        molecule = annotate_molecule(molecule, embeddings[0])
        # print(f"ScaffoldGNNScorer.get_scaffold: {smiles}")
        self._dict_scaffolds[smiles] = get_scaffold(molecule, self.threshold)
        self._dict_targets[smiles] = target
        return self._dict_scaffolds[smiles]

    def get_scaffolds(self, smiles_list: list[str]) -> list[str]:
        """Get scaffolds for a list of SMILES strings.

        Args:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            list[str]: List of scaffold SMILES strings.
        """
        smiles_list_to_process = [
            smiles
            for smiles in smiles_list
            if smiles not in self._dict_scaffolds
            and Chem.MolFromSmiles(smiles) is not None
        ]
        embeddings_list, target = self.model_loader.get_embeddings(
            smiles_list_to_process
        )
        for embeddings, smiles in zip(
            embeddings_list, smiles_list_to_process, strict=False
        ):
            molecule = Chem.MolFromSmiles(smiles)
            # Normalize embeddings for color mapping
            if embeddings is None:
                return None
            try:
                molecule = annotate_molecule(molecule, embeddings)
            except Exception as e:
                print(f"Error annotating molecule for {smiles}: {e}")
                self._dict_scaffolds[smiles] = None
                continue
            # molecule = annotate_molecule(molecule, embeddings)
            self._dict_scaffolds[smiles] = get_scaffold(
                molecule, self.threshold
            )
        return [
            Chem.MolToSmiles(self._dict_scaffolds[smiles], isomericSmiles=True)
            for smiles in smiles_list
            if smiles in self._dict_scaffolds
            and self._dict_scaffolds[smiles] is not None
        ]
