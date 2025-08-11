
from hydra import (
    compose,
    initialize_config_dir,
)
from hydra.core.global_hydra import GlobalHydra
from FORMED_PROP.trainers.Trainer3d import Trainer3d
from FORMED_PROP.data.Dataloader3D import Dataloader3D
import numpy as np
class ModelLoader:
    def __init__(self):
        self.data_loader = None
        self.model = None
        self.trainer = None

    def load_model(self,checkpoint_path):
        # Load the graph neural network model from the specified path
        # This is a placeholder for the actual model loading logic
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base=None,
            config_dir="/media/mohammed/Work/FORMED_PROP/configuration_files/conf",
        ):
            cfg = compose(
                config_name="config_pain_embed",
            )
        self.model = "Loaded model from {}".format(checkpoint_path)
        trainer = Trainer3d(cfg)
        trainer.load_model()
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)
        self.data_loader = Dataloader3D(cfg)
        self.trainer = trainer

    def get_embeddings(self, smiles):
        # Generate embeddings for the given molecular structures
        # This is a placeholder for the actual embedding generation logic
        mol_loader = self.data_loader.get_data_loader_from_smiles([smiles])
        per_atom_energy = None
        target = None
        for batch in mol_loader:
            for i in range(batch.natoms.shape[0]):
                # batch to batch id with size 1
                batch[i].batch = [0]
                per_atom_energy = self.trainer.model.get_node_embedding(batch[i])
                per_atom_energy = np.abs(per_atom_energy.cpu().detach().numpy())
            target = self.trainer.model(batch)["energy"].cpu().detach().numpy()
        return per_atom_energy,target