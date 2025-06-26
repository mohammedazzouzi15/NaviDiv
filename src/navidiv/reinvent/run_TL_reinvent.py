"""This script performs transfer learning using the REINVENT framework."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from reinvent.utils import config_parse

from navidiv.reinvent.run_transfer_learning2 import run_transfer_learning

LOGLEVEL_CHOICES = tuple(
    level.lower() for level in logging.getLevelNamesMapping()
)

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="/media/mohammed/Work/Navi_diversity/reinvent_runs/conf_folder",
    config_name="transfer_learning",
    version_base="1.1",  # Explicitly specify the compatibility version
)
def main(cfg: DictConfig) -> None:
    """Main function to execute transfer learning using Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "model").mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(cfg.data_file)
    train, validation = split_data(data, cfg.n_head, cfg.n_tail)

    # Save data
    save_data(train, validation, cfg.train_filename, cfg.validation_filename)

    # Create transfer learning parameters
    parameters = create_tl_parameters(
        cfg.stage1_checkpoint,
        cfg.train_filename,
        cfg.validation_filename,
        cfg.parameters,
        cfg.output_dir,
    )
    save_tl_config(
        parameters,
        f"{cfg.config_filename}",
    )
    run_reinvent(
        config_filename=cfg.config_filename,
        device=cfg.parameters.device,
        output_dir=cfg.output_dir,
        smiles_column=cfg.smiles_column,
    )


def split_data(
    data: pd.DataFrame, n_head: int, n_tail: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets.

    Args:
        data (pd.DataFrame): Input dataset.
        n_head (int): Number of molecules for training.
        n_tail (int): Number of molecules for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation datasets.
    """
    logger.info(
        "Number of molecules for: training=%d, validation=%d",
        n_head,
        n_tail,
    )
    train = data.head(n_head)
    validation = data.tail(n_tail)
    return train, validation


def save_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_filename: str,
    validation_filename: str,
) -> None:
    """Save training and validation data to CSV files.

    Args:
        train (pd.DataFrame): Training dataset.
        validation (pd.DataFrame): Validation dataset.
        train_filename (str): Filename for training data.
        validation_filename (str): Filename for validation data.
    """
    Path(train_filename).write_text(
        train.to_csv(sep="\t", index=False, header=False)
    )
    Path(validation_filename).write_text(
        validation.to_csv(sep="\t", index=False, header=False)
    )


def create_tl_parameters(
    stage1_checkpoint: str,
    train_filename: str,
    validation_filename: str,
    parameters: DictConfig,
    output_path: str,
) -> str:
    """Create transfer learning parameters configuration.

    Args:
        stage1_checkpoint (str): Path to stage 1 checkpoint model.
        train_filename (str): Filename for training data.
        validation_filename (str): Filename for validation data.
        parameters (DictConfig): Parameters from Hydra configuration.

    Returns:
        str: Transfer learning parameters configuration.
    """
    return f"""
run_type = "transfer_learning"
device = "{parameters.device}"
tb_logdir = "{output_path}/tb_TL"

[parameters]

num_epochs = {parameters.num_epochs}
save_every_n_epochs = {parameters.save_every_n_epochs}
batch_size = {parameters.batch_size}
sample_batch_size = {parameters.sample_batch_size}

input_model_file = "{stage1_checkpoint}"
output_model_file = "{output_path}/model/TL_reinvent.model"
smiles_file = "{train_filename}"
validation_smiles_file = "{validation_filename}"
standardize_smiles = {str(parameters.standardize_smiles).lower()}
randomize_smiles = {str(parameters.randomize_smiles).lower()}
randomize_all_smiles = {str(parameters.randomize_all_smiles).lower()}
internal_diversity = {str(parameters.internal_diversity).lower()}
"""


def save_tl_config(parameters: str, config_filename: str) -> None:
    """Save transfer learning configuration to a file.

    Args:
        parameters (str): Transfer learning parameters configuration.
        config_filename (str): Filename for configuration file.
    """
    Path(config_filename).write_text(parameters)


def run_reinvent(config_filename: str, device, output_dir,smiles_column) -> None:
    """Run the REINVENT transfer learning process.

    Args:
        config_filename (str): Filename for transfer learning configuration.
    """
    print(f"Running REINVENT with parameters: {config_filename}")
    print(f"Using device: {device}")

    input_config = config_parse.read_toml(config_filename)
    run_transfer_learning(
        input_config=extract_sections(input_config),
        device=device,
        tb_logdir=f"{output_dir}/tb_TL",
        write_config=None,
        smiles_column=smiles_column,
    )


def extract_sections(config: dict) -> dict:
    """Extract the sections of a config file

    :param config: the config file
    :returns: the extracted sections
    """
    # FIXME: stages are a list of dicts in RL, may clash with global lists
    return {k: v for k, v in config.items() if isinstance(v, (dict, list))}


if __name__ == "__main__":
    main()
