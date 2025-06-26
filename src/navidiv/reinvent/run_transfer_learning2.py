"""Reinvent transfer learning

Reads in a SMILES file and performs transfer learning.
"""

import logging
import os

import torch
import torch.optim as topt
from reinvent.chemistry import conversions
from reinvent.chemistry.standardization.rdkit_standardizer import (
    RDKitStandardizer,
)
from reinvent.models.reinvent.models.vocabulary import create_vocabulary
from reinvent.runmodes import TL, create_adapter
from reinvent.runmodes.TL.validation import TLConfig
from reinvent.utils import read_smiles_csv_file, setup_reporter

logger = logging.getLogger(__name__)


def updata_vocabulary(smiles_list, adapter, model_filename, device):
    """Update the vocabulary with new SMILES strings.

    :param smiles: list of SMILES strings
    :param adapter: adapter with tokenizer

    :return: None
    """
    vocabulary = create_vocabulary(smiles_list, adapter.tokenizer)
    adapter.vocabulary.update(vocabulary.tokens())
    adapter.save_to_file(model_filename.replace(".", "_updated."))
    adapter, _, model_type = create_adapter(
        model_filename.replace(".", "_updated."), "training", device
    )
    adapter.device = device
    adapter.model.device = device
    adapter.network._embedding = adapter.network._embedding.to(device)
    adapter.network._rnn = adapter.network._rnn.to(device)
    adapter.network._linear = adapter.network._linear.to(device)
    logger.info(f"adapter {adapter}")
    return adapter, model_type


def run_transfer_learning(
    input_config: dict,
    device: torch.device,
    tb_logdir: str = None,
    write_config: str = None,
    smiles_column: int = 0,
    *args,
    **kwargs,
):
    """Run transfer learning with Reinvent

    :param input_config: the run configuration
    :param device: torch device
    :param tb_logdir: log directory for TensorBoard
    :param write_config: callable to write config
    """
    logger.info("Starting Transfer Learning")

    config = TLConfig(**input_config)
    # device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    parameters = config.parameters
    scheduler_config = config.scheduler

    model_filename = parameters.input_model_file
    adapter, _, model_type = create_adapter(model_filename, "training", device)

    logger.info(f"Using generator {model_type}")

    smiles_filename = os.path.abspath(parameters.smiles_file)
    do_standardize = parameters.standardize_smiles

    randomize_all_smiles = parameters.randomize_all_smiles
    do_randomize = parameters.randomize_smiles and not randomize_all_smiles

    actions = []
    cols = smiles_column

    # FIXME: move to preprocessing
    if model_type == "Reinvent":
        if do_standardize:
            standardizer = RDKitStandardizer(None, isomeric=False)
            actions.append(standardizer.apply_filter)

        if do_randomize:
            actions.append(conversions.randomize_smiles)
    elif model_type == "Mol2Mol":
        if do_standardize:
            actions.append(conversions.convert_to_standardized_smiles)
    else:
        cols = slice(0, 2, None)

    # NOTE: we expect here that all data will fit into memory
    smilies = read_smiles_csv_file(
        smiles_filename, cols, actions=actions, remove_duplicates=True
    )
    logger.info(f"Read {len(smilies)} input SMILES from {smiles_filename}")
    adapter, model_type = updata_vocabulary(
        smilies, adapter, model_filename, device
    )
    if not smilies:
        msg = f"Unable to read valid SMILES from {smiles_filename}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    validation_smiles_filename = parameters.validation_smiles_file
    validation_smilies = None

    if validation_smiles_filename:
        validation_smiles_filename = os.path.abspath(
            validation_smiles_filename
        )
        validation_smilies = read_smiles_csv_file(
            validation_smiles_filename,
            cols,
            actions=actions,
            remove_duplicates=True,
        )
        logger.info(
            f"Read {len(validation_smilies)} validation SMILES from {validation_smiles_filename}"
        )

    lr_config = TL.StepLRConfiguration(**scheduler_config)
    optimizer = topt.Adam(adapter.get_network_parameters(), lr=lr_config.lr)
    lr_scheduler = topt.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_config.step,
        gamma=lr_config.gamma,
    )

    # Ensure tensors in runner are on the same device
    runner_class = getattr(TL, f"{model_type}")
    optimize = runner_class(
        adapter,
        smilies,
        validation_smilies,
        tb_logdir,
        parameters,
        optimizer,
        lr_scheduler,
        lr_config,
    )
    logger.info(f"runner class: {runner_class}, device {optimize.device}")
    if "responder" in config:
        url = config.responder.endpoint
        success = setup_reporter(url)

        if success:
            logger.info(f"Remote reporting to {url}")

    if callable(write_config):
        write_config(config.model_dump())

    optimize()
