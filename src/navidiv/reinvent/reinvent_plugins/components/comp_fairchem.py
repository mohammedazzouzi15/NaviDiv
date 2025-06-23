"""Compute scores with ChemProp

[[component.ChemProp.endpoint]]
name = "ChemProp Score"
weight = 0.7

# component specific parameters
param.checkpoint_dir = "ChemProp/3CLPro_6w63"
param.rdkit_2d_normalized = true
param.target_column = "dG"

# transform
transform.type = "reverse_sigmoid"
transform.high = -5.0
transform.low = -35.0
transform.k = 0.4

# In case of multiclass models add endpoints as needed and set the target_column
"""

from __future__ import annotations

__all__ = ["fairchem"]
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from FORMED_PROP.uncertainty_estimation.ensemble import Ensemble
from hydra import (
    compose,
    initialize_config_dir,
)
from reinvent.scoring.utils import suppress_output
from scipy.stats import norm

from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.normalize import normalize_smiles


def probability_within_threshold(mu, sigma, lower_bound, upper_bound):
    """Calculate the probability that a value is within a certain threshold.

    Parameters:
    - mu: Mean of the predictions.
    - sigma: Standard deviation of the predictions.
    - lower_bound: Lower bound of the threshold.
    - upper_bound: Upper bound of the threshold.

    Returns:
    - probability: Probability that the value is within the threshold.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)

    # Calculate the CDF values for the lower and upper bounds
    cdf_lower = norm.cdf(lower_bound, loc=mu, scale=sigma)
    cdf_upper = norm.cdf(upper_bound, loc=mu, scale=sigma)

    # The probability that the value is within the threshold
    probability = cdf_upper - cdf_lower
    probability[probability < 1e-20] = 1e-20
    probability[np.isnan(probability)] = 1e-20
    return probability


logger = logging.getLogger("reinvent")


def add_tag(label: str, text: str = "True"):
    """A simple decorator to tag a class"""

    def wrapper(cls):
        setattr(cls, label, text)
        return cls

    return wrapper


@add_tag("__parameters")
@dataclass
class Params:
    def __init__(
        self,
        property_name="S",
        model="painn",
        lower_bound=None,
        upper_bound=None,
        run_name="test",
        first_time=False,
        last_time=False,
    ):
        with initialize_config_dir(
            version_base=None,
            config_dir="/media/mohammed/Work/OPT_MOL_GEN/examples/configuration_file_models",
        ):
            self.cfg = compose(
                config_name="config_pain_UE.yaml",
                overrides=[
                    "model=" + model[0],
                    "data.property_name=" + property_name[0],
                    "UE.train_models=False",
                    "optimizer.running_dir=/media/mohammed/Work/OPT_MOL_GEN/model_training_output/training/"
                    + run_name[0]
                    + "/",
                ],
            )
        self.cfg.data.splitter_path = (
            "/media/mohammed/Work/FORMED_PROP/" + self.cfg.data.splitter_path
        )
        self.cfg.data.db_path = (
            "/media/mohammed/Work/FORMED_PROP/" + self.cfg.data.db_path
        )
        self.upper_bound = upper_bound[0]
        self.lower_bound = lower_bound[0]
        self.first_time = first_time[0]
        self.last_time = last_time[0]


def check_valloader_exists(first_time):
    folder = Path("/media/mohammed/Work/OPT_MOL_GEN/reinvent_plugins/val_temp")
    folder.mkdir(parents=True, exist_ok=True)
    if first_time:
        logger.info("create new val_loader")
        return False, folder / "val_loader_first_time.pth"
    for file in folder.glob("*.pth"):
        filename = file.name
        file_path = folder / filename
        target_column_file = filename.split("_")[-1].split(".")[0]
        logger.info(f"target_column_file: {target_column_file}")
        # check the time difference
        if target_column_file == "time":
            return True, file_path
        file.unlink()
    raise ValueError(
        "No val loader found check the first time and last time in your input"
    )


@add_tag("__component")
class FairChem:
    def __init__(self, params: Params):
        #mp.set_start_method("spawn", force=True)
        logger.info("Using Fairchem ")
        """Initialise the ensemble."""
        UE = Ensemble(params.cfg)
        # self.target_column = params.cfg["data"]["property_name"]
        dataloader = UE.dataloader(UE.config)
        UE.models = UE.load_models()
        if any([model is None for model in UE.models]):
            logger.error("Models could not be loaded")
            raise ValueError("Models could not be loaded")
        self.UE = UE
        self.dataloader = dataloader
        self.number_of_endpoints = 1
        self.smiles_type = "rdkit_smiles"
        self.first_time = params.first_time
        self.last_time = params.last_time
        self.lower_bound = params.lower_bound
        self.upper_bound = params.upper_bound

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> np.array:
        with suppress_output():
            filenam_exists, filename = check_valloader_exists(self.first_time)
            if filenam_exists:
                val_loader = torch.load(filename)
            else:
                val_loader = self.dataloader.get_data_loader_from_smiles(
                    smilies
                )
                torch.save(val_loader, filename)
            dict_output = self.UE.predict_only(val_loader)
            # delete the val_loader
            if self.last_time:
                filename.unlink()
            val_loader = None
        prediction_value = dict_output["energy"].detach().cpu().numpy()
        prediction_uncertainty = (
            dict_output["energy_uncertainty"].detach().cpu().numpy()
        )
        scores = probability_within_threshold(
            prediction_value,
            prediction_uncertainty,
            self.lower_bound,
            self.upper_bound,
        )
        for i_x, x in enumerate(dict_output["name"]):
            if x == "not_valid":
                scores[i_x] = 0.0
        # set a lower bound for the scores
        scores = [np.array(scores).reshape(-1)]
        # logger.info(f"scores: {scores}")
        return ComponentResults(scores)
