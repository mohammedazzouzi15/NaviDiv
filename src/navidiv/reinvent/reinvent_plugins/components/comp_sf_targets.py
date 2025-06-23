from __future__ import annotations

__all__ = ["ChemPropFORMED", "ChemPropFORMEDSTI", "ChemPropFORMEDT1T2"]
import logging
from typing import List

import chemprop
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from reinvent.scoring.utils import suppress_output

from .excit_obj_func import energy_score, energy_score_sti

from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.normalize import normalize_smiles


logger = logging.getLogger("reinvent")

def add_tag(label: str, text: str = "True"):
    """A simple decorator to tag a class"""

    def wrapper(cls):
        setattr(cls, label, text)
        return cls

    return wrapper


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    checkpoint_dir: List[str]
    rdkit_2d_normalized: List[bool] = Field(default_factory=lambda: [False])
    target_column: List[str] = Field(
        default_factory=lambda: ["S1"]
    )  # S1, T1, STI


@add_tag("__component")
class ChemPropFORMED:
    def __init__(self, params: Parameters):
        logger.info(f"Using ChemProp version {chemprop.__version__}")
        self.chemprop_params = []

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"

        for checkpoint_dir, rdkit_2d_normalized in zip(
            params.checkpoint_dir, params.rdkit_2d_normalized
        ):
            logging.info(f"Loading ChemProp model from {checkpoint_dir}")
            # Load the ChemProp model
            args = [
                "--checkpoint_dir",  # ChemProp models directory
                checkpoint_dir,
                "--test_path",  # required
                "/dev/null",
                "--preds_path",  # required
                "/dev/null",
            ]

            if rdkit_2d_normalized:
                args.extend(
                    ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"]
                )

            with suppress_output():
                chemprop_args = chemprop.args.PredictArgs().parse_args(args)
                chemprop_model = chemprop.train.load_model(args=chemprop_args)

                self.chemprop_params.append((chemprop_model, chemprop_args))
        logger.info(f"Loaded {len(self.chemprop_params)} ChemProp models")

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        smilies_list = [[smiles] for smiles in smilies]
        scores = []

        for model, args in self.chemprop_params:
            with suppress_output():
                preds = chemprop.train.make_predictions(
                    model_objects=model,
                    smiles=smilies_list,
                    args=args,
                    return_invalid_smiles=True,
                    return_uncertainty=False,
                )

            scores_raw = np.array(
                [
                    energy_score(val[1], val[0]) if "Invalid SMILES" not in val else np.nan
                    for val in preds
                ],
                dtype=float,
            )
            max_score = scores_raw.max()
            scores_raw = scores_raw + max_score / 2

            # assume S1, T1 are the first two values in the list
            scores.append(scores_raw)

        return ComponentResults(scores)


@add_tag("__component")
class ChemPropFORMEDT1T2:
    def __init__(self, params: Parameters):
        logger.info(f"Using ChemProp version {chemprop.__version__}")
        self.chemprop_params = []

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"

        for checkpoint_dir, rdkit_2d_normalized in zip(
            params.checkpoint_dir, params.rdkit_2d_normalized
        ):
            args = [
                "--checkpoint_dir",  # ChemProp models directory
                checkpoint_dir,
                "--test_path",  # required
                "/dev/null",
                "--preds_path",  # required
                "/dev/null",
            ]

            if rdkit_2d_normalized:
                args.extend(
                    ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"]
                )

            with suppress_output():
                chemprop_args = chemprop.args.PredictArgs().parse_args(args)
                chemprop_model = chemprop.train.load_model(args=chemprop_args)

                self.chemprop_params.append((chemprop_model, chemprop_args))

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        smilies_list = [[smiles] for smiles in smilies]
        scores = []

        for model, args in self.chemprop_params:
            with suppress_output():
                preds = chemprop.train.make_predictions(
                    model_objects=model,
                    smiles=smilies_list,
                    args=args,
                    return_invalid_smiles=True,
                    return_uncertainty=False,
                )
            # assume S1, T1 are the first two values in the list
            scores.append(
                np.array(
                    [
                        self.t2_t1_gap(val[0], val[1]) if "Invalid SMILES" not in val else np.nan
                        for val in preds
                    ],
                    dtype=float,
                )
            )

        return ComponentResults(scores)

    def t2_t1_gap(self, t1, t2):
        return t2 - t1


@add_tag("__component")
class ChemPropFORMEDSTI:
    def __init__(self, params: Parameters):
        logger.info(f"Using ChemProp version {chemprop.__version__}")
        self.chemprop_params = []

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"

        for checkpoint_dir, rdkit_2d_normalized in zip(
            params.checkpoint_dir, params.rdkit_2d_normalized
        ):
            args = [
                "--checkpoint_dir",  # ChemProp models directory
                checkpoint_dir,
                "--test_path",  # required
                "/dev/null",
                "--preds_path",  # required
                "/dev/null",
            ]

            if rdkit_2d_normalized:
                args.extend(
                    ["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"]
                )

            with suppress_output():
                chemprop_args = chemprop.args.PredictArgs().parse_args(args)
                chemprop_model = chemprop.train.load_model(args=chemprop_args)

                self.chemprop_params.append((chemprop_model, chemprop_args))

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        smilies_list = [[smiles] for smiles in smilies]
        scores = []

        for model, args in self.chemprop_params:
            with suppress_output():
                preds = chemprop.train.make_predictions(
                    model_objects=model,
                    smiles=smilies_list,
                    args=args,
                    return_invalid_smiles=True,
                    return_uncertainty=False,
                )

            # assume S1, T1 are the first two values in the list
            scores.append(
                np.array(
                    [
                        energy_score_sti(val[0], val[1]) if "Invalid SMILES" not in val else np.nan
                        for val in preds
                    ],
                    dtype=float,
                )
            )

        return ComponentResults(scores)

