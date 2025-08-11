"""The Reinvent optimization algorithm"""

from __future__ import annotations

__all__ = ["Reinvent3Learning"]
import logging
import torch
from typing import TYPE_CHECKING
import numpy as np
from reinvent.runmodes.RL.data_classes import ModelState
from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent.runmodes.RL.learning import Learning
import time
from reinvent_plugins.normalizers.rdkit_smiles import normalize
from reinvent.models.meta_data import update_model_data
import os

from navidiv.utils import groupby_results, initialize_scorer
import pandas as pd

if TYPE_CHECKING:
    from reinvent.runmodes.samplers import Sampler
    from reinvent.runmodes.RL import RLReward, terminator_callable
    from reinvent.runmodes.RL.memories import Inception
    from reinvent.models import ModelAdapter
    from reinvent.scoring import Scorer, ScoreResults


logger = logging.getLogger("reinvent")


class ScorerParams:
    def __init__(self, scorer_dicts: {}):
        """Initialize the ScorerParams class.

        Args:
            scorer_dicts (dict): Dictionary containing properties of the scorer.
            the dictionary should contain the following
            keys:
                - prop_dict: Dictionary containing properties of the
                - selectrion_criteria: Dictionary containing selection
                    criteria for the fragments.
                - score_every: Frequency of scoring.
                - groupby_every: Frequency of grouping results.
        """
        logger.info(f"Scorer dicts: {scorer_dicts.prop_dict}")
        self.prop_dict = scorer_dicts.prop_dict
        self.scorer = None
        self.score_every = scorer_dicts.score_every
        self.groupby_every = scorer_dicts.groupby_every
        self.selection_criteria = scorer_dicts.selection_criteria
        self.custom_alert_name = scorer_dicts.custom_alert_name

    def run_groupby(self):
        """Run the groupby operation on the DataFrame to find the fragments to avoid."""
        df_fragment_path = f"{self.scorer._output_path}/{self.scorer._csv_name}.csv"
        if not os.path.exists(df_fragment_path):
            raise FileNotFoundError(
                f"File {df_fragment_path} does not exist. Please run the scorer first."
            )
        if not os.path.exists(
            f"{self.scorer._output_path}/{self.scorer._csv_name}_with_score.csv"
        ):
            return

        df_fragments = pd.read_csv(
            f"{self.scorer._output_path}/{self.scorer._csv_name}_with_score.csv",
            index_col=False,
        )
        groupby_results_df = groupby_results(df_fragments)
        groupby_results_df.to_csv(
            f"{self.scorer._output_path}/groupby_results_{self.scorer._csv_name}.csv",
            index=False,
        )
        selected_fragments = groupby_results_df.copy()
        condition = pd.Series([False] * len(selected_fragments))
        for col, value in self.selection_criteria.items():
            if col not in selected_fragments.columns:
                raise ValueError(
                    f"Column {col} not found in the grouped by  DataFrame."
                )
            condition = condition | (selected_fragments[col] > value)
        selected_fragments = selected_fragments[condition]
        selected_fragments.to_csv(
            f"{self.scorer._output_path}/selected_fragments_{self.scorer._csv_name}.csv",
            index=False,
        )

    def update_custom_alerts(self, reinvnet_learning, step=0):
        """Update the custom alerts with the new SMARTS"""
        csv_path = (
            f"{self.scorer._output_path}/selected_fragments_{self.scorer._csv_name}.csv"
        )
        if not os.path.exists(csv_path):
            logger.warning(
                f"File {csv_path} does not exist. Please run the scorer first."
            )
            return
        fragment_to_add = pd.read_csv(csv_path, index_col=False)[
            "Substructure"
        ].tolist()
        logger.info(
            f"Updating custom alert {self.custom_alert_name} with fragments: {fragment_to_add}"
        )
        for scorer in reinvnet_learning.scoring_function.components.filters:
            # print(scorer.component_type)
            if scorer.component_type == self.custom_alert_name:
                old_fragments = set(scorer.params[1].frag)
                old_fragments.update(fragment_to_add)

                scorer.params[1].frag = list(old_fragments)
                logger.info(f"{scorer.component_type} updated with {old_fragments}")
                if len(fragment_to_add) > 0:
                    with open(
                        "filter_operations.txt",
                        "a",
                    ) as f:
                        f.write(
                            f"Custom alert {self.custom_alert_name} updated with fragments: {fragment_to_add} at step {step}\n"
                        )
                return
        logger.warning(f"No {self.custom_alert_name} found in the scoring function")

    def run_scorer(self, smiles_list, scores, step):
        """Run the scorer on the given DataFrame.
        Args:
            smiles_list (list): List of SMILES strings.
            scores (list): List of scores.
            step (int): Current step number.
        """
        try:
            scores_result = self.scorer.get_score(
                smiles_list=smiles_list,
                scores=scores,
                additional_columns_df={"step": step},
            )
            scores_result_copy = scores_result.copy()
            scores_result_copy.pop("Unique Fragments")
            logger.info(
                f"Step {step} - Scorer: {self.scorer._csv_name} - Score: {scores_result_copy}"
            )
        except Exception as e:
            logger.exception(f"Error processing step {step}: {e}")


class Reinvent3Learning(Learning):
    """Reinvent optimization"""

    def __init__(
        self,
        max_steps: int,
        prior: ModelAdapter,
        state: ModelState,
        scoring_function: Scorer,
        reward_strategy: RLReward,
        sampling_model: Sampler,
        smilies,
        distance_threshold: int,
        rdkit_smiles_flags: dict,
        inception: Inception = None,
        responder_config: dict = None,
        tb_logdir: str = None,
        avoid: ModelAdapter = None,
        scorer_dicts: list = [],
    ):
        super().__init__(
            max_steps,
            0,
            prior,
            state,
            scoring_function,
            reward_strategy,
            sampling_model,
            smilies,
            distance_threshold,
            rdkit_smiles_flags,
            inception,
            responder_config,
            tb_logdir,
        )

        self.avoid = avoid  # this is for avoid agents
        self.scorer_dicts = scorer_dicts
        self.scorers = []

    def save_model(self, step):
        data = self.get_state_dict()
        save_dict = update_model_data(data, comment="RL")
        output_filename = f"checkpoint_s{step}.chkpt"
        torch.save(save_dict, output_filename)

    def initialize_scorer(self):
        """Initialize the scorer based on the properties provided.

        Args:
            scorer_dict (dict): Properties of the scorer.

        Returns:
            object: Initialized scorer object.
        """
        for scorer_dict in self.scorer_dicts:
            scorer = ScorerParams(scorer_dict)
            scorer.scorer = initialize_scorer(scorer_dict.get("prop_dict"))
            self.scorers.append(scorer)
            logging.info(
                f"Scorer {scorer.scorer._csv_name} initialized with properties: {scorer_dict}"
            )
            exist = False
            for component in self.scoring_function.components.filters:
                if component.component_type == scorer.custom_alert_name:
                    logging.info(f"Custom alert for {scorer.custom_alert_name} exists.")
                    exist = True
                    break

            if not exist:
                logging.info(
                    f"Custom alert for {scorer.custom_alert_name} does not exist."
                )

    def update(self, results: ScoreResults):
        """Run the learning strategy"""

        agent_nlls = self._state.agent.likelihood_smiles(self.sampled.items2)
        prior_nlls = self.prior.likelihood_smiles(self.sampled.items2)
        if self.avoid is not None:
            avoid_nlls = [x.likelihood_smiles(self.sampled.items2) for x in self.avoid]
        return self.reward_nlls(
            agent_nlls,  # agent NLL
            prior_nlls,  # prior NLL
            results.total_scores,
            self.inception,
            results.smilies,
            self._state.agent,
            np.argwhere(self.sampled.states == SmilesState.VALID).flatten(),
            avoid_nlls=avoid_nlls,
        )

    def optimize(self, converged: terminator_callable) -> bool:
        """Run the multistep optimization loop

        Sample from the agent, score the SNILES, update the agent parameters.
        Log some key characteristics of the current step.

        :param converged: a callable that determines convergence
        :returns: whether max_steps has been reached
        """

        step = -1
        scaffolds = None
        self.start_time = time.time()
        self.initialize_scorer()

        for step in range(self.max_steps):
            self.sampled = self.sampling_model.sample(self.seed_smilies)
            self.invalid_mask = np.where(
                self.sampled.states == SmilesState.INVALID, False, True
            )
            self.duplicate_mask = np.where(
                self.sampled.states == SmilesState.DUPLICATE, False, True
            )
            mask_valid = np.where(
                (self.sampled.states == SmilesState.VALID)
                | (self.sampled.states == SmilesState.DUPLICATE),
                True,
                False,
            ).astype(bool)
            # check scorer components
            # self.update_custom_alerts(["CC"])

            results = self.score()

            if self.prior.model_type == "Libinvent":
                results.smilies = normalize(results.smilies)

            if self._state.diversity_filter:
                df_mask = np.where(self.invalid_mask, True, False)

                scaffolds = self._state.diversity_filter.update_score(
                    results.total_scores, results.smilies, df_mask
                )
            for scorer in self.scorers:
                if step % scorer.score_every == 0:
                    scorer.run_scorer(results.smilies, results.total_scores, step)
                if step % scorer.groupby_every == 0:
                    logging.info(f"Step {step} - Grouping results")
                    scorer.run_groupby()
                    scorer.update_custom_alerts(self, step)

            # FIXME: check for NaNs
            #        inception filter
            agent_lls, prior_lls, augmented_nll, loss = self.update(results)

            state_dict = self._state.as_dict()
            self._state_info.update(state_dict)

            nan_idx = np.isnan(results.total_scores)
            scores = results.total_scores[~nan_idx]
            mean_scores = scores.mean()
            self.report(
                step,
                mean_scores,
                scaffolds,
                score_results=results,
                agent_lls=agent_lls,
                prior_lls=prior_lls,
                augmented_nll=augmented_nll,
                loss=float(loss),
            )

            if converged(mean_scores, step):
                logger.info(f"Terminating early in {step = }")
                break

        if self.tb_reporter:  # FIXME: context manager?
            self.tb_reporter.flush()
            self.tb_reporter.close()

        if step >= self.max_steps - 1:
            return True

        return False

    __call__ = optimize
