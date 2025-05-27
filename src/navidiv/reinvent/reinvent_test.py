import logging
import os

import pandas as pd

from navidiv.utils import groupby_results, initialize_scorer


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
        self.prop_dict = scorer_dicts["prop_dict"]
        self.scorer = None
        self.score_every = scorer_dicts["score_every"]
        self.groupby_every = scorer_dicts["groupby_every"]
        self.selection_criteria = scorer_dicts["selection_criteria"]
        self.custom_alert_name = scorer_dicts["custom_alert_name"]

    def run_groupby(self):
        """Run the groupby operation on the DataFrame to find the fragments to avoid."""
        df_fragment_path = (
            f"{self.scorer._output_path}/{self.scorer._csv_name}.csv"
        )
        if not os.path.exists(df_fragment_path):
            raise FileNotFoundError(
                f"File {df_fragment_path} does not exist. Please run the scorer first."
            )
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

    def update_custom_alerts(self, reinvnet_learning):
        """Update the custom alerts with the new SMARTS"""
        csv_path = f"{self.scorer._output_path}/selected_fragments_{self.scorer._csv_name}.csv"
        if not os.path.exists(csv_path):
            logging.warning(
                f"File {csv_path} does not exist. Please run the scorer first."
            )
            return
        fragment_to_add = pd.read_csv(csv_path, index_col=False)[
            "Substructure"
        ].tolist()
        for scorer in reinvnet_learning.scoring_function.components.filters:
            print(scorer.component_type)
            if scorer.component_type == self.custom_alert_name:
                old_fragments = set(scorer.params[1].frag)
                old_fragments.update(fragment_to_add)

                scorer.params[1].frag = list(old_fragments)
                logging.info(
                    f"ngrams Custom alerts updated with {old_fragments}"
                )
                return
        logging.warning("No CustomAlertsngrams found in the scoring function")


class nameddict(dict):
    __getattr__ = dict.__getitem__


class reinvent_test:
    def __init__(self, csv_file: str, scorer_dicts: []) -> None:
        """Initialize the ReinventTest class.

        Args:
            csv_file (str): Path to the CSV file containing the data.
        """
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        self.scorer_dicts = scorer_dicts
        self.scoring_function = nameddict(
            {
                "components": nameddict(
                    {
                        "filters": [
                            nameddict(
                                {
                                    "component_type": "customalertsngrams",
                                    "params": [None, nameddict({"frag": []})],
                                }
                            )
                        ],
                    }
                ),
            }
        )
        print(self.scoring_function.components)
        self.scorers = []

    def run_scorer(self, step_df, scorer):
        smiles_list = step_df["SMILES"].tolist()
        scores = step_df["Score"].tolist()
        step = step_df["step"].tolist()[0]

        try:
            scores_result = scorer.get_score(
                smiles_list=smiles_list,
                scores=scores,
                additional_columns_df={"step": step},
            )
            logging.info(
                f"Step {step} - Scorer: {scorer._csv_name} - Score: {scores_result}"
            )
        except Exception as e:
            logging.exception(f"Error processing step {step}: {e}")

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
                    logging.info(
                        f"Custom alert for {scorer.custom_alert_name} exists."
                    )
                    exist = True
                    break

            if not exist:
                logging.info(
                    f"Custom alert for {scorer.custom_alert_name} does not exist."
                )

    def optimize(self):
        """Optimize the DataFrame by removing duplicates and resetting the index."""
        self.initialize_scorer()
        for step in self.df["step"].unique():
            step_df = self.df[self.df["step"] == step]

            if step_df.empty:
                continue
            for scorer in self.scorers:
                if step % scorer.score_every == 0:
                    self.run_scorer(step_df, scorer.scorer)
                if step % scorer.groupby_every == 0:
                    logging.info(f"Step {step} - Grouping results")
                    scorer.run_groupby()
                    scorer.update_custom_alerts(self)
