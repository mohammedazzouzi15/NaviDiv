import os
import subprocess
from pathlib import Path

from omegaconf import DictConfig


class model_params:
    def __init__(self, cfg, id):
        self.property_name = cfg.params_list[id]
        self.model_name = cfg.gnn_list[id]
        self.lower_bound = cfg.lower_bound[id]
        self.upper_bound = cfg.higher_bound[id]
        self.model_dir = cfg.model_dir
        self.scoring_function = cfg.scoring_function[id]


class model_params_chemprop:
    def __init__(self, cfg, id):
        self.params_list = cfg.params_list[id]
        self.model_dir = cfg.model_dir[id]
        self.weight = cfg.weight[id]
        self.higher_bound = cfg.higher_bound[id]
        self.lower_bound = cfg.lower_bound[id]
        self.scoring_function = cfg.scoring_function[id]


class InputGenerator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.wd = cfg.wd + "/" + cfg.name
        Path(self.wd).mkdir(parents=True, exist_ok=True)
        if "model_params" in cfg.stage_comp:
            self.list_of_properties = cfg.stage_comp.model_params.params_list
            self.check_list_length()
        else:
            self.list_of_properties = []
        if "model_params_chemprop" in cfg.stage_comp:
            self.list_of_properties_chemprop = (
                cfg.stage_comp.model_params_chemprop.params_list
            )
        else:
            self.list_of_properties_chemprop = []

    def run(self):
        os.chdir(self.wd)
        self.generate_input()
        subprocess.run(
            ["reinvent", "-l", "stage1.log", "stage1.toml"], check=False
        )

    def generate_input(self):
        stage1_parameters = self.get_reinvent_sampling_config()
        for i in range(len(self.list_of_properties)):
            dict_config = model_params(
                self.cfg.stage_comp.model_params_FORMOR_PROP, i
            )
            stage1_parameters = self.add_stage_component(
                stage1_parameters,
                first_time="true" if i == 0 else "false",
                last_time="true"
                if i == len(self.list_of_properties) - 1
                else "false",
                dict_config=dict_config,
            )
        for i in range(len(self.list_of_properties_chemprop)):
            dict_config = model_params_chemprop(
                self.cfg.stage_comp.model_params_chemprop, i
            )
            stage1_parameters = self.add_chempropFormed(
                stage1_parameters,
                dict_config.model_dir,
                dict_config.params_list,
                dict_config.weight,
                dict_config.higher_bound,
                dict_config.lower_bound,
                dict_config.scoring_function,
            )
        if self.cfg.reinvent_common.include_rdkit_requirements:
            stage1_parameters = self.add_stage_component_rdkit_requirement(
                stage1_parameters,
                self.cfg.stage_comp.rdkit_requirement.max_number_of_rings,
                self.cfg.stage_comp.rdkit_requirement.max_num_atoms,
                self.cfg.stage_comp.rdkit_requirement.max_ring_size,
            )
        if self.cfg.reinvent_common.include_scaffold_score:
            stage1_parameters = self.add_stage_component_scaffoldscore(
                stage1_parameters, self.cfg.scaffold_score.checkpoints
            )
        if self.cfg.reinvent_common.include_molecular_weight:
            stage1_parameters = self.add_Mw_filter(stage1_parameters)

        if self.cfg.reinvent_common.include_SA_score:
            stage1_parameters = self.add_SA_filter(stage1_parameters)

        if self.cfg.reinvent_common.ngram_filter.include:
            stage1_parameters = self.add_ngram_filter(stage1_parameters)
            stage1_parameters = self.add_ngram_filter_params(stage1_parameters)
        stage1_parameters = self.add_diversity_filter(stage1_parameters)

        if self.cfg.reinvent_common.include_inception:
            stage1_parameters = self.add_inception_filter(stage1_parameters)

        if self.cfg.reinvent_common.fragment_filter.include:
            stage1_parameters = self.add_fragments_filter(stage1_parameters)
            stage1_parameters = self.add_substructure_filter(
                stage1_parameters,
            )
        if self.cfg.reinvent_common.dissimilarity_filter.include:
            stage1_parameters = self.add_dissimilarity_filter(
                stage1_parameters,
            )
        if self.cfg.reinvent_common.scaffold_filter.include:
            stage1_parameters = self.add_scaffold_filter(
                stage1_parameters,
            )
        with Path(self.wd + "/stage1.toml").open("w") as f:
            f.write(stage1_parameters)
        return stage1_parameters

    def generate_input_sampling(self):
        stage1_parameters = self.get_reinvent_sampling_config_only()
        with Path(self.wd + "/stage1.toml").open("w") as f:
            f.write(stage1_parameters)
        return stage1_parameters

    def check_list_length(self):
        if len(self.list_of_properties) != len(
            self.cfg.model_params.higher_bound
        ) or len(self.list_of_properties) != len(
            self.cfg.model_params.lower_bound
        ):
            raise ValueError(
                "Length of list_of_properties, higher_bound and lower_bound must be equal"
            )

    def get_reinvent_sampling_config_only(self):
        stage1_parameters = f"""
    run_type = "{self.cfg.run_sampling.run_type}"
    device = "cuda:0"
    json_out_config = "_stage1.json"

    [parameters]
    num_smiles = "{self.cfg.run_sampling.num_smiles}"
    model_file = "{self.cfg.run_sampling.model_filename}"
    output_file = "{self.cfg.run_sampling.output_filename}"
        """
        return stage1_parameters

    def get_reinvent_sampling_config(self):
        stage1_parameters = f"""
    run_type = "{self.cfg.reinvent_common.run_type}"
    device = "cuda:0"
    json_out_config = "_stage1.json"

    [parameters]

    prior_file = "{self.cfg.reinvent_common.prior_filename}"
    agent_file = "{self.cfg.reinvent_common.agent_filename}"
    avoid_file = "{self.cfg.reinvent_common.avoid_filename}"
    check_diversity = "{self.cfg.reinvent_common.check_diversity}"
    check_diverisity_every = {self.cfg.reinvent_common.check_diversity_every}

    summary_csv_prefix = "{self.cfg.name}"

    batch_size = {self.cfg.reinvent_common.batch_size}

    use_checkpoint = true

    [learning_strategy]

    type = "{self.cfg.reinvent_common.learning_strategy.type}"
    sigma = "{self.cfg.reinvent_common.learning_strategy.sigma}"
    rate = "{self.cfg.reinvent_common.learning_strategy.lr}"


    [[stage]]

    max_score = 1.0
    max_steps = {self.cfg.reinvent_common.max_steps}

    chkpt_file = "{self.cfg.name}.chkpt"

    [stage.scoring]

    type = "geometric_mean"  # or arithmetic_mean
    parallel = false  # do not run scoring components in parallel
        """
        return stage1_parameters

    def add_stage_component(
        self, stage_parameters, first_time, last_time, dict_config
    ):  # stage.scoring.component.FairChemG
        component_parameters = f"""
    [[stage.scoring.component]]

    [[stage.scoring.component.{dict_config.scoring_function}.endpoint]] 
    name = "FairChem_{dict_config.property_name}"
    weight = 0.6


    params.property_name = "{dict_config.property_name}"
    params.model = "{dict_config.model_name}"
    params.lower_bound = {dict_config.lower_bound}
    params.upper_bound = {dict_config.upper_bound}
    params.model_dir = "{dict_config.model_dir}"
    params.first_time = {first_time}
    params.last_time = {last_time}


        """
        return stage_parameters + component_parameters

    def add_Mw_filter(self, stage_parameters):
        component_parameters = """
    [[stage.scoring.component]]
    [stage.scoring.component.MolecularWeight]
    [[stage.scoring.component.MolecularWeight.endpoint]]
    name = "Molecular weight"  # user chosen name for output
    weight = 0.6  # weight to fine-tune the relevance of this component

    # A transform ensures that the output from the scoring component ranges
    # from 0 to 1 to serve as a proper score.  Here we use a double sigmoid
    # to transform weights into the range 200-500 a.u.
    transform.type = "double_sigmoid"
    transform.high = 800.0
    transform.low = 300.0
    transform.coef_div = 500.0
    transform.coef_si = 20.0
    transform.coef_se = 20.0

    """
        return stage_parameters + component_parameters

    def add_SA_filter(self, stage_parameters):
        component_parameters = """

    [[stage.scoring.component]]

    [stage.scoring.component.SAScore]
    [[stage.scoring.component.SAScore.endpoint]]
    name = "SA score"
    weight = 1
    transform.type = "reverse_sigmoid"
    transform.high = 10
    transform.low = 1
    transform.k = 0.4
    """
        return stage_parameters + component_parameters

    def add_chempropFormed(
        self,
        stage_parameters,
        model_checkpoint_dir,
        name,
        weight,
        normhigh,
        normlow,
        scoring_function,
    ):
        component_parameters = f"""
    [[stage.scoring.component]]

    [[stage.scoring.component.{scoring_function}.endpoint]] 
    name = "{name}"
    weight = {weight}

    params.checkpoint_dir = "{model_checkpoint_dir}"
    params.rdkit_2d_normalized = false

    transform.type = "sigmoid"
    transform.high = {normhigh}
    transform.low = {normlow}
    transform.k = 0.4
    """
        return stage_parameters + component_parameters

    def add_stage_component_rdkit_requirement(
        self,
        stage_parameters,
        max_number_of_rings,
        max_num_atoms,
        max_ring_size,
    ):  # stage.scoring.component.rdkit_requirement
        component_parameters = f"""
    [[stage.scoring.component]]

    [[stage.scoring.component.rdkit_requirement.endpoint]]
    name = "rdkit_requirement"
    weight = 0.6

    params.max_number_of_rings = {max_number_of_rings}
    params.max_num_atoms = {max_num_atoms}
    params.max_ring_size = {max_ring_size}
        """
        return stage_parameters + component_parameters

    def add_stage_component_scaffoldscore(self, stage_parameters, checkpoints):
        component_parameters = f"""
    [[stage.scoring.component]]

    [[stage.scoring.component.scaffold_score.endpoint]]
    name = "scaffold_score"
    weight = 2.0

    params.checkpoints = "{checkpoints}"
        """
        return stage_parameters + component_parameters

    def add_diversity_filter(
        self,
        stage_parameters,
    ):
        df_parameters = f"""
    [diversity_filter]

    type = "{self.cfg.reinvent_common.diversity_filter.div_type}"
    bucket_size = {self.cfg.reinvent_common.diversity_filter.bucket_size}
    minscore = {self.cfg.reinvent_common.diversity_filter.minscore}
    """
        return stage_parameters + df_parameters

    def add_inception_filter(self, stage_parameters):
        inception_parameters = """
    [inception]

    smiles_file = ""  # no seed SMILES
    memory_size = 500
    sample_size = 10
    """
        return stage_parameters + inception_parameters

    def add_substructure_filter(self, stage_parameters):
        substructure_parameters = """
    [[stage.scoring.component]]
    [stage.scoring.component.custom_alerts]
    [[stage.scoring.component.custom_alerts.endpoint]]
    name = "custom alerts"
    weight = 1
    params.frag = [
    """
        smart_list = self.cfg.reinvent_common.fragment_filter.smart_list
        for smart in smart_list:
            substructure_parameters += f'"{smart}",\n'
        substructure_parameters += "]"
        return stage_parameters + substructure_parameters

    def add_ngram_filter(self, stage_parameters):
        ngram_parameters = """
    [[stage.scoring.component]]
    [stage.scoring.component.custom_alerts_ngrams]
    [[stage.scoring.component.custom_alerts_ngrams.endpoint]]
    name = "custom alerts ngrams"
    weight = 1
    params.frag = []
    """
        return stage_parameters + ngram_parameters

    def add_scaffold_filter(self, stage_parameters):
        ngram_parameters = f"""
    [[stage.scoring.component]]
    [stage.scoring.component.custom_alerts_scaffold]
    [[stage.scoring.component.custom_alerts_scaffold.endpoint]]
    name = "custom alerts scaffold"
    weight = 1
    params.frag = []
    params.type = "{self.cfg.reinvent_common.scaffold_filter.type}"
    """
        return stage_parameters + ngram_parameters

    def add_fragments_filter(self, stage_parameters):
        fragments_parameters = f"""
    [fragment_filter]

    min_count_fragments = {self.cfg.reinvent_common.fragment_filter.min_count_fragments}
    diff_median_score_limit = {self.cfg.reinvent_common.fragment_filter.diff_median_score_limit}"""
        return stage_parameters + fragments_parameters

    def add_ngram_filter_params(self, stage_parameters):
        ngram_parameters = f"""
    [ngram_filter]

    ngram_size = {self.cfg.reinvent_common.ngram_filter.ngram_size}
    min_count_ngram_ratio = {self.cfg.reinvent_common.ngram_filter.min_count_ngram_ratio}
    """
        return stage_parameters + ngram_parameters

    def add_dissimilarity_filter(self, stage_parameters):
        dissimilarity_parameters = f"""
    [[stage.scoring.component]]
    [stage.scoring.component.dissimilarity]
    [[stage.scoring.component.dissimilarity.endpoint]]

    name = "dissimilarity"
    weight = 1
    params.threshold = {self.cfg.reinvent_common.dissimilarity_filter.threshold}
    params.frag = [
    """
        smart_list = self.cfg.reinvent_common.dissimilarity_filter.frag
        for smart in smart_list:
            dissimilarity_parameters += f'"{smart}",\n'
        dissimilarity_parameters += "]"
        return stage_parameters + dissimilarity_parameters
