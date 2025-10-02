
from omegaconf import DictConfig

from navidiv.reinvent.InputGenerator import InputGenerator


class model_params_rdkit_physchem:
    def __init__(self, cfg, id):
        self.property_name = cfg.params_list[id]
        self.lower_bound = cfg.lower_bound[id]
        self.upper_bound = cfg.higher_bound[id]
        self.weight = cfg.weight[id]


class InputGeneratorCustom(InputGenerator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._component_handlers["model_params_rdkit_physchem"] = (
            self.add_stage_component_rdkit_physchem
        )

    def add_stage_component_rdkit_physchem(self, stage1_parameters, component):
        def _add_stage_component_rdkit_physchem(
            stage_parameters, dict_config
        ):  # stage.scoring.component.FairChemG
            component_parameters = f"""
        [[stage.scoring.component]]

        [[stage.scoring.component.{dict_config.property_name}.endpoint]] 
        name = "{dict_config.property_name}"
        weight = {dict_config.weight}


        params.property_name = "{dict_config.property_name}"
            """
            return stage_parameters + component_parameters

        for i in range(self.components[component]):
            dict_config = model_params_rdkit_physchem(
                self.cfg.stage_comp.model_params_rdkit_physchem, i
            )
            stage1_parameters = _add_stage_component_rdkit_physchem(
                stage1_parameters,
                dict_config=dict_config,
            )
        return stage1_parameters
