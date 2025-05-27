import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from reinvent.utils import config_parse, setup_logger

from reinvent.Reinvent import setup_responder
from run_staged_learning_2 import run_staged_learning

from navidiv.reinvent.InputGenerator import InputGenerator

LOGLEVEL_CHOICES = tuple(
    level.lower() for level in logging._nameToLevel.keys()
)

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="/media/mohammed/Work/OPT_MOL_GEN/cases/SF/using_new_RL/conf_folder",
    config_name="base",
    version_base="1.1",  # Explicitly specify the compatibility version
)
def main(cfg: DictConfig):
    
    input_generator = InputGenerator(cfg)
    print(cfg.diversity_scorer)
    os.chdir(input_generator.wd)
    input_generator.generate_input()
    input_config = config_parse.read_toml(
        os.path.join(input_generator.wd, "stage1.toml"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(name="reinvent", filename="stage1.log")
    device = torch.device(device)
    seed = input_config.get("seed", None)
    tb_logdir = input_config.get("tb_logdir", None)

    if tb_logdir:
        tb_logdir = os.path.abspath(tb_logdir)
        logger.info(f"Writing TensorBoard summary to {tb_logdir}")


    run_staged_learning(
        extract_sections(input_config),
        device,
        tb_logdir=tb_logdir,
        write_config=None,
        responder_config=False,
        seed=seed,
        diversity_scorer=cfg.diversity_scorer,
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
