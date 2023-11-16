import numpy as np
import habitat

from vlnce_baselines.common.environments import VLNCEDaggerIntuitionEnv
import gzip
import json
from vlnce_baselines.config.default import get_config

config = get_config(
        config_paths="vlnce_baselines/config/paper_configs/intuition_config/cma_pm_aug.yaml"
    )
print(config)
with VLNCEDaggerIntuitionEnv(config=config) as env:
    print("Environment creation successful")
    env.reset()
    action = {"action": 1}
    env.step(action=action)
