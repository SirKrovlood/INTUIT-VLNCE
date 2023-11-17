import numpy as np
import habitat

from vlnce_baselines.common.environments import VLNCEDaggerIntuitionEnv, VLNCEDaggerEnv
import gzip
import json
from vlnce_baselines.config.default import get_config

config = get_config(
        config_paths="vlnce_baselines/config/paper_configs/law_pano_config/cma_pm_aug.yaml"
    )
print(config)
with VLNCEDaggerEnv(config=config) as env:
    print("Environment creation successful")
    env.reset()
    action = {"action": 1}
    done = False
    #print(env.step)
    while not done:
        observations, _, done, _ = env.step(action=action)
        print(observations["vln_law_action_sensor"][0])
        action = {"action": observations["vln_law_action_sensor"][0]}
