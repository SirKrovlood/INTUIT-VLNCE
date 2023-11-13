from habitat.tasks.nav.nav import NavigationTask
from habitat.core.registry import registry

import numpy as np

from typing import Any, Dict, Iterable, List, Optional, Union
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)

@registry.register_task(name="Nav-dual")
class NavigationTask(NavigationTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def step(self, action: Dict[str, Any], episode: Episode):
        #print("HEYHEHE", action)
        if "action" in action.keys():
            if "action_args" not in action or action["action_args"] is None:
                action["action_args"] = {}
            action_name = action["action"]
            if isinstance(action_name, (int, np.integer)):
                action_name = self.get_action_name(action_name)
            assert (
                action_name in self.actions
            ), f"Can't find '{action_name}' action in {self.actions.keys()}."
            task_action = self.actions[action_name]
            observations = task_action.step(**action["action_args"], task=self)

        else:
            observations = self._sim.step(action)

        #print("observations", observations)
        if observations is None:
            observations = {}
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations
