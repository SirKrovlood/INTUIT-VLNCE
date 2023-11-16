from typing import Optional

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat.utils import profiling_wrapper
import pickle

@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(current_position, target_position)
        return distance

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        return self._env.episode_over

    def get_info(self, observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }

@baseline_registry.register_env(name="VLNCEDaggerIntuitionEnv")
class VLNCEDaggerIntuitionEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):

        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        self.intuition_steps = config.MODEL.INTUITION_STEPS

        self.goal_radius = config.TASK_CONFIG.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR.GOAL_RADIUS
        self.main_agent_target = 0

        super().__init__(config.TASK_CONFIG, dataset)


    def get_reward_range(self):
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(current_position, target_position)
        return distance

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    @profiling_wrapper.RangeContext("VLNCEDaggerIntuitionEnv.reset")
    def reset(self):
        self.main_agent_target = 0
        return self._env.reset()

    '''
    episoder attributes
    'episode_id', 'scene_id', 'start_position',
    'start_rotation', 'info', '_shortest_path_cache',
    'start_room', 'shortest_paths', 'instruction',
    'trajectory_id', 'instruction_index_string',
    'goals', 'reference_path',
    '__module__',
    '__annotations__', '__doc__', '__attrs_attrs__',
    '__repr__', '__eq__', '__ne__', '__lt__',
    '__le__', '__gt__', '__ge__', '__hash__',
    '__init__', '__getstate__', '__setstate__',
    '__dict__', '__weakref__', '__str__',
    '__getattribute__', '__setattr__',
    '__delattr__', '__new__',
    '__reduce_ex__', '__reduce__',
    '__subclasshook__', '__init_subclass__',
    '__format__', '__sizeof__', '__dir__', '__class__'
    '''
    @profiling_wrapper.RangeContext("VLNCEDaggerIntuitionEnv.step")
    def step(self, *args, **kwargs):

        #print("eps", self._env.current_episode.goals[0].position)
        #print("eps info", self._env.current_episode.info)
        #print("eps.__dir__()", self._env.current_episode.__dir__())

        #print("kwargs", kwargs["action"])
        incoming_ac = kwargs["action"]["action"]
        #print("incoming_ac", incoming_ac)
        self._env.sim.get_agent(1).state = self._env.sim.get_agent(
            0).state

        corrected_actions = np.zeros(self.intuition_steps, dtype=np.float16)
        geist_ac = (
            self._env.task.sensor_suite.sensors[
                "vln_law_action_sensor"].get_observation(None,
                                                         episode=self._env.current_episode,
                                                         agent_id=1))[0]

        #print("geist_ac", geist_ac)
        for i in range(self.intuition_steps):
            observations = self._env.step({1: geist_ac})
            geist_ac = observations["vln_law_action_sensor"][0]
            if geist_ac == 0:
                break
            corrected_actions[i] = geist_ac

        print(corrected_actions)
        observations = self._env.step({0: incoming_ac})
        #print(self.current_episode.reference_path, flush=True)

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        #print("step observations", observations)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")

        return observations, reward, done, info
