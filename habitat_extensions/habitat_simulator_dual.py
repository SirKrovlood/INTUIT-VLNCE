from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSim,
    overwrite_config,
    HabitatSimVizSensors
)


from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from gym import spaces
from gym.spaces.box import Box
from numpy import ndarray

if TYPE_CHECKING:
    from torch import Tensor

import habitat_sim
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    BumpSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space

@registry.register_simulator(name="Sim-dual")
class HabitatSimDual(HabitatSim):
    def __init__(self, config: Config) -> None:
        #print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        #print("config")
        #print(config)
        self.habitat_config = config
        agent_config = self._get_agent_config()
        #print("mmmmmmmmmmmmmmmmmmmmmmagent_config")
        #print(agent_config)
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.habitat_config, sensor_name)
            #print("sensor_cfg.TYPE", sensor_cfg.TYPE)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)
            #print("sensor_type", sensor_type)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)

        self.sim_config = self.create_sim_config(self._sensor_suite)
        #print("sim_config")
        #print(self.sim_config)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        super(HabitatSim, self).__init__(self.sim_config)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs: Optional[Observations] = None
        #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.habitat_config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.habitat_config.SCENE
        agent_config = habitat_sim.AgentConfiguration()
        #print("prev agent_config")
        #print(agent_config)
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
            },
        )
        #print("new agent_config")
        #print(agent_config)
        sensor_specifications = []

        #propList = ["channels", "encoding", "gpu2gpu_transfer", "noise_model", "noise_model_kwargs", "observation_space", "orientation", "parameters", "position", "resolution", "sensor_subtype","sensor_type", "uuid"]
        for sensor in _sensor_suite.sensors.values():
            #print("llllllllllllllllllllllllllllllllllllllllllllllll")
            sim_sensor_cfg = habitat_sim.SensorSpec()
            #print("prev sim_sensor_cfg")
            #print(sim_sensor_cfg)
            #for prp in propList:
            #    print(prp, getattr(sim_sensor_cfg, prp))
            # TODO Handle configs for custom VisualSensors that might need
            # their own ignore_keys. Maybe with special key / checking
            # SensorType
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys={
                    "height",
                    "hfov",
                    "max_depth",
                    "min_depth",
                    "normalize_depth",
                    "type",
                    "width",
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sensor = cast(HabitatSimVizSensors, sensor)
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.habitat_config.HABITAT_SIM_V0.GPU_GPU
            )
            #print("late sim_sensor_cfg")
            #print(sim_sensor_cfg)
            #for prp in propList:
            #    print(prp, getattr(sim_sensor_cfg, prp))
            #print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.habitat_config.ACTION_SPACE_CONFIG
        )(self.habitat_config).get()
        #print("even newer agent_config")
        #print(agent_config)

        #NEW ADDITION
        agent_config2 = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(1),
            config_to=agent_config2,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
            },
        )

        sensor_specifications2 = habitat_sim.SensorSpec()
        #setattr(sensor_specifications2, "channels", 0)
        #setattr(sensor_specifications2, "sensor_type", "habitat_sim.SensorType.NONE")
        #setattr(sensor_specifications2, "uuid", "bump")
        sensor_specifications2.gpu2gpu_transfer = (
            self.habitat_config.HABITAT_SIM_V0.GPU_GPU
        )
        #agent_config2.sensor_specifications = sensor_specifications2
        agent_config2.sensor_specifications = []

        agent_config2.action_space = registry.get_action_space_configuration(
            self.habitat_config.ACTION_SPACE_CONFIG
        )(self.habitat_config).get()

        #END NEW ADDITION

        return habitat_sim.Configuration(sim_config, [agent_config, agent_config2])

    def step(self, action: Union[str, int]) -> Observations:
        if isinstance(action, int):
            action = {0: action}
        sim_obs = super(HabitatSim, self).step(action)

        self._prev_sim_obs = sim_obs

        if 0 in sim_obs.keys():
            observations = self._sensor_suite.get_observations(sim_obs[0])
            return observations

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_name = self.habitat_config.AGENTS[agent_id]
        #print('agent_name')
        #print(agent_name)
        agent_config = getattr(self.habitat_config, agent_name)
        #print('agent_config')
        #print(agent_config)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        #assert agent_id == 0, "No support of multi agent in {} yet.".format(
        #    self.__class__.__name__
        #)
        return self.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        #print("Now What")
        #print("agent_id", agent_id)
        #print("position", position)
        #print("rotation", rotation)
        agent = self.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True
