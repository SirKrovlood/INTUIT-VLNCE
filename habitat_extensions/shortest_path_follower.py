#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import numpy as np

import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
)

EPSILON = 1e-6


def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class ShortestPathFollower:
    r"""Utility class for extracting the action on the shortest path to the
        goal.
    Args:
        sim: HabitatSim instance.
        goal_radius: Distance between the agent and the goal for it to be
            considered successful.
        return_one_hot: If true, returns a one-hot encoding of the action
            (useful for training ML agents). If false, returns the
            SimulatorAction.
    """

    def __init__(
        self, sim: HabitatSim, goal_radius: float, return_one_hot: bool = True
    ):
        assert (
            getattr(sim, "geodesic_distance", None) is not None
        ), "{} must have a method called geodesic_distance".format(
            type(sim).__name__
        )

        self._sim = sim
        self._max_delta = self._sim.habitat_config.FORWARD_STEP_SIZE - EPSILON
        self._goal_radius = goal_radius
        self._step_size = self._sim.habitat_config.FORWARD_STEP_SIZE

        self._mode = (
            "geodesic_path"
            if getattr(sim, "get_straight_shortest_path_points", None)
            is not None
            else "greedy"
        )
        self._return_one_hot = return_one_hot

    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action_for_waypoint(
        self, goal_pos: np.array, agent_pos: np.array, agent_rotation
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path to the waypoint.
        """

        #print('---in get_next_action_for_waypoint:agent_pos=', agent_pos,' ;;agent_rotation=',agent_rotation,'goal_pos=', goal_pos)
        """ if (
            self._sim.geodesic_distance(
                agent_pos, goal_pos
            )
            <= self._goal_radius
        ):
            return None """

        max_grad_dir = self._est_max_grad_dir_waypoint(goal_pos, agent_pos)
        if max_grad_dir is None:
            return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
        return self._step_along_grad_waypoint(max_grad_dir, agent_rotation)

    def _est_max_grad_dir_waypoint(self, goal_pos: np.array, agent_pos: np.array) -> np.array:

        points = self._sim.get_straight_shortest_path_points(
            agent_pos, goal_pos
        )
        # Add a little offset as things get weird if
        # points[1] - points[0] is anti-parallel with forward
        if len(points) < 2:
            return None
        max_grad_dir = quaternion_from_two_vectors(
            self._sim.forward_vector,
            points[1]
            - points[0]
            + EPSILON
            * np.cross(self._sim.up_vector, self._sim.forward_vector),
        )
        max_grad_dir.x = 0
        max_grad_dir = np.normalized(max_grad_dir)
        return max_grad_dir

    def _step_along_grad_waypoint(
        self, grad_dir: np.quaternion, agent_rotation, agent_id=0
    ) -> Union[int, np.array]:
        alpha = angle_between_quaternions(grad_dir, agent_rotation)
        if alpha <= np.deg2rad(self._sim.config.TURN_ANGLE) + EPSILON:
            return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
        else:
            sim_action = {agent_id: HabitatSimActions.TURN_LEFT}
            self._sim.step(sim_action)
            best_turn = (
                HabitatSimActions.TURN_LEFT
                if (
                    angle_between_quaternions(
                        grad_dir, self._sim.get_agent_state(agent_id).rotation
                    )
                    < alpha
                )
                else HabitatSimActions.TURN_RIGHT
            )
            return self._get_return_value(best_turn)

    def _get_agent_position(self, agent_id=0) -> np.array:
        return self._sim.get_agent_state(agent_id).position

    def get_next_action(
        self, goal_pos: np.array, agent_id=0
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path.
        """
        #print('---in get_next_action:', self._sim.get_agent_state(agent_id).position,' ;;goal_pos=', goal_pos)
        if (
            self._sim.geodesic_distance(
                self._sim.get_agent_state(agent_id).position, goal_pos
            )
            <= self._goal_radius
        ):
            return None

        max_grad_dir = self._est_max_grad_dir(goal_pos, agent_id)
        if max_grad_dir is None:
            return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
        return self._step_along_grad(max_grad_dir, agent_id)

    def _step_along_grad(
        self, grad_dir: np.quaternion, agent_id=0
    ) -> Union[int, np.array]:
        current_state = self._sim.get_agent_state(agent_id)
        alpha = angle_between_quaternions(grad_dir, current_state.rotation)
        if alpha <= np.deg2rad(self._sim.habitat_config.TURN_ANGLE) + EPSILON:
            return self._get_return_value(HabitatSimActions.MOVE_FORWARD)
        else:
            sim_action = {agent_id: HabitatSimActions.TURN_LEFT}
            self._sim.step(sim_action)
            best_turn = (
                HabitatSimActions.TURN_LEFT
                if (
                    angle_between_quaternions(
                        grad_dir, self._sim.get_agent_state(agent_id).rotation
                    )
                    < alpha
                )
                else HabitatSimActions.TURN_RIGHT
            )
            self._reset_agent_state(current_state, agent_id)
            return self._get_return_value(best_turn)

    def _reset_agent_state(self, state: habitat_sim.AgentState, agent_id=0) -> None:
        self._sim.set_agent_state(
            state.position, state.rotation, agent_id, reset_sensors=False
        )

    def _geo_dist(self, goal_pos: np.array, agent_id) -> float:
        return self._sim.geodesic_distance(
            self._sim.get_agent_state(agent_id).position, goal_pos
        )

    def _geo_dist_between_points(self, point1, point2) -> float:
        return self._sim.geodesic_distance(point1, point2)

    def _est_max_grad_dir(self, goal_pos: np.array, agent_id) -> np.array:

        current_state = self._sim.get_agent_state(agent_id)
        current_pos = current_state.position

        if self.mode == "geodesic_path":
            points = self._sim.get_straight_shortest_path_points(
                self._sim.get_agent_state(agent_id).position, goal_pos
            )
            # Add a little offset as things get weird if
            # points[1] - points[0] is anti-parallel with forward
            if len(points) < 2:
                return None
            max_grad_dir = quaternion_from_two_vectors(
                self._sim.forward_vector,
                points[1]
                - points[0]
                + EPSILON
                * np.cross(self._sim.up_vector, self._sim.forward_vector),
            )
            max_grad_dir.x = 0
            max_grad_dir = np.normalized(max_grad_dir)
        else:
            current_rotation = self._sim.get_agent_state(agent_id).rotation
            current_dist = self._geo_dist(goal_pos, agent_id)

            best_geodesic_delta = -2 * self._max_delta
            best_rotation = current_rotation
            for _ in range(0, 360, self._sim.config.TURN_ANGLE):
                sim_action = {agent_id: HabitatSimActions.MOVE_FORWARD}
                self._sim.step(sim_action)
                new_delta = current_dist - self._geo_dist(goal_pos, agent_id)

                if new_delta > best_geodesic_delta:
                    best_rotation = self._sim.get_agent_state(agent_id).rotation
                    best_geodesic_delta = new_delta

                # If the best delta is within (1 - cos(TURN_ANGLE))% of the
                # best delta (the step size), then we almost certainly have
                # found the max grad dir and should just exit
                if np.isclose(
                    best_geodesic_delta,
                    self._max_delta,
                    rtol=1 - np.cos(np.deg2rad(self._sim.config.TURN_ANGLE)),
                ):
                    break

                self._sim.set_agent_state(
                    current_pos,
                    self._sim.get_agent_state(agent_id).rotation,
                    agent_id,
                    reset_sensors=False,
                )

                sim_action = {agent_id: HabitatSimActions.TURN_LEFT}
                self._sim.step(sim_action)

            self._reset_agent_state(current_state, agent_id)

            max_grad_dir = best_rotation

        return max_grad_dir

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        r"""Sets the mode for how the greedy follower determines the best next
            step.
        Args:
            new_mode: geodesic_path indicates using the simulator's shortest
                path algorithm to find points on the map to navigate between.
                greedy indicates trying to move forward at all possible
                orientations and selecting the one which reduces the geodesic
                distance the most.
        """
        assert new_mode in {"geodesic_path", "greedy"}
        if new_mode == "geodesic_path":
            assert (
                getattr(self._sim, "get_straight_shortest_path_points", None)
                is not None
            )
        self._mode = new_mode
