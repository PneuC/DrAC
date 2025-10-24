"""A maze environment with the Gymnasium Ant agent (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v4.py).

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve reusing the code in Gymnasium for the Ant environment and in `point_maze/maze_env.py`.
The new code also follows the Gymnasium API and Multi-goal API

This project is covered by the Apache 2.0 License.
"""

import sys
from os import path
from typing import Dict, List, Optional, Union
import numpy as np
import gymnasium
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.utils import EzPickle

from envs.mgmaze.maps import get_map
from envs.mgmaze.maze import MultiGoalMaze
from envs.mgmaze.utils.mujoco_utils import MujocoModelNames


class MultiGoalAntMaze(MultiGoalMaze, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        maze_map: List[List[Union[str, int]]] = 'simple',
        reward_type: str = "dense",
        continuing_task: bool = True,
        reset_target: bool = False,
        **kwargs,
    ):
        # Get the ant.xml path from the Gymnasium package
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )
        super().__init__(
            agent_xml_path=ant_xml_file_path,
            maze_map=get_map(maze_map),
            maze_height=2.0,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        # Create the MuJoCo environment, include position observation of the Ant for GoalEnv
        self.ant_env = AntEnv(
            xml_file=self.tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            reset_noise_scale=0.0,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.ant_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]
        self.pos = self.reset_pos.copy()
        self.counter = 0
        self.num_goals = len(self.maze.unique_goal_locations)

        self.action_space = self.ant_env.action_space
        self.observation_space = self.ant_env.observation_space
        self.render_mode = render_mode
        EzPickle.__init__(
            self,
            render_mode,
            maze_map,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        self.ant_env.init_qpos[:2] = self.reset_pos

        obs, info = self.ant_env.reset(seed=seed)
        info.update(target=0, pos=self.ant_env.data.qpos[:2].copy())
        self.counter = 0
        self.pos = self.reset_pos.copy()
        return obs, info

    def step(self, action):
        ant_obs, _, _, _, info = self.ant_env.step(action)
        pos_after = ant_obs[:2]

        reward = self.compute_reward(self.pos, pos_after)
        terminated, reached_goal = self.compute_terminated(pos_after)
        info.update(target=reached_goal, pos=self.pos.copy())
        self.counter += 1
        self.pos = pos_after
        if self.render_mode == "human":
            self.render()

        return ant_obs, reward, terminated, self.counter >= self.max_steps, info

    def plot(self, ax):
        return self.maze.plot(ax)

    def render(self):
        return self.ant_env.render()

    def close(self):
        super().close()
        self.ant_env.close()

    @property
    def model(self):
        return self.ant_env.model

    @property
    def data(self):
        return self.ant_env.data


def register_ant_maze():
    gymnasium.register('MultiGoalAntMaze', entry_point=MultiGoalAntMaze)
