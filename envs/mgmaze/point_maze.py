"""A point mass maze environment with Gymnasium API.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

from os import path
from typing import Dict, List, Optional, Union

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from envs.mgmaze.maps import get_map
from envs.mgmaze.maze import MultiGoalMaze
from envs.mgmaze.point import PointEnv
from envs.mgmaze.utils.mujoco_utils import MujocoModelNames


class MultiGoalPointMaze(MultiGoalMaze, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]] = 'simple',
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        # max_steps_ratio=10,
        maze_eval_mode=False,
        **kwargs,
    ):
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "./assets/point.xml"
        )
        match maze_map:
            case 'simple':
                max_steps = 150
            case 'medium':
                max_steps = 300
            case 'hard':
                max_steps = 600
            case _:
                max_steps = 500
        super().__init__(
            agent_xml_path=point_xml_file_path,
            maze_map=get_map(maze_map),
            maze_size_scaling=1.0,
            maze_height=2.0,
            reward_type=reward_type,
            # max_steps_ratio=max_steps_ratio,
            max_steps=max_steps,
            eval_mode=maze_eval_mode
        )

        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        x_lim, y_lim = self.maze.get_xlim(), self.maze.get_ylim()
        low = np.array([-x_lim, -y_lim, -5., -5.])
        high = np.array([x_lim, y_lim, 5., 5.])
        self.observation_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float64)

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            **kwargs,
        )
        self.pos = self.reset_pos.copy()
        self.counter = 0
        self.num_goals = len(self.maze.unique_goal_locations)

    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed, options=None)
        self.point_env.init_qpos[:2] = self.reset_pos
        self.pos = self.reset_pos.copy()

        obs, info = self.point_env.reset(seed=seed)
        # info['target'] = 0
        info.update(target=0, pos=self.pos.copy())
        self.counter = 0
        return obs, info

    def step(self, action):
        obs, _, terminated1, _, info = self.point_env.step(action)
        pos_after = obs[:2]

        reward = self.compute_reward(self.pos, pos_after)
        terminated2, reached_goal = self.compute_terminated(pos_after)
        # Update the goal position if necessary
        # info['target'] = reached_goal
        info.update(target=reached_goal, pos=self.pos.copy())
        self.counter += 1
        self.pos = pos_after
        return obs, reward, terminated1 or terminated2, self.counter >= self.max_steps, info

    def plot(self, ax, eval_obstacles_mode=0):
        return self.maze.plot(ax, eval_obstacles_mode)

    def render(self):
        return self.point_env.render()

    def close(self):
        super().close()
        self.point_env.close()

    @property
    def model(self):
        return self.point_env.model

    @property
    def data(self):
        return self.point_env.data


def register_point_maze():
    gymnasium.register('MultiGoalPointMaze', entry_point=MultiGoalPointMaze)


if __name__ == '__main__':
    env = MultiGoalPointMaze()
    # print(env.maze.unique_goal_locations)
    env.reset()
    for stp in range(100):
        o, r, d, t, info = env.step(np.array([1.0, 1.0]))
        print(env.pos, env.point_env.data.qvel)
        if d or t:
            print(info['target'], o[:2], d, t, stp)
            break
    env.reset()
    print('-' * 64)
    for stp in range(100):
        o, r, d, t, info = env.step(np.array([1.0, 0.0]))
        print(env.pos, env.point_env.data.qvel)
        if d or t:
            print(info['target'], o[:2], d, t, stp)
            break
    env.reset()
    print('-' * 64)
    for stp in range(100):
        o, r, d, t, info = env.step(np.array([-1.0, 0.3]))
        print(env.pos, env.point_env.data.qvel)
        if d or t:
            print(info['target'], o[:2], d, t, stp)
            break
    env.reset()
    env.close()
    pass
