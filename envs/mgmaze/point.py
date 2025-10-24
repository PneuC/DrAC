"""A point mass environment with Gymnasium API for the Gymnasium-Robotics PointMaze environments.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""
import time
from os import path
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


class PointEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, xml_file: Optional[str] = None, **kwargs):

        if xml_file is None:
            xml_file = path.join(
                path.dirname(path.realpath(__file__)), "./assets/point.xml"
            )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        super().__init__(
            model_path=xml_file,
            frame_skip=2,
            observation_space=observation_space,
            **kwargs,
        )

    def reset_model(self) -> np.ndarray:
        self.set_state(self.init_qpos, self.init_qvel)
        obs, _ = self._get_obs()

        return obs

    def step(self, action):
        # action = np.clip(action, -1.0, 1.0)
        # real_action = euc_clip_box(action, 1.0)
        # Implement fixed power dynamics
        self.do_simulation(np.clip(action, -1.0, 1.0), self.frame_skip)
        obs, info = self._get_obs()
        self._clip_velocity()
        # This environment class has no intrinsic task, thus episodes don't end and there is no reward
        terminated = False
        for contact in self.data.contact[:self.data.ncon]:
            name1, name2 = self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name
            if name1 == 'particle_geom' and 'obstacle' in name2 or name2 == 'particle_geom' and 'obstacle' in name1:
                # print(contact.dist)
                terminated = contact.dist < 0.0
        if self.render_mode == "human":
            self.render()

        return obs, 0, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel(), {}

    def _clip_velocity(self):
        """The velocity needs to be limited because the ball is
        force actuated and the velocity can grow unbounded."""

        m = np.abs(self.data.qvel).max()
        if m > 5:
            qvel = self.data.qvel * 5 / m
            self.set_state(self.data.qpos, qvel)


