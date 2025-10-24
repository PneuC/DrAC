"""A maze environment with Gymnasium API for the Gymnasium-Robotics PointMaze environments.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""
import collections
import math
import os
import time
import random
import tempfile
from itertools import combinations

import numpy as np
import itertools as itt
import gymnasium as gym
import xml.etree.ElementTree as ET
from os import path
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple
from matplotlib.patches import Rectangle
from envs.mgmaze.maps import COMBINED, GOAL, RESET, SIMPLE


def all_goals_connected_to_reset(map_grid):
    # Collect RESET and GOAL positions
    reset_positions = []
    goals = []
    for i, row in enumerate(map_grid):
        for j, cell in enumerate(row):
            if cell == RESET:
                reset_positions.append((i, j))
            elif cell == GOAL:
                goals.append((i, j))

    # Handle edge cases
    if not goals:
        return True  # No goals to check
    if not reset_positions:
        return False  # Goals exist but no resets

    # Multi-source BFS initialization
    visited = set(reset_positions)
    queue = collections.deque(reset_positions)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Perform BFS
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(map_grid) and 0 <= ny < len(map_grid[0]):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    # Verify all goals are reachable
    return all((g[0], g[1]) in visited for g in goals)

def add_random_obstacles(map_, n):
    h, w = len(map_), len(map_[0])
    valid = False
    while not valid:
        new_map = deepcopy(map_)
        for _ in range(n):
            i = random.randrange(1, h-1)
            j = random.randrange(1, w-1)
            while new_map[i][j] != 0:
                i = random.randrange(1, h - 1)
                j = random.randrange(1, w - 1)
            new_map[i][j] = 1
        valid = all_goals_connected_to_reset(new_map)
    return new_map

def rand_remove_goals(mp, n):
    if n <= 0:
        return mp
    new_map = deepcopy(mp)
    h, w = len(new_map), len(new_map[0])
    goal_locations = []
    for i, j in itt.product(range(h), range(w)):
        if mp[i][j] == GOAL:
            goal_locations.append((i, j))
    removed = random.sample(goal_locations, n)
    for i, j in removed:
        new_map[i][j] = 0
    return new_map

def get_removed_maps(map_, n):
    h, w = len(map_), len(map_[0])
    goal_locations = []
    for i, j in itt.product(range(h), range(w)):
        if map_[i][j] == GOAL:
            goal_locations.append((i, j))
    # num_goals = len(goal_locations)
    removed_maps = []
    for comb in combinations(goal_locations, n):
        new_map = deepcopy(map_)
        for i, j in comb:
            new_map[i][j] = 0
        removed_maps.append(new_map)
    return removed_maps


class Maze:
    r"""This class creates and holds information about the maze in the MuJoCo simulation.

    The accessible attributes are the following:
    - :attr:`maze_map` - The maze discrete data structure.
    - :attr:`maze_size_scaling` - The maze scaling for the continuous coordinates in the MuJoCo simulation.
    - :attr:`maze_height` - The height of the walls in the MuJoCo simulation.
    - :attr:`unique_goal_locations` - All the `(i,j)` possible cell indices for goal locations.
    - :attr:`unique_reset_locations` - All the `(i,j)` possible cell indices for agent initialization locations.
    - :attr:`combined_locations` - All the `(i,j)` possible cell indices for goal and agent initialization locations.
    - :attr:`map_length` - Maximum value of j cell index
    - :attr:`map_width` - Mazimum value of i cell index
    - :attr:`x_map_center` - The x coordinate of the map's center
    - :attr:`y_map_center` - The y coordinate of the map's center

    The Maze class also presents a method to convert from cell indices to `(x,y)` coordinates in the MuJoCo simulation:
    - :meth:`cell_rowcol_to_xy` - Convert from `(i,j)` to `(x,y)`

    ### Version History
    * v4: Refactor compute_terminated into a pure function compute_terminated and a new function update_goal which resets the goal position. Bug fix: missing maze_size_scaling factor added in generate_reset_pos() -- only affects AntMaze.
    * v3: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v2 & v1: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        maze_size_scaling: float,
        maze_height: float,
    ):

        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._maze_height = maze_height

        self._unique_goal_locations = []
        self._unique_reset_locations = []
        self._combined_locations = []

        # Get the center cell Cartesian position of the maze. This will be the origin
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = self.map_width / 2 * maze_size_scaling
        self._y_map_center = self.map_length / 2 * maze_size_scaling

    @property
    def maze_map(self) -> List[List[Union[str, int]]]:
        """Returns the list[list] data structure of the maze."""
        return self._maze_map

    @property
    def maze_size_scaling(self) -> float:
        """Returns the scaling value used to integrate the maze
        encoding in the MuJoCo simulation.
        """
        return self._maze_size_scaling

    @property
    def maze_height(self) -> float:
        """Returns the un-scaled height of the walls in the MuJoCo
        simulation.
        """
        return self._maze_height

    @property
    def unique_goal_locations(self) -> List[np.ndarray]:
        """Returns all the possible goal locations in discrete cell
        coordinates (i,j)
        """
        return self._unique_goal_locations

    @property
    def unique_reset_locations(self) -> List[np.ndarray]:
        """Returns all the possible reset locations for the agent in
        discrete cell coordinates (i,j)
        """
        return self._unique_reset_locations

    @property
    def combined_locations(self) -> List[np.ndarray]:
        """Returns all the possible goal/reset locations in discrete cell
        coordinates (i,j)
        """
        return self._combined_locations

    @property
    def map_length(self) -> int:
        """Returns the length of the maze in number of discrete vertical cells
        or number of rows i.
        """
        return self._map_length

    @property
    def map_width(self) -> int:
        """Returns the width of the maze in number of discrete horizontal cells
        or number of columns j.
        """
        return self._map_width

    @property
    def x_map_center(self) -> float:
        """Returns the x coordinate of the center of the maze in the MuJoCo simulation"""
        return self._x_map_center

    @property
    def y_map_center(self) -> float:
        """Returns the x coordinate of the center of the maze in the MuJoCo simulation"""
        return self._y_map_center

    def get_xlim(self):
        return self.map_width / 2 * self.maze_size_scaling

    def get_ylim(self):
        return self.map_length / 2 * self.maze_size_scaling

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        """Converts a cell index `(i,j)` to x and y coordinates in the MuJoCo simulation"""
        x = (rowcol_pos[1] + 0.5) * self.maze_size_scaling - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self.maze_size_scaling

        return np.array([x, y])

    def cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        """Converts a cell x and y coordinates to `(i,j)`"""
        i = math.floor((self.y_map_center - xy_pos[1]) / self.maze_size_scaling)
        j = math.floor((xy_pos[0] + self.x_map_center) / self.maze_size_scaling)
        return np.array([i, j])

    def plot(self, ax, eval_obstacles_mode=0):
        for i, j in itt.product(range(self._map_length), range(self.map_width)):
            x, y = self.cell_rowcol_to_xy(np.array([i, j]))
            if self.maze_map[i][j] == GOAL:
                color = 'limegreen'
            elif self.maze_map[i][j] == 1:
                color = 'gray'
            elif self.maze_map[i][j] == 2:
                if eval_obstacles_mode == 0:
                    continue
                elif eval_obstacles_mode == 1:
                    color = '#D0D0D0'
                else:
                    color = 'gray'
            else:
                continue
            s = self.maze_size_scaling
            patch = Rectangle((x-0.5*s, y-0.5*s), s, s, fill=True, linewidth=0, color=color)
            ax.add_patch(patch)
        x, y = self.unique_reset_locations[0]
        ax.scatter([x], [y], color='red', s=100)
        pass

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
        eval_obstacles=False,
        collision_body=None
    ):
        """Class method that returns an instance of Maze with a decoded maze information and the temporal
           path to the new MJCF (xml) file for the MuJoCo simulation.

        Args:
            agent_xml_path (str): the goal that was achieved during execution
            maze_map (list[list[str,int]]): the desired goal that we asked the agent to attempt to achieve
            maze_size_scaling (float): an info dictionary with additional information
            maze_height (float): an info dictionary with additional information

        Returns:
            Maze: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
            str: The xml temporal file to the new mjcf model with the included maze.
        """
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")
        contact = tree.find(".//contact")

        maze = cls(maze_map, maze_size_scaling, maze_height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze_size_scaling
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that maze is centered.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 1.0",
                    )

                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))
                elif struct == 2:
                    if eval_obstacles:
                        ET.SubElement(
                            worldbody,
                            "geom",
                            name=f"obstacle_{i}_{j}",
                            pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                            size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                            type="box",
                            material="",
                            contype="1",
                            conaffinity="1",
                            rgba="0.7 0.5 0.3 1.0",
                        )
                    else:
                        empty_locations.append(np.array([x, y]))
        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
            size=f"{0.2 * maze_size_scaling}",
            rgba="1 0 0 0.7",
            type="sphere",
        )

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            maze._combined_locations = empty_locations
        elif not maze._unique_reset_locations and not maze._combined_locations:
            # If there are no given "r" or "c" cells in the maze data structure,
            # any empty cell can be a reset location at initialization.
            maze._unique_reset_locations = empty_locations
        elif not maze._unique_goal_locations and not maze._combined_locations:
            # If there are no given "g" or "c" cells in the maze data structure,
            # any empty cell can be a gaol location at initialization.
            maze._unique_goal_locations = empty_locations

        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_name = f"ant_maze{str(time.time())}.xml"
            temp_xml_path = path.join(path.dirname(tmp_dir), temp_xml_name)
            tree.write(temp_xml_path)

        return maze, temp_xml_path

    # def get_space_size(self):
    #     return (self._map_width - 2) * (self._map_length - 2)


class MultiGoalMaze(gym.Env):
    def __init__(
        self,
        agent_xml_path: str,
        reward_type: str = "dense",
        maze_map: List[List[Union[int, str]]] = SIMPLE,
        maze_size_scaling: float = 1.0,
        maze_height: float = 2.5,
        # max_steps_ratio: int = 20,
        max_steps=500,
        eval_mode=False,
        **kwargs,
    ):

        self.reward_type = reward_type
        self.maze, self.tmp_xml_file_path = Maze.make_maze(
            agent_xml_path, maze_map, maze_size_scaling, maze_height, eval_mode
        )

        self.reset_pos = self.maze.unique_reset_locations[0]
        self.max_steps = max_steps

    def compute_reward(self, pos: np.ndarray, pos_after: np.ndarray) -> float:
        reward = 0.0
        for goal in self.maze.unique_goal_locations:
            if np.max(abs(pos_after - goal), axis=-1) <= 0.5 * self.maze.maze_size_scaling:
                return 100
        if self.reward_type == "dense":
            distance = min(np.linalg.vector_norm(pos - goal, axis=-1) for goal in self.maze.unique_goal_locations)
            distance_after = min(
                np.linalg.vector_norm(pos_after - goal, axis=-1) for goal in self.maze.unique_goal_locations)
            reward += (distance - distance_after) * 10
        return reward

    def compute_terminated(
        self, pos: np.ndarray
    ) -> Tuple[bool, int]:
        terminated, reached = False, 0
        for i, goal in enumerate(self.maze.unique_goal_locations):
            terminated = terminated or np.max(abs(pos - goal), axis=-1) <= 0.5 * self.maze.maze_size_scaling
            if terminated:
                reached = i + 1
                break
        return terminated, reached

    def close(self):
        os.remove(self.tmp_xml_file_path)
