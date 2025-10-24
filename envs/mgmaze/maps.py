"""A collection of maze map structures for the Gymnasium-Robotics PointMaze environments.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

RESET = R = "r"  # Initial Reset position of the agent
GOAL = G = "g"
COMBINED = C = "c"  # These cells can be selected as goal or reset locations
OUTER = O = "o"  # Out of range cells


SIMPLE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, G, 0, 0, 0, 0, 0, 0, 0, G, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, R, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, G, 0, 0, 0, 0, 0, 0, 0, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
MEDIUM = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, G, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, R, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, G, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

HARD = [
    [O, O, O, 1, 1, 1, 1, O, 1, 1, 1, 1, O, O, O],
    [O, O, O, 1, 0, G, 1, 1, 1, G, 0, 1, O, O, O],
    [O, O, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, O, O],
    [1, 1, 1, 0, 0, 2, 1, 0, 1, 2, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, G, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, G, 1],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [O, 1, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 1, O],
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, G, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, G, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 2, 1, 0, 1, 2, 0, 0, 1, 1, 1],
    [O, O, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, O, O],
    [O, O, O, 1, 0, G, 1, 1, 1, G, 0, 1, O, O, O],
    [O, O, O, 1, 1, 1, 1, O, 1, 1, 1, 1, O, O, O],
]

def get_map(map_):
    if type(map_) is not str:
        return map_
    match map_:
        case 'simple':
            return SIMPLE
        case 'medium':
            return MEDIUM
        case 'hard':
            return HARD
        case _:
            raise NotImplementedError(f'Unknown maze map: {map_}')


