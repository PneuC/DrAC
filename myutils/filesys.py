import csv
import json
import os
import re
import pandas as pds
import yaml
from root import PRJROOT


def auto_dire(path=None, name='trial'):
    dire_id = 0
    folder = PRJROOT if path is None else gp(path)
    tar = os.path.join(folder, f'{name}{dire_id}')
    while os.path.exists(tar):
        dire_id += 1
        tar = os.path.join(folder, f'{name}{dire_id}')
    os.makedirs(tar)
    return tar

def gp(*args):
    """ if is absolute root_folder or related path, return {path}, else return {PRJROOT/path} """
    path = os.path.join(*args)
    if os.path.isabs(path) or re.match(r'\.+[\\/].*', path):
        return path
    else:
        return os.path.join(PRJROOT, path)

def load_json(path):
    with open(gp(path), 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(gp(path), 'w') as f:
        json.dump(data, f)

def load_yaml(path):
    with open(gp(path), 'r') as f:
        return yaml.unsafe_load(f)

def save_yaml(data, path):
    with open(gp(path), 'w') as f:
        yaml.dump(data, f)

def save_csv(path, header, content):
    with open(gp(path), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(content)
    pass


if __name__ == '__main__':
    # config = load_yaml('data/Ant/SwitchPTSAC1/trial0/config.yaml')  # test loading yaml file
    # print(config)
    pass
    save_csv('test.csv', ['A', 'B', 'C'], [[1, 2, 3], [4, 5, 6]])
