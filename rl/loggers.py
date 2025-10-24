import os
import csv
import numpy as np
from myutils.filesys import gp
from myutils.fmt import fmt_timespan


class CsvLogger:
    def __init__(self, target, keys, step_itv=1000, std_print=True):
        self.keys = keys
        self.buffer = {k: [] for k in keys}
        self.step_itv = step_itv
        self.horizon = step_itv
        self.ftarget = open(gp(target), 'w', newline='')
        self.writer = csv.writer(self.ftarget)
        self.writer.writerow(['steps', 'time', 'agent_time', *keys])
        self.std_print = std_print

    def update(self, step, time, agent_time, info):
        for key in self.keys:
            self.buffer[key].append(info[key])
        if step >= self.horizon:
            self.write(step, time, agent_time)
            self.horizon += self.step_itv

    def write(self, step, time, agent_time):
        line = {key: np.mean(self.buffer[key]) for key in self.keys}
        self.writer.writerow([step, time, agent_time, *(line[key] for key in self.keys)])
        self.ftarget.flush()
        if self.std_print:
            step_msg = f'Step: {step}'
            time_msg = f'Time: {fmt_timespan(time)}'
            rest = [f'{key}: ' + '%.4g' % line[key] for key in self.keys]
            print(', '.join([step_msg, time_msg, *rest]))
        for key in self.keys:
            self.buffer[key].clear()

    def close(self, step, time, agent_time):
        if any(len(self.buffer[key]) for key in self.keys):
            line = {key: np.mean(self.buffer[key]) for key in self.keys}
            self.writer.writerow([step, time, agent_time, *(line[key] for key in self.keys)])
        self.buffer.clear()
        self.ftarget.close()


class TrajectoryLogger:
    def __init__(self, folder, step_itv=1000):
        self.folder = folder
        self.step_itv = step_itv
        self.states = []
        self.final_states = []
        self.actions = []
        self.rewards = []
        self.end_points = []
        self.flush_horizon = step_itv
        self.target = gp(folder, 'trajectories')
        os.makedirs(self.target, exist_ok=True)

    def update(self, step, o, a, r, op, end):
        self.states.append(o)
        self.actions.append(a)
        self.rewards.append(r)
        if end:
            self.end_points.append(len(self.states))
            self.final_states.append(op)
        if step >= self.flush_horizon:
            self.flush()
            self.flush_horizon += self.step_itv

    def flush(self):
        np.save(f'{self.target}/states.npy', self.states)
        np.save(f'{self.target}/actions.npy', self.actions)
        np.save(f'{self.target}/rewards.npy', self.rewards)
        np.save(f'{self.target}/end_points.npy', self.end_points)
        np.save(f'{self.target}/final_states.npy', self.final_states)

    def close(self):
        self.flush()

    @staticmethod
    def load_trajectories(path, add_final_state=True):
        flatten_states = np.load(f'{path}/states.npy')
        flatten_actions = np.load(f'{path}/actions.npy')
        flatten_rewards = np.load(f'{path}/rewards.npy')
        final_states = np.load(f'{path}/final_states.npy')
        end_points = np.load(f'{path}/end_points.npy')
        s = 0
        states, actions, rewards = [], [], []
        for i, e in enumerate(end_points):
            n, d = e-s, flatten_states.shape[-1]
            if add_final_state:
                traj = np.zeros([n+1, d])
                traj[:n] = flatten_states[s:e]
                traj[-1] = final_states[i]
            else:
                traj = flatten_states[s:e]
            states.append(traj)
            actions.append(flatten_actions[s:e])
            rewards.append(flatten_rewards[s:e])
            s = e
        return states, actions, rewards

