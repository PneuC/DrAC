import torch
import numpy as np
from myutils.datastruct import SumTree


class ReplayMem:
    def __init__(self, do, da, capacity=int(1e6), device='cpu'):
        # self.queue = RingQueue(capacity)
        self.o_buffer = np.zeros((capacity, do), np.float32)
        self.op_buffer = np.zeros((capacity, do), np.float32)
        self.a_buffer = np.zeros((capacity, da), np.float32)
        self.r_buffer = np.zeros((capacity,), np.float32)
        self.d_buffer = np.zeros((capacity,), np.float32)
        self.device = device
        self.size = 0
        self.p = 0
        self.capacity = capacity
        # self.dtype = dtype

    def add(self, o, a, r, op, d):
        self.o_buffer[self.p] = o
        self.a_buffer[self.p] = a
        self.r_buffer[self.p] = r
        self.op_buffer[self.p] = op
        self.d_buffer[self.p] = float(d)
        self.p = (self.p + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, n):
        idxes = np.random.randint(0, self.size, size=n)
        data = (
            torch.as_tensor(self.o_buffer[idxes], device=self.device),
            torch.as_tensor(self.a_buffer[idxes], device=self.device),
            torch.as_tensor(self.r_buffer[idxes], device=self.device),
            torch.as_tensor(self.op_buffer[idxes], device=self.device),
            torch.as_tensor(self.d_buffer[idxes], device=self.device),
        )
        return data, None

    def __len__(self):
        return self.size

    def clear(self):
        self.size = 0


class PERMem:
    eps = 1e-5

    def __init__(self, capacity=int(1e6), alpha=0.6, beta_start=0.5, beta_final=1.0, dtype=torch.float32, device='cpu'):
        self.tree = SumTree(capacity)
        self.device = device
        self.dtype = dtype
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_span = beta_final - beta_start
        self.batch_to_update = None
        self.idxes_to_update = None
        self.apply_priority = True

    def _get_priority(self, error):
        return (error + self.eps) ** self.alpha

    def add(self, *transition):
        self.tree.add(self.tree.maximum, transition)

    def sample(self, n):
        transitions = []
        idxes = []
        priorities = []

        for s in np.random.uniform(0, self.tree.total(), [n]):
            if self.apply_priority:
                i, p, data = self.tree.get(s)
            else:
                i, p, data = self.tree.uniform_get()
            priorities.append(p)
            transitions.append(data)
            idxes.append(i)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        importance = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        importance /= importance.max()

        batch = tuple(self.__to_tensor([tran[i] for tran in transitions]) for i in range(5))
        self.idxes_to_update = idxes
        self.batch_to_update = batch
        return batch, self.__to_tensor(importance)

    def update(self, agent, complete_rate):
        errors = agent.compute_error(self.batch_to_update)
        for i, e in zip(self.idxes_to_update, errors):
            self.tree.update(i, self._get_priority(e))
        self.beta = self.beta_start + complete_rate * self.beta_span

    def clear(self):
        self.tree.clear()

    def __to_tensor(self, data):
        return torch.as_tensor(np.stack(data), dtype=self.dtype, device=self.device)

    def __len__(self):
        return self.tree.n_entries
