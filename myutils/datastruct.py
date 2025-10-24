import random
import numpy as np

# TODO: Replace recursive implementation with iterative implementation
# TODO: Replace with C++ class call

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
        self.maximum = 1.0

    # update to the root node
    def _propagate(self, idx, change):

        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent >= 1:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1
        self.maximum = max(self.maximum, p)

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] += change
        self._propagate(idx, change)
        self.maximum = max(self.maximum, p)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def uniform_get(self):
        idx = random.randrange(0, self.n_entries)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def clear(self):
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)
        self.n_entries = 0
        self.write = 0


class RingQueue:
    def __init__(self, capacity):
        self.main = []
        self.p = 0
        self.capacity = capacity

    def push(self, item):
        if len(self.main) < self.capacity:
            self.main.append(item)
        else:
            self.main[self.p] = item
        self.p = (self.p + 1) % self.capacity

    def rear(self):
        if len(self.main) < self.capacity:
            return self.main[-1]
        else:
            return self.main[(self.p - 1) % self.capacity]

    def front(self):
        if len(self.main) < self.capacity:
            return self.main[0]
        else:
            return self.main[self.p]

    def to_list(self):
        return self.main[self.p:] + self.main[:self.p]

    def clear(self):
        self.main.clear()
        self.p = 0

    def __len__(self):
        return len(self.main)

    def full(self):
        return len(self.main) == self.capacity

    def empty(self):
        return len(self.main) == 0

def deepget(data, *keys):
    res = data
    for k in keys:
        res = res[k]
    return res

def multiget(data, *keys):
    return tuple(data[k] for k in keys)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

