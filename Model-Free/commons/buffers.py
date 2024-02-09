from collections import deque

import numpy as np
import random


class PrioritizedReplayMemory:
    def __init__(self, N, alpha, beta, method="proportional"):
        """
        N: size of buffer
        alpha: how much prioritization is used
        beta: how much importance sampling is used
        method: "proportional" or "ranked"
        """
        self.N = N
        self.buffer = deque(maxlen=N)
        self.priorities = deque(maxlen=N)
        self.alpha = alpha
        self.beta = beta
        self.method = method

    def store(self, state, action, reward, next_state, done):
        priority = max(self.priorities) if self.priorities else 1
        self.buffer.append([state, action, reward, next_state, done])
        self.priorities.append(priority)

    def get_priority(self):
        if self.method == "proportional":
            Pj = np.array(self.priorities) ** self.alpha
            sumPj = sum(Pj)
            Pj = Pj / sumPj
        elif self.method == "ranked":
            Pj = np.array(self.priorities)
            sorted_indices = np.argsort(Pj)[::-1]
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, len(Pj) + 1)
            Pj = 1 / ranks
        else:
            raise NotImplementedError
        return Pj

    def replay(self, size):
        Pj = self.get_priority()
        # sample transition
        idx = np.random.choice(len(self), size, p=Pj)
        samples = [self.buffer[i] for i in idx]
        _Pj = Pj[idx]
        wj = (self.N * _Pj) ** (-self.beta)
        wj = wj / max(wj)
        return samples, idx, wj

    def set_priority(self, delta, idx):
        for i, d in zip(idx, delta):
            self.priorities[i] = abs(d) + 1e-6

    def __len__(self):
        return len(self.buffer)


class replay_memory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state.tolist(), action, reward, next_state.tolist(), done])

    def replay(self, size):
        return random.sample(list(self.buffer), size)

    def __len__(self):
        return len(self.buffer)
