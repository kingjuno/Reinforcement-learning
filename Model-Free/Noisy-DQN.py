import math
import random
from collections import deque

import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn


class NoisyLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_std = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias_mean = nn.Parameter(torch.empty((out_features), **factory_kwargs))
        self.bias_std = nn.Parameter(torch.empty((out_features), **factory_kwargs))

        self.weight_noise = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.bias_noise = nn.Parameter(torch.empty((out_features), **factory_kwargs))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        range = math.sqrt(3 / self.in_features)
        self.weight_mean.data.uniform_(-range, range)
        self.weight_std.data.fill_(0.017)
        self.bias_mean.data.uniform_(-range, range)
        self.bias_std.data.fill_(0.017)

    def reset_noise(self):
        range = math.sqrt(1 / self.out_features)
        self.weight_noise.data.uniform_(-range, range)
        self.bias_noise.data.fill_(0.5 * range)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def forward(self, x):
        if self.training:
            w = self.weight_mean + self.weight_std.mul(self.weight_noise)
            b = self.bias_mean + self.bias_std.mul(self.bias_noise)
        else:
            w = self.weight_mean
            b = self.bias_mean
        return nn.functional.linear(x, w, b)


# D is a distribution over the replay, it can be uniform or implementing prioritised replay
class replay_memory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state.tolist(), action, reward, next_state.tolist(), done])

    def replay(self, size):
        return random.sample(list(self.buffer), size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, 2),
        )

    def forward(self, x):
        return self.l(x)

    def act(self, x, epsilon):
        with torch.no_grad():
            if random.random() < epsilon:
                return np.random.randint(2)
            else:
                x = torch.FloatTensor(x)
                q_values = self.forward(x)
                act = q_values.argmax()
                return act.item()

    def reset_noise(self):
        for layer in self.l:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


def gradient_descent():
    batches = buffer.replay(size=batch_size)
    states, actions, rewards, new_states, done = zip(*batches)
    states = torch.tensor(states)
    with torch.no_grad():
        new_states = torch.tensor(new_states)
    rewards = torch.FloatTensor(rewards)
    actions = torch.LongTensor(actions)
    done = torch.FloatTensor(done)
    q_values = net(states)
    q_value_next = net(new_states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    y = rewards + gamma * (1 - done) * q_value_next.max(1)[0]
    loss = (y - q_value).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    net.reset_noise()

env = gym.make("CartPole-v1")
net = DQN()
buffer = replay_memory(1000)
optimizer = torch.optim.Adam(net.parameters())


no_of_frames = 10000
_e1, _e2, _ed = 1.0, 0.01, 500
f = lambda x: _e2 + (_e1 - _e2) * np.exp(-x / _ed)
batch_size = 32
gamma = 0.99

rew = 0
rewards = []
state, _ = env.reset()
for i in tqdm.tqdm(range(1, 1 + no_of_frames), desc="Mean Reward: 0"):
    action = net.act(state, f(i))
    new_state, reward, done, _, _ = env.step(action)
    buffer.store(state, action, reward, new_state, float(done))
    rew += reward
    state = new_state
    if len(buffer) > batch_size:
        gradient_descent()
    if done:
        rewards.append(rew)
        state, _ = env.reset()
        mean_reward = sum(rewards) / len(rewards)
        rew = 0
        tqdm.tqdm.write(f"\rMean Reward: {mean_reward} Max Reward: {max(rewards)}")

plt.plot(rewards)
plt.show()
print(f"Maximum Reward: {max(rewards)}")
