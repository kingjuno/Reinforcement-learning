import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn


class replay_memory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state.tolist(), action, reward, next_state.tolist(), done])

    def replay(self, size):
        return random.sample(list(self.buffer), size)

    def __len__(self):
        return len(self.buffer)


class DuelDQN(nn.Module):
    def __init__(self):
        super(DuelDQN, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.state = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.l(x)
        s = self.state(x)
        a = self.advantage(x)
        q = s + a - a.mean()
        return q

    def act(self, x, epsilon):
        with torch.no_grad():
            if random.random() < epsilon:
                return np.random.randint(2)
            else:
                x = torch.FloatTensor(x)
                q_values = self.forward(x)
                act = q_values.argmax()
                return act.item()


def gradient_descent():
    batches, idx, weights = buffer.replay(size=batch_size)
    states, actions, rewards, new_states, done = zip(*batches)
    states = torch.tensor(states)
    with torch.no_grad():
        new_states = torch.tensor(new_states)
    rewards = torch.FloatTensor(rewards)
    actions = torch.LongTensor(actions)
    done = torch.FloatTensor(done)
    weights = torch.FloatTensor(weights)
    q_values = net(states)
    q_value_next = net(new_states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    y = rewards + gamma * (1 - done) * q_value_next.max(1)[0]
    td_error = torch.abs(y - q_value).detach()
    loss = (weights * (y - q_value) ** 2).mean()
    buffer.set_priority(td_error, idx)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
    gradient = (y - q_value).pow(2).mean()
    optimizer.zero_grad()
    gradient.backward()
    optimizer.step()


env = gym.make("CartPole-v1")
net = DuelDQN()
buffer = replay_memory(1000)
optimizer = torch.optim.Adam(net.parameters())


episodes = 10000
_e1, _e2, _ed = 1.0, 0.01, 500
f = lambda x: _e2 + (_e1 - _e2) * np.exp(-x / _ed)
batch_size = 32
gamma = 0.99

rew = 0
rewards = []
state, _ = env.reset()
for i in tqdm.tqdm(range(1, 1 + episodes), desc="Mean Reward: 0"):
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
