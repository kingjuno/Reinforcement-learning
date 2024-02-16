import copy
import random

import gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from commons.buffers import PrioritizedReplayMemory
from commons.layers import NoisyLinear
from torch import nn


class RainbowDQN(nn.Module):
    def __init__(self):
        super(RainbowDQN, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.state = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, num_atoms),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, 2 * num_atoms),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.l(x)
        s = self.state(x)
        a = self.advantage(x)
        s = s.reshape(batch_size, 1, num_atoms)
        a = a.reshape(batch_size, 2, num_atoms)
        q = s + a - a.mean(dim=1, keepdim=True)
        q = torch.nn.functional.softmax(q, dim=2)
        return q

    def act(self, x, epsilon):
        with torch.no_grad():
            if random.random() < epsilon:
                return np.random.randint(2)
            else:
                x = torch.tensor(x).float().unsqueeze(0).to(device)
                dist = self.forward(x)
                dist = dist.mul(torch.linspace(Vmin, Vmax, num_atoms).to(device))
                action = dist.sum(2).max(1)[1][0].item()
                return action

    def reset_noise(self):
        self.state[0].reset_noise()
        self.state[-1].reset_noise()
        self.advantage[0].reset_noise()
        self.advantage[-1].reset_noise()


def projection_distribution(next_state, rewards, dones):
    batch_size = next_state.size(0)
    delta_z = (Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms).to(device)
    next_dist = net(next_state) * support
    next_act = next_dist.sum(2).max(1)[1]
    target.reset_noise()
    next_dist = target(next_state) * support
    next_dist = next_dist[torch.arange(next_dist.size(0)), next_act.data]

    rewards = rewards.view(-1, 1).expand([batch_size, num_atoms])
    dones = dones.view(-1, 1).expand([batch_size, num_atoms])

    Tz = rewards + gamma * (1 - dones) * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    bj = (Tz - Vmin) / delta_z
    l = torch.floor(bj).long()
    u = torch.ceil(bj).long()

    probl = next_dist * (u - bj)
    probu = next_dist * (bj - l)

    m = torch.zeros(batch_size, num_atoms).to(device)
    m.scatter_add_(1, l, probl)
    m.scatter_add_(1, u, probu)
    return m


def gradient_descent():
    batches, idx, weights = buffer.replay(size=batch_size)
    states, actions, rewards, next_states, done = zip(*batches)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    with torch.no_grad():
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    actions = torch.LongTensor(actions).to(device)
    done = torch.FloatTensor(done).to(device)
    weights = torch.FloatTensor(weights).to(device)

    # projection distribution
    pro_dist = projection_distribution(next_states, rewards, done)

    dist = net(states)
    dist = dist[torch.arange(len(actions)), actions]

    td_error = -(pro_dist * dist.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)
    loss = (td_error * weights).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    net.reset_noise()

    buffer.set_priority(td_error.detach().cpu().numpy(), idx)


no_of_frames = 10000
_e1, _e2, _ed = 1.0, 0.01, 500
eps = lambda x: _e2 + (_e1 - _e2) * np.exp(-x / _ed)
batch_size = 32
gamma = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Vmin, Vmax = -10, 10
num_atoms = 51

env_id = "CartPole-v1"
env = gym.make(env_id)
net = RainbowDQN().to(device)
target = copy.deepcopy(net).to(device)
buffer = PrioritizedReplayMemory(1000, alpha=0.8, beta=1)
optimizer = torch.optim.Adam(net.parameters())

rew = 0
rewards = []
state, _ = env.reset()
for i in tqdm.tqdm(range(1, 1 + no_of_frames)):
    action = net.act(state, eps(i))
    new_state, reward, done, _, _ = env.step(action)
    buffer.store(state, action, reward, new_state, float(done))
    rew += reward
    state = new_state
    if len(buffer) > 100:
        gradient_descent()
    if i % 100 == 0:
        net.reset_noise()
        target = copy.deepcopy(net)
    if done:
        rewards.append(rew)
        state, _ = env.reset()
        mean_reward = sum(rewards) / len(rewards)
        rew = 0
        tqdm.tqdm.write(f"\rMean Reward: {mean_reward} Max Reward: {max(rewards)}")


plt.plot(rewards)
plt.show()
print(f"Maximum Reward: {max(rewards)}")
