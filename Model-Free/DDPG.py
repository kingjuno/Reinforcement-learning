import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from commons.buffers import replay_memory
from tqdm import tqdm

warnings.filterwarnings("ignore")


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 else np.zeros_like(self.mu)

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return torch.tensor(x).cuda()


def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def ddpg_optimization():
    batches = buffer.replay(batch_size)
    states, actions, next_states, rewards, done = batches
    states = torch.tensor(states, dtype=torch.float32).cuda()
    next_states = torch.tensor(next_states, dtype=torch.float32).cuda()
    rewards = torch.FloatTensor(rewards).unsqueeze(1).cuda()
    actions = torch.tensor(actions, dtype=torch.float32).cuda()
    done = torch.FloatTensor(done).unsqueeze(1).cuda()
    optimC.zero_grad()
    with torch.no_grad():
        target_actions = target_actor(next_states)
        target_values = target_crit(next_states, target_actions)
        y = rewards + gamma * (1 - done) * target_values

    critic_loss = torch.nn.functional.mse_loss(crit(states, actions), y.detach())
    critic_loss.backward()
    optimC.step()

    optimA.zero_grad()
    actor_loss = -crit(states, actor(states)).mean()
    actor_loss.backward()
    optimA.step()
    soft_update(target_actor, actor)
    soft_update(target_crit, crit)


env = gym.make("Pendulum-v1")
num_actions = env.action_space.shape[-1]
num_in = env.observation_space.shape[0]
observation, _ = env.reset()
buffer = replay_memory(100000)
crit = Critic(num_in, num_actions, 1).cuda()
actor = Actor(num_in, num_actions, 1).cuda()
optimC = torch.optim.Adam(crit.parameters(), lr=1e-3)
optimA = torch.optim.Adam(actor.parameters(), lr=1e-4)
target_crit = Critic(num_in, num_actions, 1).cuda()
target_actor = Actor(num_in, num_actions, 1).cuda()
noise = OrnsteinUhlenbeckNoise(
    mu=np.zeros(num_actions), sigma=float(0.2) * np.ones(num_actions)
)
print(env.action_space, env.observation_space)
episodes = 10000
batch_size = 32
gamma = 0.9
all_rewards = []
for i in tqdm(range(episodes)):
    noise.reset()
    state, _ = env.reset()
    done = False
    rewards = 0
    steps = 0
    while True:
        with torch.no_grad():
            action = actor(torch.tensor(state, dtype=torch.float32).cuda())
            action += noise()
            action = np.clip(action.cpu().numpy(), -3, 3)
        next_state, reward, done, _, _ = env.step(action)
        buffer.store(state, action, next_state, reward, done)
        state = next_state
        rewards += reward
        steps += 1
        if len(buffer) >= batch_size:
            ddpg_optimization()

        if done or (steps >= 1000):
            all_rewards.append(rewards)
            print(sum(all_rewards) / len(all_rewards), rewards)
            break
