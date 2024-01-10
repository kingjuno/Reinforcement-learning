# Deep Q Learning

# Index

1. [List of Algorithms](#list-of-algorithms)
2. [Algorithms](#algorithms)
   - [DQN](#dqn)
   - [DDQN](#ddqn)
   - [PER](#per)
   - [Dueling DQN](#dueling-dqn)
3. [References](#references)

# List of Algorithms

- [x] DQN
- [ ] DRQN
- [x] Double DQN (DDQN)
- [x] Dueling DQN
- [x] Prioritized Experience Replay (PER)
- [ ] Rainbow DQN

# Algorithms

## DQN

![Alt text](../assets/DQN.png)

1. Initialize replay memory D to capacity N

   ```py
   class replay_memory:
       def __init__(self, capacity):
           self.buffer = deque(maxlen=capacity)

       def store(self, state, action, reward, next_state, done):
           self.buffer.append([state.tolist(), action, reward, next_state.tolist(), done])

       def replay(self, size):
           return random.sample(list(self.buffer), size)

       def __len__(self):
           return len(self.buffer)
   ```

2. Initialize action-value function Q with random weights

   ```py
    class QNet(nn.Module):
        def __init__(self):
            super(QNet, self).__init__()
            self.l = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
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

   net = QNet()
   ```

3. Perform a gradient descent step on $(y_j − Q(\phi(j), a_j; θ))^2$ according to equation 3
   ```py
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
   ```

## DDQN

$$
\begin{equation}
y_j = r_j + \gamma Q(S_{t+1}, \argmax_{a'} Q(S_{t+1}, a'; \theta); \theta^-)
\end{equation}
$$

where $\theta^-$ are the parameters of the target network, which are only updated with the main network every $C$ steps.

Compared to DQN, the main difference is in gradient_descent function.

```py
q_values = net(states)
q_value_next = net(new_states)
q_value_next_target = target(new_states)


q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
# Double DQN
# y = R + gamma Q(s', argmax_a Q(s', a))
_argmax = q_value_next_target.argmax(1)
y = rewards + gamma * (1 - done) * q_value_next[torch.arange(len(_argmax)), _argmax]
```

## PER

![PER](../assets/PER.png)

### Without Sum Tree

- $P(j)$
  - Prioritized
    - $P(j) = \frac{p_j^\alpha}{\sum_k p_k^\alpha}$
    - $p_j = |\delta_j| + \epsilon$
    ```py
    Pj = np.array(self.priorities) ** self.alpha
    sumPj = sum(Pj)
    Pj = Pj / sumPj
    ```
  - Rank-based
    - $P(j) = \frac{1}{rank(j)}$
    ```py
    Pj = np.array(self.priorities)
    sorted_indices = np.argsort(Pj)[::-1]
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(Pj) + 1)
    Pj = 1 / ranks
    ```
- $w_j = \frac{\left({N}. {P(j)}\right)^{-\beta}}{max_i w_i}$
  ```py
  wj = (self.N * _Pj) ** (-self.beta)
  wj = wj / max(wj)
  ```

## Dueling DQN

![Dueling DQN](../assets/DuelingDQN.png)

Changes compared to DQN:

Authors introduced:
```py
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
```

Forward function:

$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \left(A(s,a;\theta,\alpha) - \frac{1}{|A|} \sum_{a'} A(s,a';\theta,\alpha)\right)$

Nice read: https://ai.stackexchange.com/questions/8128/questions-on-the-identifiability-issue-and-equations-8-and-9-in-the-d3qn-paper

# References

[1] [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), Mnih et al, 2013. Algorithm: DQN.
[2] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf), Hasselt et al, 2015. Algorithm: DDQN.
[3] [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf), Schaul et al, 2015. Algorithm: PER.
[4] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf), Wang et al, 2015. Algorithm: Dueling DQN.
