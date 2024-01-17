import argparse

import numpy as np
from _2048 import GAME_2048
from MCTS import MCTS
from tqdm import tqdm

env = GAME_2048()
AI = MCTS()

arg = argparse.ArgumentParser()
arg.add_argument("--mode", type=str, default="mcts")
args = arg.parse_args()


points = []
for i in tqdm(range(1)):
    env.reset()
    step = 10
    while True:
        if args.mode == "random":
            action = np.random.randint(4)
        else:
            action = AI.search(env, step)
        _, rew, *_ = env.step(action)
        env.render()
        if env.check_done():
            break
        step += 1
        step = min(step, 1000)
    points.append(env.points)

# expetation of points
print(sum(points) / len(points))
