import argparse
import multiprocessing

import numpy as np
from _2048 import GAME_2048
from MCTS import MCTS
from tqdm import tqdm


def run_experiment(env, args, progress_counter):
    env.reset()
    while True:
        if args.mode == "random":
            action = np.random.randint(4)
        else:
            action = AI.search(env)
        _, rew, *_ = env.step(action)
        if env.check_done():
            break
    with progress_counter.get_lock():
        progress_counter.value += 1
    return env.points


def run_experiment_parallel(experiment_id, env, args, result_queue, progress_counter):
    result = run_experiment(env, args, progress_counter)
    result_queue.put((experiment_id, result))


if __name__ == "__main__":
    num_experiments = 10
    env = GAME_2048()
    AI = MCTS()
    arg = argparse.ArgumentParser()
    arg.add_argument("--mode", type=str, default="mcts")
    args = arg.parse_args()

    result_queue = multiprocessing.Queue()
    progress_counter = multiprocessing.Value("i", 0)
    processes = []

    for i in range(num_experiments):
        process = multiprocessing.Process(
            target=run_experiment_parallel,
            args=(i, env, args, result_queue, progress_counter),
        )
        processes.append(process)
        process.start()

    for process in tqdm(processes, desc="Experiments", unit="experiment"):
        process.join()

    results = [0] * num_experiments
    while not result_queue.empty():
        experiment_id, result = result_queue.get()
        results[experiment_id] = result

    print(results)
