import logging

from Coach import Coach
from gobang import GobangGame as Game
from net import NNetWrapper as nn


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict(
    {
        "numIters": 1000,
        "numEps": 100,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 200000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 25,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 40,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        "numItersForTrainExamplesHistory": 20,
    }
)


def main():
    g = Game(6)
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()


if __name__ == "__main__":
    main()
