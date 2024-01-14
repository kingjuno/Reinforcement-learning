import copy

import gym
import numpy as np
from gym import spaces


class GAME_2048(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.board = np.zeros(shape=(4, 4), dtype=np.int32)
        self.points = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2048, shape=(4, 4), dtype=np.int32
        )
        self.place_object()

    def check_done(self):
        filled = (self.board != 0).all()
        if not filled:
            return False
        temp_board = np.pad(self.board, 1)
        for i in range(1, len(temp_board)-1):
            for j in range(1, len(temp_board[i])-1):
                if (temp_board[i][j] in [
                    temp_board[i][j-1], temp_board[i][j+1],
                    temp_board[i+1][j], temp_board[i-1][j]
                ]):
                    return False
        return True

    def place_object(self):
        value_to_place = np.random.choice([2, 4], p=[0.95, 0.05])
        empty_spaces = np.argwhere(self.board == 0)
        space = np.random.randint(0, len(empty_spaces))
        self.board[tuple(empty_spaces[space])] = value_to_place

    def transform(self, board):
        reward = 0
        for ind, i in enumerate(board):
            stack = []
            for j in i[::-1]:
                if not j:
                    continue
                if not stack:
                    stack.append((j, 1))
                elif stack[-1][0] == j and stack[-1][1]:
                    stack[-1] = (stack[-1][0]*2, 0)
                    reward += stack[-1][0]
                else:
                    stack.append((j, 1))

            if not stack:
                stack = np.zeros(len(i))
            else:
                stack = np.array(stack[::-1])[:, 0]
            if len(stack) != len(i):
                stack = np.concatenate(
                    [stack, np.zeros(len(i)-len(stack))]
                )
            board[ind] = stack
        return board, reward

    def reset(self):
        self.board = np.zeros(shape=(4, 4), dtype=np.int32)
        self.points = 0
        self.place_object()
        return self.board, {}

    def step(self, action):
        """
        parameters: 
        action: int, [0-3]
        0 - move down
        1 - move right
        2 - move up
        3 - move left
        """
        assert 0 <= action <= 3, "action must be in range [0,3]"

        state = copy.deepcopy(self.board)
        if action == 0:
            self.board = np.rot90(self.board, 3)
            self.board, reward = self.transform(self.board)
            self.board = np.rot90(self.board, 1)
        elif action == 1:
            self.board, reward = self.transform(self.board)
        elif action == 2:
            self.board = np.rot90(self.board, 1)
            self.board, reward = self.transform(self.board)
            self.board = np.rot90(self.board, 3)
        else:
            self.board = np.rot90(self.board, 2)
            self.board, reward = self.transform(self.board)
            self.board = np.rot90(self.board, 2)

        if (self.board == state).all():
            return self.board, -10, False, {}

        self.points += reward
        self.place_object()
        done = self.check_done()
        return self.board, reward, done, {}

    def render(self, mode='human', close=False):
        print(f'----- points: {self.points} -----')
        for i in self.board:
            for j in i:
                spaces = 4-len(str(j))
                print(f'|{j}{" "*spaces}', end='')
            print('|')

    def generate_states(self):
        # print(self.board)
        # print('generating states')
        states = []
        for i in range(4):
            game = copy.deepcopy(self)
            _, rew, _, _ = game.step(i)
            if rew != -10:
                states.append([i,game])
        # shuffle states
        np.random.shuffle(states)
        return states