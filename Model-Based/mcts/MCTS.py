import copy
import math
import random
import time

from _2048 import GAME_2048


class TreeNode:
    def __init__(self, board: GAME_2048, parent):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.expanded = False
        self.terminal = board.check_done()
        self.score = 0


class MCTS:
    def search(self, state, num_simulations=1000):
        root = TreeNode(state, None)
        start = time.time()
        for _ in range(num_simulations):
            node = self.select(root)
            score = self.simulate(node)
            self.back_propagate(node, score)
            if time.time() - start >= 1:
                break
        action = self.select_best_action(root)
        return action

    def select(self, node):
        while not node.terminal:
            if node.expanded:
                node = self.select_best_action(node, 1)
            else:
                return self.expand(node)

        return node

    def expand(self, node):
        _all_states = node.board.generate_states()
        for action, state in _all_states:
            if action not in node.children:
                _new_child = TreeNode(state, node)
                node.children[action] = _new_child
                if len(node.children) == len(_all_states):
                    node.expanded = True
                return _new_child

    def simulate(self, node):
        score = 0
        gamma_ini = 1
        gamma_end = 0.01
        gamma_steps = 100
        gamma = lambda i: max(gamma_ini - (gamma_ini - gamma_end) * i / gamma_steps, gamma_end)
        step = 0
        env: GAME_2048 = copy.deepcopy(node.board)
        while not env.check_done():
            boards = env.generate_states()
            acts = [i for i, _ in boards]
            act = random.choice(acts)
            _, rew, *_ = env.step(act)
            score += gamma(step)*env.points
            step += 1
        return score

    def back_propagate(self, node, score):
        while node:
            node.score += score
            node.visits += 1
            node = node.parent

    def select_best_action(self, node, c=0):
        best_score = -math.inf
        best_moves = []
        # print(node.children)
        for action, child in node.children.items():
            score = (child.score) / child.visits + c * math.sqrt(
                2 * math.log(node.visits) / child.visits
            )
            # if c == 0: print(f"action: {action}, score: {score}")
            if c == 0:
                if score > best_score:
                    best_score = score
                    best_moves = [action]
                elif score == best_score:
                    best_moves.append(action)
            else:
                if score > best_score:
                    best_score = score
                    best_moves = [child]
                elif score == best_score:
                    best_moves.append(child)
        # if c == 0: print(f"best score: {best_score}")
        return random.choice(best_moves)