import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorldEnv(discrete.DiscreteEnv):
    """
    Grid World Environment from Sutton's RL book (Chapter 4).
    The agent is on a MxN grid but 2 terminal states on the top-left and bottom-right corners.

    For example, a 4x4 grid looks like:

    T o o o
    o x o o
    o o o o
    o o o T

    where x is the position and T are terminal states

    Actions can be taking in 4 directions, with actions over the edge
    leaving you in the current state.
    You receive a reward of -1 for every transition.
    """
    def __init__(self, shape=[4,4]) -> None:
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        # Number of states in grid
        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        # Reshape number of states as a grid
        grid = np.arange(nS).reshape(shape)
        # Set up iterator
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # If we are stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]

            # If not in terminal state
            else:
                # Get what is the next state depending on the move
                ns_up = s if y == 0 else (s - MAX_X)
                ns_right = s if x == (MAX_X - 1) else (s + 1)
                ns_down = s if y == (MAX_Y - 1) else (s + MAX_X)
                ns_left = s if x == 0 else (s - 1)

                # Calculate the probability, next_stage, what is the reward, and whether we reach terminal state
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            # Move to the next state
            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        self.P = P
        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        """
        Renders the current gridworld layout
        """
        if close:
            return

        # Save file or print on terminal
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()
                output = output.write("\n")
            output.write(output)

            it.iternext()



if __name__=="__main__":
    grid = GridWorldEnv()