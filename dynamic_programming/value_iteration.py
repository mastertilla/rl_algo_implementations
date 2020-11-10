"""
TODO: Now, the evaluation picks value randomly if ties
Check if we can solve for ties with not chosing randomly
"""

from sys import intern
import numpy as np
import sys
import time

from numpy.core.arrayprint import IntegerFormat
from numpy.testing._private.utils import break_cycles
if "../" not in sys.path:
    sys.path.append('../')
from lib.envs.gridworld import GridWorldEnv
import pdb

env = GridWorldEnv()

def value_interation(env, theta=0.0001, discount_factor=1.0):
    """
    Value iteration algorithm

    :param env: OpenAI env. env.P represents the transition probabilities 
                of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done)
                env.nS is the number of states in the environment
                env.nA is the number of actions in the environment

    :param discount_factor: Gamma discount factor, defaults to 1.0
    :param theta: We stope the evaluation once our value function change is less 
                  than theta for all states, defaults to 0.00001

    :return: Vector of length env.nS representing the value function for each state
    """
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        for s in range(env.nS-1):
            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, _ in env.P[s][a]:
                    A[a] = reward + (discount_factor * prob * V[next_state])

            best_a = np.max(A)
            delta = max(delta, np.abs(best_a - V[s]))
            V[s] = best_a

        if delta < theta:
            break
        
    policy = np.zeros([env.nS, env.nA])
    
    for s in range(env.nS):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                A[a] = reward + (discount_factor * prob * V[next_state])

        best_a = np.argmax(A)

        policy[s][best_a] = 1.0

    return policy, V


start = time.time()

policy, v = value_interation(env)
# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

print(f'It took: {time.time() - start}')