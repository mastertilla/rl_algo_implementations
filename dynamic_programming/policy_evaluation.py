from sys import intern
import numpy as np
import sys

from numpy.core.arrayprint import IntegerFormat
if "../" not in sys.path:
    sys.path.append('../')
from lib.envs.gridworld import GridWorldEnv
import pdb

env = GridWorldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description
    of the environment's dynamics.

    :param policy: [S, A] shaped matrix representing the policy
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
    # Starts with 0 as value function for all states
    V = np.zeros(env.nS)

    while True:
        delta = 0
        # For each state, we perform a full sweep
        for s in range(env.nS):
            v = 0
            # Look at the possible actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states
                for prob, next_state, reward, done in env.P[s][a]:
                    # calculate the expected value
                    v += action_prob * prob * \
                        (reward + discount_factor * V[next_state])
            # How much our value function changed (across all states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break

    return np.array(V)


if __name__ == "__main__":
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(random_policy, env)
    # Testing the results
    expected_v = np.array(
        [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(
        v, expected_v, decimal=2, verbose=True)
