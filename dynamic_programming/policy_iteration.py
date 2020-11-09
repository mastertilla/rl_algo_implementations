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
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across all states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v


        if delta < theta:
            break

    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy improvement algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found..

    :param env: OpenAI environment
    :param policy_eval_fn: Policy evaluation function that takes 3 arguments: policy, env, 
                           discount_factor, defaults to policy_eval
    :param discount_factor: gamma discount factor, defaults to 1.0
    
    :return policy: Policy is the optimal policy, a matrix of shape [S, A] where each state
                    s contains a valid probability distribution over actions.
                    V is the value function for the optimal policy.
    :return policy [type]: tuple (policy, V)
    """
    # Start with random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # Evaluate the value function for a certain policy
        v = policy_eval(policy, env)
        policy_stable = True
        for s in range(env.nS - 1): # We update the policy for each state
            chosen_a = np.argmax(policy[s]) # If ties, the first element is returned
            A = np.zeros(env.nA) # We evaluate q for each action
            for a,_ in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    A[a] = prob * (reward + discount_factor * v[next_state])

            # Select the best action
            # Ties are resolved arbitrarly
            best_a = np.argmax(A)

            if best_a != chosen_a:
                policy_stable = False
            
            # We have assigned 1 if action is better or equal that current v
            policy[s] = np.eye(env.nA)[best_a]

        # If the policy has not changed
        if policy_stable is True:
            break

    return policy, v

start = time.time()
policy, v = policy_improvement(env)

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

print(f'It took: {time.time() - start}')