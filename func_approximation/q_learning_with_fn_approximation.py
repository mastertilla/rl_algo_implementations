import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import pdb

if "../" not in sys.path:
    sys.path.append("../")

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    The function takes observations and returns action probabilities.

    :param Q: Maps state to action-values. Each value is a numpy array of length nA
    :type Q: Dict
    :param epsilon: Probability of selecting a random action
    :type epsilon: Float (between 0 and 1)
    :param nA: Number of actions in the environment
    :type nA: Integer

    :return: Probabilities for each action.
    :rtype: Numpy array
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)

        return A

    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm - off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    :param env: Environment
    :type env: OpenAI environment
    :param num_episodes: Number of episodes to run for
    :type num_episodes: Integer
    :param discount_factor: Gamma discount factor, defaults to 1.0
    :type discount_factor: float, optional
    :param alpha: TD learning rate, defaults to 0.5
    :type alpha: float, optional
    :param epsilon: Chance to sample a random action, defaults to 0.1
    :type epsilon: float, optional (between 0 and 1)

    :return Q: Optimal action-value function, mapping state to action values
    :rtype Q: Dict
    :return stats: Includes episode_lengths and episode_rewards
    :rtype stats: Numpy array
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keep track of stats
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    # The policy that we are following - initialisation
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print(f'Episode {i_episode + 1}/{num_episodes}')
            sys.stdout.flush()
        
        # Reset environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            # take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]

            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state
        
    return Q, stats

if __name__=="__main__":
    Q, stats = q_learning(env, 500)
    plotting.plot_episode_stats(stats, noshow=True)