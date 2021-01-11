import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import pdb
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
    sys.path.append("../")

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

env = gym.envs.make('MountainCar-v0')
# pdb.set_trace()
# Feature preprocessing - normalise to zero mean and unit variance
# We use a few samples from the observation space 
observation_samples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_samples)

# We convert states to a featurised representation
featuriser = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
])

featuriser.fit(scaler.transform(observation_samples))

class Estimator:
    """Value Function Approximator
    """
    def __init__(self):
        """We create a separate model for each action in the evironment's
        action sapce. Alternatively we could somehow encode the action into
        the features (harder to code)
        """
        self.models = []
        # self.env = env
        # self.scaler = scaler
        # self.featuriser = featuriser
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to use partial_fit once to initialise the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurise_state(env.reset())], [0])
            self.models.append(model)

    def featurise_state(self, state):
        """
        Returns the featurised representation for a state
        """
        scaled = scaler.transform([state])
        featurised = featuriser.transform(scaled)

        return featurised[0]

    def predict(self, s, a=None):
        """This function makes value fn predictions

        :param s: state to make a prediction for
        :param a: (Optional) Action to make a prediction for
        """
        pdb.set_trace()
        features = self.featurise_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action
        towards the target y
        """
        features = self.featurise_state(s)
        self.models[a].partial_fit([features], [y])

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    The function takes observations and returns action probabilities.

    :param estimator: An estimator that returns q values for a given state
    :type estimator: Function
    :param epsilon: Probability of selecting a random action
    :type epsilon: Float (between 0 and 1)
    :param nA: Number of actions in the environment
    :type nA: Integer

    :return: Probabilities for each action.
    :rtype: Numpy array
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)

        return A

    return policy_fn

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm - off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    :param env: Environment
    :type env: OpenAI environment
    :param estimator: Action-value function estimator
    :type estimator: Function
    :param num_episodes: Number of episodes to run for
    :type num_episodes: Integer
    :param discount_factor: Gamma discount factor, defaults to 1.0
    :type discount_factor: float, optional
    :param alpha: TD learning rate, defaults to 0.5
    :type alpha: float, optional
    :param epsilon: Chance to sample a random action, defaults to 0.1
    :type epsilon: float, optional (between 0 and 1)
    :param epsilon_decay: Each episode, epsilon is decayed by this factor
    :type epsilon_decay: float, optional

    :return Q: Optimal action-value function, mapping state to action values
    :rtype Q: Dict
    :return stats: Includes episode_lengths and episode_rewards
    :rtype stats: Numpy array
    """
    # Keep track of stats
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    for i_episode in range(num_episodes):
        # The policy that we are following
        policy = make_epsilon_greedy_policy(estimator, 
                            epsilon * epsilon_decay**i_episode, env.action_space.n)

        # Print the episode we are on and reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        # Reset environment and pick the first action
        state = env.reset()

        # Only used for SARSA, not Q-learning
        next_action = None

        # One step in the environment
        for t in itertools.count():
            # Choose an action to take
            # If we are using SARSA, next_action is already defined
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            # take a step
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)
            
            # Update the function approximator using our target
            estimator.update(state, action, td_target)

            print(f'Step {t} @ Episode {i_episode + 1}/{num_episodes} ({last_reward})')

            if done:
                break

            state = next_state
        
    return stats

if __name__=="__main__":
    estimator = Estimator()

    # Mountain car does not need epsilon > 0
    # Initial estimate for all states is too "optimistic"
    stats = q_learning(env, estimator, 100, epsilon=0.0)

    # Plotting
    plotting.plot_cost_to_go_mountain_car(env, estimator, name='q_learning_fn_estimator')
    plotting.plot_episode_stats(stats, name='q_learning_fn_estimator', smoothing_window=25, noshow=True)