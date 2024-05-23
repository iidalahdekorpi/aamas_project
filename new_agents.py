import random
import numpy as np
from lbforaging.foraging.agent import Agent
from lbforaging.foraging.environment import Action, ForagingEnv, Env

from itertools import repeat, product

import pandas as pd


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def chooseAction(self, observation, epsilon=None):
        return random.choice([Action.NONE, Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD])
class GreedyAgent:

    def __init__(self, env, action_space):
        self.action_space = action_space
        self.env = env

    def act(self, observation):

        next_states, rewards, _, _ = self.env.step(self.action_space.sample()) 
        rewards = rewards[: self.env.n_agents] 
        best_action_index = np.argmax(rewards)
        return self.action_space.sample()[best_action_index]


class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def _center_of_players(self, positions):
        coords = np.array(positions).reshape(-1, 2)
        return np.rint(coords.mean(axis=0))

    def _move_towards(self, target, allowed):
        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def step(self, obs):
        raise NotImplementedError("Heuristic agent is implemented by H1-H4")



class H1(HeuristicAgent):
    """
    H1 agent always goes to the closest food
    """

    name = "H1"

    def step(self, obs):
        # Assuming the observation array is structured as follows:
        # [agent1_x, agent1_y, agent2_x, agent2_y, level1, level2, food1_x, food1_y, food2_x, food2_y, ...]
        num_agents = 2
        positions = [obs[i * 2:(i + 1) * 2] for i in range(num_agents)]
        levels = [obs[2 * num_agents + i] for i in range(num_agents)]

        self.observed_position = positions[0]

        # Wrap the obs in the ObservationWrapper
        wrapped_obs = ObservationWrapper(obs)

        try:
            r, c = self._closest_food(wrapped_obs)
        except TypeError:
            return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.LOAD])
        except ValueError:
            return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])

        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
        except ValueError:
            return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])

class H2(HeuristicAgent):
    """
	H2 Agent goes to the one visible food which is closest to the centre of visible players
	"""

    name = "H2"

    def step(self, obs):

        #players_center = self._center_of_players(obs.players)
        wrapped_obs = ObservationWrapper(obs)
        num_agents = 2
        positions = [obs[i * 2:(i + 1) * 2] for i in range(num_agents)]
        self.observed_position = positions[0]
        try:
            r, c = self._closest_food(wrapped_obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H3(HeuristicAgent):
    """
	H3 Agent always goes to the closest food with compatible level
	"""

    name = "H3"

    def step(self, obs):

        try:
            r, c = self._closest_food(obs, self.level)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H4(HeuristicAgent):
    name = "H4"

    def step(self, obs):
        num_agents = 2
        positions = [obs[i * 2:(i + 1) * 2] for i in range(num_agents)]
        levels = [obs[2 * num_agents + i] for i in range(num_agents)]

        self.observed_position = positions[0]
        players_center = self._center_of_players(positions)
        players_sum_level = sum(levels)

        # Wrap the obs in the ObservationWrapper
        wrapped_obs = ObservationWrapper(obs)

        try:
            r, c = self._closest_food(wrapped_obs, players_sum_level, players_center)
        except TypeError:
            return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.LOAD])
        except ValueError:
            return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
        
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
        except ValueError:
            return random.choice([Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST])
        
import gym
import numpy as np

def discretize_observation(observation, num_bins=5):
  """Discretizes the observation into a list of integer bin indices.

  Args:
      observation: A list of floats representing the agent's observation.
      num_bins: The number of bins to use for discretization (default: 5).

  Returns:
      A list of integers representing the discretized state.
  """
  discretized_state = []
  for obs in observation:
    # Normalize observation value (optional, adjust based on observation range)
    normalized_obs = (obs - min(observation)) / (max(observation) - min(observation))
    # Discretize the normalized value into a bin index
    bin_index = int(normalized_obs * (num_bins - 1))
    discretized_state.append(bin_index)
  return discretized_state

class QAgent:
    def __init__(self, observation_space, env):
        self.observation_space = observation_space[0].shape
        self.action_space = len(env.action_set)

        # Hyperparameters for Q-learning
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 1.0  # Exploration rate (decays over time)
        self.epsilon_decay = 0.995  # Rate of epsilon decay
        self.min_epsilon = 0.01  # Minimum exploration rate

        # Q-table initialization (replace with your preferred initialization method)
        self.Q_table = np.zeros(self.observation_space + (self.action_space,)).T

    def step(self, observation):
        # Epsilon-greedy exploration

        observation = discretize_observation(observation)
        if np.random.rand() < self.epsilon:
            action = random.randint(0,self.action_space) # Random action
        else:
            # Choose the action with the highest Q-value in the current state
            action = np.argmax(np.round(self.Q_table[observation]))

        # Update epsilon for decaying exploration
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        return action

    def update(self, observation, action, reward, next_observation, done):
        # Update Q-table based on Bellman equation
        Q_current = self.Q_table[observation][action]

        # If episode is done, set target reward to 0
        if done:
            Q_target = reward
        else:
            # Get the max Q-value from the next state
            Q_target = reward + self.discount_factor * np.max(self.Q_table[next_observation])

        # Update Q-value with learning rate
        Q_new = Q_current + self.learning_rate * (Q_target - Q_current)
        self.Q_table[observation][action] = Q_new



class Agent:
    def __init__(self, id, alpha=0.1, gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.id = id
        self.Q = np.ones((5, 5, 5, 6))  # Q-table for each (y, x, level) state and 6 actions

    def observ2state(self, x):
        return x[self.id]  # Independent agent observes its own state (y, x, level)

    def update(self, x, nx, a, r):
        xi = self.observ2state(x)
        nxi = self.observ2state(nx)
        self.Q[xi[0], xi[1], xi[2], a] += self.alpha * (r[self.id] + self.gamma * np.max(self.Q[nxi[0], nxi[1], nxi[2], :]) - self.Q[xi[0], xi[1], xi[2], a])

        return self.Q[x, :]  # Return the Q-values for the current agent's state

    def chooseAction(self, x, e):
        xi = self.observ2state(x)
        # Explore vs Exploit using e-greedy strategy
        return egreedy(self.Q[xi[0], xi[1], xi[2], :], e=e)
    

    # egreedy function
# e is the probability of choosing the best action
def egreedy(v,e=0.95):
    NA = len(v)
    b = np.isclose(v,np.max(v))
    no = np.sum(b)
    if no<NA:
        p = b*e/no+(1-b)*(1-e)/(NA-no)
    else:
        p = b/no

    return int(np.random.choice(np.arange(NA),p=p))

