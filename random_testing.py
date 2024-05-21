import gym
import lbforaging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from lbforaging.foraging.environment import Action
import random
from agent import Agent
from lbforaging.foraging.environment import ForagingEnv
#from new_agents import RandomAgent, GreedyAgent, H1, H2, H3, H4, QAgent
from utils import compare_results

def custom_spawn_food(self, specific_locations, level=1):
    self.field = np.zeros((self.rows, self.cols), dtype=int)
    for row, col in specific_locations:
        self.field[row, col] = level
    self._food_spawned = self.field.sum()


def run_episode(env, agents):
    obs = env.reset()
    done = False
    A = np.ones(2,dtype = int)
    steps = 0
    epsilon = 0.999
    total_rewards = [0,0]
    while not done:
        for i in range(len(agents)):
            A[i] = agents[i].chooseAction(obs[i], epsilon)
        nx, rewards, dones, info = env.step(A)
        steps += 1
        for i in range(len(agents)):
            agents[i].update(obs[i], nx[i], A,rewards)
            total_rewards[i] += rewards[i]

        obs = nx
        env.render()
        done = np.all(dones)
        #time.sleep(0.5)
    agents[0].n_apples = 2
    agents[1].n_apples = 2
    return steps, total_rewards

def plot_learning_curve(rewards, episode_length):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Learning Curve - Total Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')

    plt.subplot(1, 2, 2)
    plt.plot(episode_length)
    plt.title('Learning Curve - Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length (Steps)')

    plt.tight_layout()
    plt.show()

def plot_q_value_heatmap(Q):
    plt.figure(figsize=(8, 6))
    plt.imshow(Q, cmap='hot', interpolation='nearest')
    plt.title('Q-Value Heatmap')
    plt.colorbar(label='Q-Value')
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.show()


if __name__ == "__main__":

    env = gym.make("Foraging-5x5-2p-2f-v2")
    '''
    env = gym.make("Foraging-5x5-2p-2f-coop-v2")

    food_locations = [(2, 2), (3, 4)]
    food_level = 3
    env.spawn_food = custom_spawn_food
    env.spawn_food(food_locations, food_level)
    '''
    # Create agents
    agents = []
    n_agents = 2
    agents = [Agent(id=i, grid_size=5, n_apples=2, n_agents=2) for i in range(2)]
    n_episodes = 2
    episode_lengths = []
    total_rewards = []
    for episode in range(n_episodes):
        steps, rewards = run_episode(env, agents)
        total_rewards.append(sum(rewards))
        episode_lengths.append(steps)
        #print(f"Episode {episode + 1}: Steps taken = {steps}, Rewards = {rewards}")
    env.close()

    plot_learning_curve(total_rewards, episode_lengths)
    #plot_q_value_heatmap(agents[0].Q[:, :, 0])