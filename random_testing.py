import gym
import lbforaging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from lbforaging.foraging.environment import Action
import random
from agent import Agent, RandomAgent
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
    epsilon = 0.95
    total_rewards = [0,0]
    while not done:
        for i in range(len(agents)):
            A[i] = agents[i].chooseAction(obs[i], epsilon)
        nx, rewards, dones, info = env.step(A)
        steps += 1
        for i in range(len(agents)):
            agents[i].update(obs[i], nx[i], A, rewards)
            total_rewards[i] += rewards[i]
            #print(steps, agents[i].n_apples)
            if agents[i].n_apples == 0:
                done = True

        obs = nx
        env.render()
        done = np.all(dones)
        #time.sleep(0.5)
    for i in range(len(agents)):
        agents[i].n_apples = 2
    return steps, total_rewards

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curve(rewards, episode_length, window_size=100):
    avg_rewards = [
        [np.mean(agent_rewards[max(0, i - window_size):(i + 1)]) for i in range(len(agent_rewards))]
        for agent_rewards in rewards
    ]
    avg_episode_length = [np.mean(episode_length[max(0, i - window_size):(i + 1)]) for i in range(len(episode_length))]

    plt.figure(figsize=(15, 5))
    # Subplot for Average Rewards
    plt.subplot(1, 2, 1)
    for idx, agent_rewards in enumerate(rewards):
        agent_type = "Independent" if idx == 0 else "JALAM"
        plt.plot(avg_rewards[idx], label=f'Agent {idx+1} ({agent_type}) Average Rewards (Window size = {window_size})', linestyle='-')
    plt.title('Learning Curve - Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()

    # Subplot for Episode Length and Average Episode Length
    plt.subplot(1, 2, 2)
    plt.plot(avg_episode_length, label=f'Average Episode Length (Window size = {window_size})', linestyle='-')
    plt.title('Learning Curve - Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length (Steps)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curve_jalam_and_independent_2.png')
    plt.show()

def run_random(env, agents):
    obs = env.reset()
    done = False
    A = np.ones(len(agents), dtype=int)
    steps = 0
    total_rewards = [0, 0]

    while not done:
        for i in range(len(agents)):
            action = agents[i].chooseAction(obs[i])
            A[i] = action.value if isinstance(action, Action) else action

        nx, rewards, dones, info = env.step(A)
        steps += 1

        for i in range(len(agents)):
            if not isinstance(agents[i], RandomAgent):
                agents[i].update(obs[i], nx[i], A, rewards)
            total_rewards[i] += rewards[i]

        obs = nx
        env.render()
        done = np.all(dones)

    for i in range(len(agents)):
        agents[i].n_apples = 2

    return steps, total_rewards

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
    #agents = [Agent(id=i, grid_size=5, n_apples=2, n_agents=2, maxlevel=env.max_player_level) for i in range(2)]
    #agents = [RandomAgent(env.action_space) for _ in range(2)]
    agents = [
        Agent(id=0, grid_size=5, n_apples=2, n_agents=2, maxlevel=env.max_player_level, agentType='JALAM'),
        Agent(id=1, grid_size=5, n_apples=2, n_agents=2, maxlevel=env.max_player_level, agentType='independent')
    ]
    n_episodes = 1000
    episode_length = []
    total_rewards = [[] for _ in range(2)]
    i=0
    for episode in range(n_episodes):
        steps, rewards = run_episode(env, agents)
        episode_length.append(steps)
        for idx in range(2):
            total_rewards[idx].append(rewards[idx])
        if (i % 100 == 0):
            print(i)
        i += 1
        #print(f"Episode {episode + 1}: Steps taken = {steps}, Rewards = {rewards}"
    results = {
        "Agent 1 (JALAM)": np.array(total_rewards[0]),
        "Agent 2 (Independent)": np.array(total_rewards[1])
    }
    #plot_learning_curve(total_rewards, episode_length)
    #plot_q_value_heatmap(agents[0].Q[:, :, 0])
    compare_results(results, confidence=0.95, title="Agents Comparison", metric="Total Rewards")

    env.close()