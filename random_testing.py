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
from new_agents import RandomAgent

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

def plot_learning_curve(rewards, episode_length, window_size=1000):
    avg_rewards = [
        [np.mean(agent_rewards[max(0, i - window_size):(i + 1)]) for i in range(len(agent_rewards))]
        for agent_rewards in rewards
    ]
    avg_episode_length = [np.mean(episode_length[max(0, i - window_size):(i + 1)]) for i in range(len(episode_length))]
    
    
    plt.figure(figsize=(15, 5))
    
    # Subplot for Total Rewards and Average Rewards
    plt.subplot(1, 2, 1)
    for idx, agent_rewards in enumerate(rewards):
        plt.plot(avg_rewards[idx], label=f'Agent {idx+1} Average Rewards (Window size = {window_size})', linestyle='-')
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
    plt.savefig('jalam_123.png')
    plt.show()

def plot_q_value_heatmap(Q):
    plt.figure(figsize=(8, 6))
    plt.imshow(Q, cmap='hot', interpolation='nearest')
    plt.title('Q-Value Heatmap')
    plt.colorbar(label='Q-Value')
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.show()

def run_random(env, agent):
    total_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()  
        episode_reward = 0 
        done = False 
        steps = 0
        while not done:
            steps += 1
            action = agent.act(obs)
            
            next_obs, reward, done, _ = env.step(action)
            
            episode_reward += reward
            
            obs = next_obs
        
        total_rewards.append(episode_reward)
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
    agents = [RandomAgent(env.action_space) for _ in range(2)]
    n_episodes = 10000
    episode_length = []
    total_rewards = [[] for _ in range(2)]
    i=0
    for episode in range(n_episodes):
        steps, rewards = run_random(env, agents)
        episode_length.append(steps)
        for idx in range(2):
            total_rewards[idx].append(rewards[idx])
        if (i % 100 == 0):
            print(i)
        i += 1
        #print(f"Episode {episode + 1}: Steps taken = {steps}, Rewards = {rewards}"

    plot_learning_curve(total_rewards, episode_length)
    #plot_q_value_heatmap(agents[0].Q[:, :, 0])
    env.close()