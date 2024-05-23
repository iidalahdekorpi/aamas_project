import gym
import lbforaging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from lbforaging.foraging.environment import Action
from agent import Agent, Agent2, RandomAgent
from lbforaging.foraging.environment import ForagingEnv
from utils import compare_results
from change_environment import custom_spawn_food, custom_spawn_players
from plot_curves import plot_learning_curve

def run_episode(env, agents):

    # For all except random agent

    obs = env.reset()
    done = False

    # Initilizing the actions list
    A = np.ones(len(agents),dtype = int)

    steps = 0
    epsilon = 0.95

    total_rewards = [0 for _ in range(len(agents))]
    while not done:
        for i in range(len(agents)):
            A[i] = agents[i].chooseAction(obs[i], epsilon)
        nx, rewards, dones, info = env.step(A)
        steps += 1
        for i in range(len(agents)):
            agents[i].update(obs[i], nx[i], A, rewards)
            total_rewards[i] += rewards[i]

        obs = nx
        env.render()
        done = np.all(dones)
        time.sleep(0.2)  # ADJUST THIS TO MAKE THE GAME RUN SLOWER
    return steps, total_rewards


def run_random(env, agents):

    # For the random agent

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

    return steps, total_rewards

if __name__ == "__main__":

    env1 = gym.make("Foraging-5x5-2p-2f-v2")

    env2 = gym.make("Foraging-6x6-2p-2f-v2")
    env2.spawn_food = custom_spawn_food.__get__(env2, ForagingEnv)
    env2.spawn_players = custom_spawn_players.__get__(env2, ForagingEnv)
    env2.spawn_food()
    env2.spawn_players()

    # Choose the environment
    env = env2
    
    n_apples = 2

    # Grid size 5 for env1 and 6 for env2
    grid_size = 6

    # Choose the agent types ('JALAM', 'independent', 'observ', 'central')
    agent_types = ['independent','JALAM']
    n_agents = len(agent_types)
    # Choose agents (Agent for the random environment, Agent2 for the fixed environment)
    agents = [Agent2(id=i, grid_size=grid_size, n_apples=n_apples, n_agents=n_agents, maxlevel=env.max_player_level+1, agentType=agent_types[i]) for i in range(n_agents)]
    # Choose the number of episodes run
    n_episodes = 100
    episode_length = []
    total_rewards = [[] for _ in range(len(agents))]

    i=0
    for episode in range(n_episodes):

        steps, rewards = run_episode(env, agents) # For random agent, run run_random
        episode_length.append(steps)
        for idx in range(len(agents)):
            total_rewards[idx].append(rewards[idx])

        if (i % 100 == 0):
            print(i) # Printing the progress
        i += 1
        #print(f"Episode {episode + 1}: Steps taken = {steps}, Rewards = {rewards}"

    results = {
        "Agent 1": np.array(total_rewards[0]),
        "Agent 2": np.array(total_rewards[1])
    }

    plot_learning_curve(total_rewards, episode_length,agent_types)
    #compare_results(results, confidence=0.95, title="Agents Comparison", metric="Total Rewards")

    env.close()