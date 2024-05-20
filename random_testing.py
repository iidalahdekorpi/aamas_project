import gym
import lbforaging
import numpy as np
import time
from lbforaging.foraging.environment import Action
import random
from agent import Agent
#from new_agents import RandomAgent, GreedyAgent, H1, H2, H3, H4, QAgent


def run_episode(env, agents):
    obs = env.reset()
    done = False
    A = np.ones(2,dtype = int)
    steps = 0
    epsilon = 0.9
    while not done:
        for i in range(len(agents)):
            A[i] = agents[i].chooseAction(obs[i], epsilon)
        nx, rewards, dones, info = env.step(A)
        obs = nx
        steps += 1
        agents[0].update(obs[0], nx[0], A,rewards)
        agents[1].update(obs[1], nx[1], A, rewards)
        env.render()
        done = np.all(dones)
        time.sleep(0.2)
    return steps, rewards



if __name__ == "__main__":

    env = gym.make("Foraging-5x5-2p-2f-v2")

    # Create agents
    agents = []
    n_agents = 2
    for i in range(n_agents):
        agents.append(Agent(id = i, grid_size=5, n_apples=2, n_agents=n_agents))
    n_episodes = 1
    for episode in range(n_episodes):
        steps, rewards = run_episode(env, agents)
        print(f"Episode {episode + 1}: Steps taken = {steps}, Rewards = {rewards}")

    env.close()