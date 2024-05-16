import gym
import lbforaging
import numpy as np
import time
from lbforaging.foraging.environment import Action
from lbforaging.foraging.agent import Agent
import random

from new_agents import RandomAgent, GreedyAgent, H1, QAgent


def run_episode(env, agents):
    observations = env.reset()
    done = False
    steps = 0

    while not done:
        actions = [agent.act(obs) for agent, obs in zip(agents, observations)]
        observations, rewards, ndone, info = env.step(actions)
        steps += 1
        env.render()
        done = np.all(ndone)
        time.sleep(0.2)

    return steps, rewards



if __name__ == "__main__":

    env = gym.make("Foraging-8x8-2p-2f-v2")

    # Create agents
    agents = [RandomAgent(env.action_space) for _ in range(2)]
    n_episodes = 2
    for episode in range(n_episodes):
        steps, rewards = run_episode(env, agents)
        print(f"Episode {episode + 1}: Steps taken = {steps}, Rewards = {rewards}")

    env.close()