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

def custom_spawn_food(self,max_food = 2,max_level = 3):
    min_level = max_level if self.force_coop else 1
    specific_locations = [(2, 2, 2), (4, 4, 1)]

    for row,col,level in specific_locations:

        if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue
        self.field[row, col] = level
    self._food_spawned = self.field.sum()


def custom_spawn_players(self, max_player_level=1):
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        1,
                        self.field_size,
                    )
                    break
                attempts += 1


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
        #time.sleep(0.5)  # ADJUST THIS TO MAKE THE GAME RUN SLOWER
    return steps, total_rewards

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curve(rewards, episode_length, window_size=100):

    # Function for running the learning curve

    avg_rewards = [
        [np.mean(agent_rewards[max(0, i - window_size):(i + 1)]) for i in range(len(agent_rewards))]
        for agent_rewards in rewards
    ]
    avg_episode_length = [np.mean(episode_length[max(0, i - window_size):(i + 1)]) for i in range(len(episode_length))]

    plt.figure(figsize=(15, 5))
    # Subplot for Average Rewards
    plt.subplot(1, 2, 1)
    for idx, agent_rewards in enumerate(rewards):
        agent_type = "Central" if idx == 0 else "JALAM"
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
    plt.savefig('images/learning_curve_central_v_jalam.png')
    plt.show()

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

    for i in range(len(agents)):
        agents[i].n_apples = 2

    return steps, total_rewards

if __name__ == "__main__":

    env1 = gym.make("Foraging-5x5-2p-2f-v2")

    env2 = gym.make("Foraging-6x6-2p-2f-v2")
    env2.spawn_food = custom_spawn_food.__get__(env2, ForagingEnv)
    env2.spawn_players = custom_spawn_players.__get__(env2, ForagingEnv)
    env2.spawn_food()
    env2.spawn_players()

    # Choose the environment
    env = env1

    # Create agents
    agents = []
    n_agents = 2

    # Grid size 5 for env1 and 6 for env2
    grid_size = 5

    # Choose agents
    agents = [
        Agent(id=0, grid_size=grid_size, n_apples=2, n_agents=2, maxlevel=env.max_player_level, agentType='central'),
        Agent(id=1, grid_size=grid_size, n_apples=2, n_agents=2, maxlevel=env.max_player_level, agentType='JALAM')
    ]

    # Choose the number of episodes run
    n_episodes = 10000
    episode_length = []
    total_rewards = [[] for _ in range(2)]

    i=0
    for episode in range(n_episodes):

        steps, rewards = run_episode(env, agents) # For random agent, run run_random
        episode_length.append(steps)
        for idx in range(2):
            total_rewards[idx].append(rewards[idx])

        if (i % 100 == 0):
            print(i) # Printing the progress
        i += 1
        #print(f"Episode {episode + 1}: Steps taken = {steps}, Rewards = {rewards}"

    results = {
        "Agent 1 (JALAM)": np.array(total_rewards[0]),
        "Agent 2 (Independent)": np.array(total_rewards[1])
    }

    plot_learning_curve(total_rewards, episode_length)
    #plot_q_value_heatmap(agents[0].Q[:, :, 0])
    #compare_results(results, confidence=0.95, title="Agents Comparison", metric="Total Rewards")

    env.close()