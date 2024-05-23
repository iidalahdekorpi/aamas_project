# AAMAS PROJECT

This project experiments the performance of different agents on a lb-foraging food collecting game. 

## Installation

To install required packages run

```bash
pip install -r requirements.txt
```

## Running

The program can be run by running the main.py file on python. In the main.py script there are some parameters that can be changed, such as the environment and the type of the agents. If you want to use the environment where the apples are fixed (both levels and position) use env2, else use env1. To change the grid, change both the grid_size parapmeter and 5x5 in "Foraging-5x5-2p-2f-v2". 2p is the number of agents, and 2f is the number o apples

```bash

env1 = gym.make("Foraging-5x5-2p-2f-v2")
env = env1
n_apples = 2
grid_size = 5

```

If you are using the fixed environment, use Agent2 to create the agents, otherwise use Agent. 


```bash

agent_types = ['independent','JALAM']
agents = [Agent(id=i, grid_size=grid_size, n_apples=n_apples, n_agents=n_agents, maxlevel=env.max_player_level+1, agentType=agent_types[i]) for i in range(n_agents)]
    

```

The following parameter changes the number of episodes in which the agents are trained.

```bash
n_episodes = 1000

```