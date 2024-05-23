import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(rewards, episode_length,agent_types, window_size=100):

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
        plt.plot(avg_rewards[idx], label=f'Agent {idx+1} ({agent_types[idx]}) Average Rewards (Window size = {window_size})', linestyle='-')
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