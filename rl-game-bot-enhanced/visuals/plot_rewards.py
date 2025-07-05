import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("visuals/reward_history.npy")
plt.plot(rewards)
plt.title("Episode Rewards Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.savefig("visuals/reward_plot.png")
plt.show()