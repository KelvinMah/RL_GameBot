import gym
import numpy as np
import matplotlib.pyplot as plt
from environment.grid_env import GridWorldEnv
from agent.dqn_agent import DQNAgent
import torch

env = GridWorldEnv(grid_size=5)
state_dim = 2
action_dim = env.action_space.n

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

episodes = 500
target_update_freq = 20
rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    if ep % target_update_freq == 0:
        agent.update_target()

    if ep % 50 == 0:
        print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Save reward history
np.save("visuals/reward_history.npy", np.array(rewards))