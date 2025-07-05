# 🧠 Reinforcement Learning Game Bot (DQN GridWorld)

This project implements a **Deep Q-Learning (DQN) Agent** trained to navigate a custom **2D GridWorld** environment. The agent learns optimal movement strategies using **PyTorch**, **OpenAI Gym**, and **Pygame**.

---

## 🎮 Features

- ✅ **Custom Grid Environment** using OpenAI Gym API
- ✅ **DQN Agent** with experience replay, target network
- ✅ **Adjustable Difficulty** (`easy`, `medium`, `hard`)
- ✅ **Obstacle Support** for navigation challenge
- ✅ **Live GUI Rendering** with Pygame
- ✅ **Reward Tracking Visualization** with Matplotlib

---

## 🧠 DQN Agent

- Learns to reach the goal while avoiding obstacles
- Reward:
  - Step penalty: -1 to -3 based on difficulty
  - Goal reward: +10 to +30
- Uses epsilon-greedy policy and updates target Q-network periodically

---

## 📈 Sample Reward Visualization

After training, use:
```bash
python visuals/plot_rewards.py
