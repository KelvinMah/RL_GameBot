import gym
from gym import spaces
import numpy as np
import pygame
import random

class GridWorldEnv(gym.Env):
    COLORS = {
        'agent': (0, 0, 255),
        'goal': (0, 255, 0),
        'obstacle': (255, 0, 0),
        'bg': (240, 240, 240),
        'grid': (200, 200, 200)
    }

    def __init__(self, grid_size=5, num_obstacles=0, difficulty='easy', render_mode=False):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.cell_size = 50
        self.window_size = self.grid_size * self.cell_size
        self.move_penalty = {'easy': -1, 'medium': -2, 'hard': -3}[difficulty]
        self.goal_reward = {'easy': 10, 'medium': 20, 'hard': 30}[difficulty]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)
        self.reset()

        if self.render_mode:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld")

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        self.obstacles = set()

        while len(self.obstacles) < self.num_obstacles:
            pos = tuple(np.random.randint(0, self.grid_size, size=2))
            if pos != tuple(self.agent_pos) and pos != tuple(self.goal_pos):
                self.obstacles.add(pos)

        return self.agent_pos.copy()

    def step(self, action):
        old_pos = self.agent_pos.copy()
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

        if tuple(self.agent_pos) in self.obstacles:
            self.agent_pos = old_pos

        reward = self.move_penalty
        done = False
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = self.goal_reward
            done = True

        return self.agent_pos.copy(), reward, done, {}

    def render(self, mode='human'):
        if not self.render_mode:
            return
        self.window.fill(self.COLORS['bg'])
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, self.COLORS['grid'], rect, 1)
        for (ox, oy) in self.obstacles:
            pygame.draw.rect(self.window, self.COLORS['obstacle'],
                             pygame.Rect(oy * self.cell_size, ox * self.cell_size, self.cell_size, self.cell_size))
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        pygame.draw.rect(self.window, self.COLORS['agent'],
                         pygame.Rect(ay * self.cell_size, ax * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.window, self.COLORS['goal'],
                         pygame.Rect(gy * self.cell_size, gx * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def close(self):
        if self.render_mode:
            pygame.quit()