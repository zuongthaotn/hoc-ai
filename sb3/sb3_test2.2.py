import gymnasium as gym
from gymnasium import spaces
import numpy as np


class OneJumpLineWorld(gym.Env):

    def __init__(self):
        super().__init__()

        self.min_position = 0
        self.max_position = 100

        # action = [move_type, move_steps]
        self.action_space = spaces.MultiDiscrete([2, 101])

        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([100, 100], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start = self.np_random.integers(0, 101)

        while True:
            goal = self.np_random.integers(0, 101)
            if goal != start:
                break

        self.state = np.array([start, goal], dtype=np.float32)

        return self.state, {}

    def step(self, action):

        move_type, move_steps = action
        position, goal = self.state

        # Thực hiện đúng 1 lần nhảy
        if move_type == 0:
            new_position = position - move_steps
        else:
            new_position = position + move_steps

        new_position = np.clip(new_position, self.min_position, self.max_position)

        distance = abs(new_position - goal)
        reward = 100 - distance

        # 👉 Kết thúc ngay sau 1 action
        terminated = True
        truncated = False

        self.state = np.array([new_position, goal], dtype=np.float32)

        return self.state, reward, terminated, truncated, {}

from stable_baselines3 import PPO

env = OneJumpLineWorld()
model = PPO("MlpPolicy", env, ent_coef=0.0, verbose=1)
model.learn(total_timesteps=50000)

for i in range(4):
    obs, _ = env.reset()

    action, _ = model.predict(obs, deterministic=True)

    new_obs, reward, terminated, truncated, _ = env.step(action)

    start, goal = obs
    end_position = new_obs[0]

    direction = "right" if end_position > start else "left"

    print(f"Start: {int(start)}")
    print(f"Goal: {int(goal)}")
    print(f"Jump {abs(int(end_position - start))} steps to {direction}")
    print(f"Reward: {reward}")
    print("-------------------------------------------")