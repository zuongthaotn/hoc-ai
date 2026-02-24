import gymnasium as gym
from gymnasium import spaces
import numpy as np


import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HiddenGoalLineWorld(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.min_position = 0
        self.max_position = 100

        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)

        # CHỈ cho agent biết position
        self.observation_space = spaces.Box(
            low=np.array([0], dtype=np.float32),
            high=np.array([100], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.goal = None
        self.score = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 50

        start = self.np_random.integers(0, 101)

        while True:
            goal = self.np_random.integers(0, 101)
            if goal != start:
                break

        self.goal = goal
        self.state = np.array([start], dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state, {}

    def step(self, action):

        position = int(self.state[0])

        if action == 0:
            position -= 1
        else:
            position += 1

        position = np.clip(position, self.min_position, self.max_position)

        self.score -= 1
        reward = -1
        terminated = False

        if position == self.goal:
            reward += 100
            terminated = True

        if position == self.min_position or position == self.max_position:
            terminated = True

        self.state = np.array([position], dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        position = int(self.state[0])

        line = ["-"] * (self.max_position + 1)
        line[position] = "A"
        line[self.goal] = "G"

        print("".join(line))
        print(f"Score: {self.score}")

from stable_baselines3 import PPO

env = HiddenGoalLineWorld()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)


env = HiddenGoalLineWorld(render_mode="human")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

print("Final Reward:", reward)