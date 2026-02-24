import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO




class RandomGoalLineWorld(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.min_position = 0
        self.max_position = 100

        self.render_mode = render_mode

        # 0 = left, 1 = right
        self.action_space = spaces.Discrete(2)

        # observation = [position, goal]
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([100, 100], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.goal = None
        self.steps = None
        self.steps = None
        self.min_steps = None
        self.max_reward = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # random start
        start = self.np_random.integers(0, 101)

        # random goal khác start
        while True:
            goal = self.np_random.integers(0, 101)
            if goal != start:
                break

        self.goal = goal
        self.state = np.array([start, goal], dtype=np.float32)
        self.min_steps = abs(start - goal)
        self.max_reward = 100

        if self.render_mode == "human":
            self.render()

        return self.state, {
            'start': start,
            'goal': goal,
            'min_steps': self.min_steps,
            'max_reward': self.max_reward
        }

    def step(self, action):
        position = int(self.state[0])
        goal = int(self.state[1])

        # di chuyển
        if action == 0:
            position -= 1
        else:
            position += 1

        position = np.clip(position, self.min_position, self.max_position)

        # mỗi bước
        self.steps += 1
        reward = self.max_reward - self.steps

        terminated = False
        truncated = False

        # tới goal
        if position == goal:
            terminated = True
        # chạm biên
        elif position == self.min_position or position == self.max_position:
            reward = -self.steps
            terminated = True
        # hết bước
        elif self.steps >= 150: 
            reward = -self.steps
            truncated = True

        self.state = np.array([position, goal], dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}

    def render(self):
        position = int(self.state[0])
        goal = int(self.state[1])

        line = ["-"] * (self.max_position + 1)
        line[position] = "A"
        line[goal] = "G"

        print("".join(line))
        print(f"steps: {self.steps}")

    def close(self):
        pass

model = None
model_path = "sb3_test2.4.zip"
if os.path.exists(model_path):
    model = PPO.load("sb3_test2.4")

if not model:
    env = RandomGoalLineWorld()
    model = PPO("MlpPolicy", env, seed=42, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("sb3_test2.4")

env = RandomGoalLineWorld(render_mode="human")
obs, info = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

print(f"Start: {info['start']}")
print(f"Goal: {info['goal']}")
print(f"Min steps: {info['min_steps']}")
print(f"Max Reward: {info['max_reward']}")
print("Final Reward:", reward)