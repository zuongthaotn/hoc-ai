import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class OneDNavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Action: 0 = left (-1), 1 = right (+1)
        self.action_space = spaces.Discrete(2)

        # Observation: normalized distance in [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.max_steps = 150
        self.min_pos = 0
        self.max_pos = 100

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.start = np.random.randint(0, 101)
        self.pos = self.start

        self.goal = np.random.randint(0, 101)
        while self.goal == self.pos:
            self.goal = np.random.randint(0, 101)

        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        distance = self.goal - self.pos
        distance_norm = distance / 100.0
        return np.array([distance_norm], dtype=np.float32)

    def step(self, action):
        old_distance = abs(self.goal - self.pos)

        # Apply action
        if action == 0:
            self.pos -= 1
        elif action == 1:
            self.pos += 1

        # Clamp to boundary
        self.pos = np.clip(self.pos, self.min_pos, self.max_pos)

        self.steps += 1

        new_distance = abs(self.goal - self.pos)

        # Reward shaping
        reward = old_distance - new_distance

        terminated = self.pos == self.goal
        truncated = self.steps >= self.max_steps

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {}
        )



if __name__ == "__main__":
    model = None
    model_path = "sb3_test2-5.zip"
    if os.path.exists(model_path):
        model = PPO.load("sb3_test2-5")

    env = OneDNavigationEnv()
    if not model:
        # Check environment validity
        check_env(env)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            gamma=0.99,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.0
        )
        model.learn(total_timesteps=50_000)
        model.save("sb3_test2-5")

    for i in range(5):
        print("-------------------------------------------")
        # Test trained agent
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print("Start:", env.start)
        print("Goal:", env.goal)
        print("Total reward:", total_reward)