import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class DB3TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=10, initial_balance=10000):
        super(DB3TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Action: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)

        # Observation: window_size x 4 (O,H,L,C)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 4),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position = 0  # 0 = no position, 1 = long
        self.current_step = self.window_size

        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.iloc[
            self.current_step - self.window_size:self.current_step
        ][["Open", "High", "Low", "Close"]].values

        return obs.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0

        current_price = self.df.iloc[self.current_step]["Close"]

        # Buy
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price

        # Sell
        elif action == 2 and self.position == 1:
            profit = current_price - self.entry_price
            reward = profit
            self.balance += profit
            self.position = 0

        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            done = True

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}")


df = pd.read_csv("db3_ohlc.csv")
env = DB3TradingEnv(df)


from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)