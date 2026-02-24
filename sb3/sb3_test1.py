import gymnasium as gym
from stable_baselines3 import PPO # Thuật toán tối ưu nhất hiện nay

# 1. Tạo môi trường
# render_mode="human" để bạn có thể xem Agent chơi trực tiếp
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Khởi tạo Model
# MlpPolicy: Sử dụng mạng thần kinh đa lớp (Multi-layer Perceptron)
# verbose=1: Để hiện thị tiến trình huấn luyện
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# 3. Huấn luyện (Training)
print("Đang huấn luyện... Chờ một chút nhé.")
model.learn(total_timesteps=10000)

# 4. Lưu mô hình (Tùy chọn)
model.save("ppo_lunar_lander")

# 5. Xem thành quả
print("Huấn luyện xong! Xem Agent thể hiện:")
obs, info = env.reset()
for _ in range(1000):
    # Model dự đoán hành động dựa trên quan sát (obs)
    action, _states = model.predict(obs, deterministic=True)
    
    # Thực hiện hành động vào môi trường
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
