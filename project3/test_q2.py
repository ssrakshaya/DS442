import gymnasium as gym
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode="human", is_slippery=True) #initialization
observation, info = env.reset()

for _ in range(50):
    action = env.action_space.sample() # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()