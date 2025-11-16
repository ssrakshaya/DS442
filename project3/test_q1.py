import gymnasium as gym
env = gym.make("Blackjack-v1", render_mode="human") # Initializing environments
observation, info = env.reset()

for _ in range(50):
    action = env.action_space.sample() # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()