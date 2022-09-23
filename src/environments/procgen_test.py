# %%
# Imports
from src.environments import procgen_envs
import tensorflow as tf
import numpy as np

# %%
# Requirements
""" 
    All The environments must:
        - Load completely
        - Have a observation_spec and action_spec
        - Run parallel 
"""

# %%
# Simulate run


def test_load_environments():
    for env_name, env in procgen_envs.environment_dictionary_gym.items():
        print(f"Testing {env_name}")
        tf_env = procgen_envs.environment_dictionary_tf[env_name]

        time_step = tf_env.reset()
        rewards = []
        steps = []
        num_episodes = 1

        for _ in range(num_episodes):
            episode_reward = 0
            episode_steps = 0
            while not time_step.is_last():
                action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
                time_step = tf_env.step(action)
                episode_steps += 1
                episode_reward += time_step.reward.numpy()
            rewards.append(episode_reward)
            steps.append(episode_steps)
            time_step = tf_env.reset()

        num_steps = np.sum(steps)
        avg_length = np.mean(steps)
        avg_reward = np.sum(rewards) / 5

        print("num_episodes:", num_episodes, "num_steps:", num_steps)
        print("avg_length", avg_length, "avg_reward:", avg_reward)


# %%
