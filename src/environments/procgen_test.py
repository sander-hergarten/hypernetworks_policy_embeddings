# %%
# Imports
from src.environments import procgen_names, ProcgenEnvironment
import tensorflow as tf
import numpy as np

# %%
# Requirements
""" 
    All The environments must:
        - Load completely
        - Have a observation_spec and action_spec
        - Run parallel 
        - be accessable with the interface
"""

# %%
# Simulate run


def test_load_environments():
    for env_name in procgen_names:
        print(f"Testing {env_name}")

        tf_env = ProcgenEnvironment(env_name).to_tf_env() 

        time_step = tf_env.reset()
        rewards = []
        steps = []
        num_episodes = 1

        for _ in range(num_episodes):
            episode_reward = 0
            episode_steps = 0

            action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
            time_step = tf_env.step(action)
            episode_steps += 1
            episode_reward += time_step.reward.numpy()  # type: ignore

            rewards.append(episode_reward)
            steps.append(episode_steps)
            time_step = tf_env.reset()

        num_steps = np.sum(steps)
        avg_length = np.mean(steps)
        avg_reward = np.sum(rewards) / 5

        print("num_episodes:", num_episodes, "num_steps:", num_steps)
        print("avg_length", avg_length, "avg_reward:", avg_reward)


def test_tracking_interface():
    class TestTracker:
        testing_rendering_env = False
        rendering_env = True
        def reset(self, **kwargs):
            print(f"is and episode in progress:{kwargs['episode_in_progress']()}")

            kwargs["switch_to_default_env" 
                    if self.testing_rendering_env 
                    else "switch_to_render_env"]()
            
            self.testing_rendering_env = not self.testing_rendering_env


        def __call__(self, step_data, **kwargs):
            assert self.testing_rendering_env == ("rgb" in step_data[-1].keys())
    
    
    env = ProcgenEnvironment("coinrun")
    tracker = TestTracker()

    env.add_tracker(tracker)

    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
    env.step(0)

    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
# %%
