# %%
# Imports
from dataclasses import dataclass

import gym
import procgen

import tensorflow as tf

from tf_agents import specs
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

# %%
# Helpers

gym_to_tf = lambda env: tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(env))
load_gym_env = lambda env_name: gym.make(f"procgen-{env_name}-v0")

# %%
# List Environemnents
env_names = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

# %%
# Load Environments
environment_dictionary_gym = {
    env_name: load_gym_env(env_name) for env_name in env_names
}
environment_dictionary_tf = {
    env_name: gym_to_tf(env) for env_name, env in environment_dictionary_gym.items()
}
# %%
@dataclass
class spec:
    reward_spec = specs.TensorSpec(shape=(), dtype=tf.float32, name="reward")  # type: ignore
    step_type_spec = specs.TensorSpec(shape=(), dtype=tf.int32, name="step_type")  # type: ignore
    action_spec = specs.BoundedTensorSpec([], tf.int32, minimum=0, maximum=14)  # type: ignore

    discount_spec = specs.BoundedTensorSpec(
        shape=(), dtype=tf.float32, minimum=0, maximum=1, name="discount"
    )  # type: ignore
    observation_spec = specs.TensorSpec(
        shape=(64, 64, 3), dtype=tf.uint8, name="observation"
    )  # type: ignore
    time_step_spec = ts.time_step_spec(
        observation_spec=observation_spec, reward_spec=reward_spec
    )  # type: ignore
