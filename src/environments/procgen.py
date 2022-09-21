# %%
# Imports
import gym
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

# %%
# Helpers

gym_to_tf = lambda env: tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(env))
load_gym_env = lambda env_name: gym_to_tf(gym.make(f"procgen:procgen-{env_name}-v0"))

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
envirnment_dictionary = {env_name: load_gym_env(env_name) for env_name in env_names}
