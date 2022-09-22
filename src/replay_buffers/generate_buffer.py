# Imports
from abc import ABC
from tf_agents.replay_buffers import tf_uniform_replay_buffer

# ABC
class BufferGenerator(ABC):
    """this Abstract Base Class defines the interface for all buffer generators"""

    def __init__(self, tf_env, gym_env) -> None:
        ...

    def generate_buffer(
        self, num_episodes, batch_size
    ) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:
        ...
