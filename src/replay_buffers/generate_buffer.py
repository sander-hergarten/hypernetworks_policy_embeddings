# Imports
import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.drivers import dynamic_episode_driver


class BufferGenerator:
    """This Class defines the interface for all buffer generators"""

    buffer = None

    def __init__(self, tf_env) -> None:
        """initalizes the buffer generator

        Args:
            tf_env (tf.Environment): the environment to generate the buffer with
            agent (tf.Agent): the agent that acts in the environment
        """
        self.tf_env = tf_env

    def get_last_buffer(self) -> tf_uniform_replay_buffer.TFUniformReplayBuffer | None:
        return self.buffer

    def generate_buffer(
        self, num_episodes: int, data_spec, policy
    ) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:

        print(f"generating buffer {num_episodes} episodes")
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=num_episodes,
        )

        replay_observer = [replay_buffer.add_batch]

        dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            policy,
            observers=replay_observer,
            num_episodes=num_episodes,
        ).run()

        self.buffer = replay_buffer
        return replay_buffer
