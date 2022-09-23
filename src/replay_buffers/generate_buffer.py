# Imports
from abc import ABC
import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents import specs
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.drivers import dynamic_episode_driver


# ABC
class BufferGenerator(ABC):
    """this Abstract Base Class defines the interface for all buffer generators"""

    agent = NotImplemented
    num_train_steps: int = NotImplemented
    is_trained = False

    def __init__(self, tf_env) -> None:
        self.tf_env = tf_env

        ...

    def generate_buffer(
        self, num_episodes: int
    ) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:
        """generates a buffer using the DQN agent. If the agent is not trained, it will train it first

        Args:
            num_episodes (int): the number of episodes to generate the buffer with

        Returns:
            tf_uniform_replay_buffer.TFUniformReplayBuffer: the buffer
        """

        if not self.is_trained:
            self.train()

        self.buffer = self._generate_buffer(num_episodes)
        return self.buffer

    def train(self):
        """trains the agent for self.num_train_steps. When the training is complete, the agent is marked as trained"""
        print("training")
        buffer = self._generate_buffer(self.num_train_steps)
        dataset = buffer.as_dataset(sample_batch_size=4, num_steps=2)

        iterator = iter(dataset)

        for _ in range(self.num_train_steps):
            trajectories, _ = next(iterator)
            loss = self.agent.train(experience=trajectories)

        self.is_trained = True

    def get_last_buffer(self) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:
        return self.buffer

    def _generate_buffer(self, num_episodes):
        print(f"generating buffer {num_episodes} episodes")
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=num_episodes,
        )

        replay_observer = [replay_buffer.add_batch]

        dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=replay_observer,
            num_episodes=num_episodes,
        ).run()

        return replay_buffer


class ProcgenBufferGenerator(BufferGenerator):
    def __init__(self, tf_env) -> None:
        super().__init__(tf_env)

        self.observation_spec = specs.TensorSpec(
            shape=(64, 64, 3), dtype=tf.uint8, name="observation"
        )
        self.discount_spec = specs.BoundedTensorSpec(
            shape=(), dtype=tf.float32, minimum=0, maximum=1, name="discount"
        )
        self.reward_spec = specs.TensorSpec(shape=(), dtype=tf.float32, name="reward")
        self.step_type_spec = specs.TensorSpec(
            shape=(), dtype=tf.int32, name="step_type"
        )
        self.action_spec = specs.BoundedTensorSpec([], tf.int32, minimum=0, maximum=14)

        self.time_step_spec = ts.time_step_spec(
            observation_spec=self.observation_spec, reward_spec=self.reward_spec
        )


# Generators
class DQNGenerator(ProcgenBufferGenerator):
    """this class generates a buffer using a DQN agent"""

    def __init__(self, tf_env, num_training_steps: int = 10000) -> None:
        super().__init__(tf_env)

        self.num_train_steps = num_training_steps

        q_net = q_network.QNetwork(
            self.observation_spec,
            self.action_spec,
            fc_layer_params=(100,),
        )

        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.time_step_spec,
            self.action_spec,
            q_network=q_net,
            optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
            train_step_counter=self.train_step_counter,
        )

        self.agent.initialize()
