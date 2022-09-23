# %%
# Imports
from src.replay_buffers import generate_buffer
from src.environments import procgen_envs
from tf_agents.policies import random_tf_policy

# %%
# Requirements
"""
    The buffer generator must:
        - Train the agent
        - generate a buffer
"""

#%%
# Tests


def test_BufferGenerator():
    policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=procgen_envs.spec.time_step_spec,
        action_spec=procgen_envs.spec.action_spec,
    )

    data_spec = policy.collect_data_spec

    generator = generate_buffer.BufferGenerator(
        procgen_envs.environment_dictionary_tf["bigfish"],
    )

    buffer = generator.generate_buffer(10, policy=policy, data_spec=data_spec)
    assert buffer.num_frames() == 10
