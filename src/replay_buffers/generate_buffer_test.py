# %%
# Imports
from src.replay_buffers import generate_buffer
from src.environments import procgen_envs

# %%
# Requirements
"""
    The buffer generator must:
        - Train the agent
        - generate a buffer
"""

#%%
# Tests


def test_DQNBufferGenerator():
    generator = generate_buffer.DQNGenerator(
        procgen_envs.environment_dictionary_tf["bigfish"],
        num_training_steps=10
    )
    buffer = generator.generate_buffer(10)
    assert buffer.num_frames() == 10


# %%
