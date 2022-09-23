from typing import Protocol
from dataclasses import dataclass

from tf_agents.trajectories import time_step as ts
from tf_agents import specs

t_spec = specs.TensorSpec | specs.BoundedTensorSpec

@dataclass
class EnvironmentSpec(Protocol):
    observation_spec: t_spec
    action_spec: t_spec
    reward_spec: t_spec
    step_type_spec: t_spec
    time_step_spec: ts.TimeStep
     