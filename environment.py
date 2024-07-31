# File: environment.py
"""Environment setup for Melting Pot Clean-up game."""

from meltingpot import substrate
from shimmy import MeltingPotCompatibilityV0
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


def create_env(num_agents):
    """Create the Melting Pot Clean-up environment."""
    roles = ['default'] * num_agents
    base_env = substrate.get_factory("clean_up").build(roles)
    shimmy_env = MeltingPotCompatibilityV0(base_env, render_mode="rgb_array")
    return PettingZooEnv(shimmy_env)
