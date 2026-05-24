"""
Model definition module.
Creates and configures the DQN agent with specified architecture.
"""

import os
from typing import Optional, Union

import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from config import EnvConfig, TrainConfig
from utils import make_env


# Map string names to PyTorch activation functions
ACTIVATION_MAP = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "LeakyReLU": nn.LeakyReLU,
}


def create_model(
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> DQN:
    """Create a DQN model with the specified configuration.

    Args:
        env_config: Environment configuration.
        train_config: Training hyperparameters.

    Returns:
        Configured DQN model ready for training.
    """
    env = make_env(env_config, seed=train_config.seed)

    activation_fn = ACTIVATION_MAP.get(train_config.activation_fn, nn.ReLU)

    policy_kwargs = {
        "net_arch": train_config.net_arch,
        "activation_fn": activation_fn,
    }

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_config.learning_rate,
        buffer_size=train_config.buffer_size,
        learning_starts=train_config.learning_starts,
        batch_size=train_config.batch_size,
        tau=train_config.tau,
        gamma=train_config.gamma,
        train_freq=train_config.train_freq,
        target_update_interval=train_config.target_update_interval,
        exploration_fraction=train_config.exploration_fraction,
        exploration_initial_eps=train_config.exploration_initial_eps,
        exploration_final_eps=train_config.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=train_config.verbose,
        seed=train_config.seed,
        tensorboard_log=None,
    )

    return model


def load_model(model_path: str, env_config: EnvConfig) -> DQN:
    """Load a previously saved DQN model.

    Args:
        model_path: Path to the saved model zip file.
        env_config: Environment configuration for the loaded model.

    Returns:
        Loaded DQN model.
    """
    env = make_env(env_config)
    model = DQN.load(model_path, env=env)
    return model
