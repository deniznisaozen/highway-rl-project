"""
Utility functions for environment creation, plotting, and helpers.
"""

import os
from typing import List, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401 — registers highway envs
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import EnvConfig


def make_env(env_config: EnvConfig, seed: int = 42) -> gym.Env:
    """Create and configure a Highway environment.

    Args:
        env_config: Environment configuration dataclass.
        seed: Random seed for reproducibility.

    Returns:
        Configured Gymnasium environment wrapped with Monitor.
    """
    env = gym.make(
        env_config.env_id,
        render_mode="rgb_array",
        config=env_config.to_env_config(),
    )
    env.reset(seed=seed)
    env = Monitor(env)
    return env


class RewardLoggerCallback(BaseCallback):
    """Custom callback to log episode rewards during training.

    Stores per-episode rewards and lengths for later plotting.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self._current_reward: float = 0.0
        self._current_length: int = 0

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Monitor wrapper stores episode info in 'infos'
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


def plot_training_results(
    rewards: List[float],
    lengths: List[float],
    save_path: str,
    window: int = 20,
) -> None:
    """Plot and save training reward and episode length graphs.

    Args:
        rewards: List of episode rewards.
        lengths: List of episode lengths.
        save_path: Path to save the plot image.
        window: Rolling average window size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    episodes = np.arange(1, len(rewards) + 1)

    # --- Reward plot ---
    axes[0].plot(episodes, rewards, alpha=0.3, color="steelblue", label="Raw reward")
    if len(rewards) >= window:
        rolling_avg = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        axes[0].plot(
            episodes[window - 1 :],
            rolling_avg,
            color="darkblue",
            linewidth=2,
            label=f"Rolling avg ({window} ep)",
        )
    axes[0].set_xlabel("Episode", fontsize=12)
    axes[0].set_ylabel("Total Reward", fontsize=12)
    axes[0].set_title("Training Reward over Episodes", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Episode length plot ---
    axes[1].plot(episodes, lengths, alpha=0.3, color="coral", label="Raw length")
    if len(lengths) >= window:
        rolling_avg_len = np.convolve(
            lengths, np.ones(window) / window, mode="valid"
        )
        axes[1].plot(
            episodes[window - 1 :],
            rolling_avg_len,
            color="darkred",
            linewidth=2,
            label=f"Rolling avg ({window} ep)",
        )
    axes[1].set_xlabel("Episode", fontsize=12)
    axes[1].set_ylabel("Episode Length (steps)", fontsize=12)
    axes[1].set_title("Episode Length over Episodes", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training plot saved to: {save_path}")


def evaluate_agent(
    model: object,
    env: gym.Env,
    n_episodes: int = 10,
) -> Tuple[float, float]:
    """Evaluate a trained agent over multiple episodes.

    Args:
        model: Trained SB3 model.
        env: Gymnasium environment.
        n_episodes: Number of evaluation episodes.

    Returns:
        Tuple of (mean_reward, std_reward).
    """
    all_rewards: List[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        all_rewards.append(total_reward)

    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f} over {n_episodes} episodes")
    return mean_reward, std_reward
