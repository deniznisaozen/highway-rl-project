"""
Configuration file for Highway-Env RL project.
All hyperparameters and environment settings are centralized here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EnvConfig:
    """Highway environment configuration."""

    env_id: str = "highway-v0"
    lanes_count: int = 4
    vehicles_count: int = 15
    duration: int = 40
    simulation_frequency: int = 15
    policy_frequency: int = 5
    observation_type: str = "Kinematics"
    observation_vehicles_count: int = 5
    observation_features: List[str] = field(
        default_factory=lambda: ["presence", "x", "y", "vx", "vy"]
    )
    normalize_observations: bool = True
    collision_reward: float = -1.0
    right_lane_reward: float = 0.1
    high_speed_reward: float = 0.4
    reward_speed_range: List[float] = field(default_factory=lambda: [20, 30])

    def to_env_config(self) -> Dict[str, Any]:
        """Convert to highway-env config dictionary."""
        return {
            "lanes_count": self.lanes_count,
            "vehicles_count": self.vehicles_count,
            "duration": self.duration,
            "simulation_frequency": self.simulation_frequency,
            "policy_frequency": self.policy_frequency,
            "observation": {
                "type": self.observation_type,
                "vehicles_count": self.observation_vehicles_count,
                "features": self.observation_features,
                "normalize": self.normalize_observations,
            },
            "collision_reward": self.collision_reward,
            "right_lane_reward": self.right_lane_reward,
            "high_speed_reward": self.high_speed_reward,
            "reward_speed_range": self.reward_speed_range,
        }


@dataclass
class TrainConfig:
    """Training hyperparameters for DQN."""

    algorithm: str = "DQN"
    total_timesteps: int = 30_000
    learning_rate: float = 5e-4
    batch_size: int = 64
    gamma: float = 0.99
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    tau: float = 1.0
    train_freq: int = 4
    target_update_interval: int = 500
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "ReLU"
    verbose: int = 0
    seed: int = 42

    # Checkpoint settings
    save_freq: int = 10_000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Evaluation settings
    eval_episodes: int = 10
    eval_freq: int = 5_000


@dataclass
class VideoConfig:
    """Video recording settings."""

    fps: int = 10
    untrained_episodes: int = 3
    halftrained_episodes: int = 3
    trained_episodes: int = 3
    output_dir: str = "videos"
    gif_filename: str = "evolution.gif"
