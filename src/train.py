"""
Training script for Highway-Env RL agent.
Trains a PPO agent and saves checkpoints at untrained, half-trained,
and fully trained stages.
"""

import os
import sys

from stable_baselines3.common.callbacks import CheckpointCallback

from config import EnvConfig, TrainConfig
from model import create_model
from utils import RewardLoggerCallback, evaluate_agent, make_env, plot_training_results


def train() -> None:
    """Run the full training pipeline.

    Steps:
        1. Create environment and model
        2. Save untrained checkpoint
        3. Train to halfway and save checkpoint
        4. Train to completion and save final checkpoint
        5. Plot training results
        6. Evaluate the final model
    """
    env_config = EnvConfig()
    train_config = TrainConfig()

    # Create directories
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    os.makedirs(train_config.log_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    print("=" * 60)
    print("  Highway-Env RL Training")
    print(f"  Algorithm: {train_config.algorithm}")
    print(f"  Total timesteps: {train_config.total_timesteps:,}")
    print(f"  Learning rate: {train_config.learning_rate}")
    print(f"  Network: {train_config.net_arch}")
    print("=" * 60)

    # --- Step 1: Create model ---
    print("\n[1/6] Creating model...")
    model = create_model(env_config, train_config)

    # --- Step 2: Save untrained checkpoint ---
    print("\n[2/6] Saving untrained checkpoint...")
    untrained_path = os.path.join(train_config.checkpoint_dir, "model_untrained")
    model.save(untrained_path)
    print(f"  Saved: {untrained_path}")

    # --- Step 3: Train first half ---
    half_steps = train_config.total_timesteps // 2
    print(f"\n[3/6] Training first half ({half_steps:,} timesteps)...")
    reward_callback = RewardLoggerCallback()
    model.learn(total_timesteps=half_steps, callback=reward_callback, progress_bar=False)

    # Save half-trained checkpoint
    halftrained_path = os.path.join(train_config.checkpoint_dir, "model_halftrained")
    model.save(halftrained_path)
    print(f"  Saved: {halftrained_path}")

    # --- Step 4: Train second half ---
    print(f"\n[4/6] Training second half ({half_steps:,} timesteps)...")
    model.learn(
        total_timesteps=half_steps,
        callback=reward_callback,
        progress_bar=False,
        reset_num_timesteps=False,
    )

    # Save fully trained checkpoint
    trained_path = os.path.join(train_config.checkpoint_dir, "model_trained")
    model.save(trained_path)
    print(f"  Saved: {trained_path}")

    # --- Step 5: Plot results ---
    print("\n[5/6] Plotting training results...")
    plot_training_results(
        rewards=reward_callback.episode_rewards,
        lengths=reward_callback.episode_lengths,
        save_path="assets/reward_plot.png",
    )

    # --- Step 6: Evaluate ---
    print("\n[6/6] Evaluating final model...")
    eval_env = make_env(env_config, seed=123)
    evaluate_agent(model, eval_env, n_episodes=5)
    eval_env.close()

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Checkpoints saved in: {train_config.checkpoint_dir}/")
    print(f"  Plot saved: assets/reward_plot.png")
    print("=" * 60)


if __name__ == "__main__":
    train()
