"""
Evaluation and video generation script.
Loads checkpoints and creates evolution GIF showing
untrained → half-trained → fully trained agent.
"""

import os
from typing import List, Optional

import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import EnvConfig, VideoConfig
from model import load_model
from utils import evaluate_agent, make_env


def record_episode_frames(
    model: object,
    env_config: EnvConfig,
    seed: int = 42,
) -> List[np.ndarray]:
    """Record frames from a single episode.

    Args:
        model: Trained (or untrained) SB3 model.
        env_config: Environment configuration.
        seed: Random seed.

    Returns:
        List of RGB frames as numpy arrays.
    """
    env = gym.make(
        env_config.env_id,
        render_mode="rgb_array",
        config=env_config.to_env_config(),
    )
    obs, _ = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    return frames


def find_best_seed(
    model: object,
    env_config: EnvConfig,
    seeds: List[int],
) -> int:
    """Try multiple seeds and return the one with the longest episode.

    Args:
        model: SB3 model.
        env_config: Environment configuration.
        seeds: List of seeds to try.

    Returns:
        Seed that produced the longest episode.
    """
    best_seed = seeds[0]
    best_length = 0

    for seed in seeds:
        env = gym.make(
            env_config.env_id,
            render_mode="rgb_array",
            config=env_config.to_env_config(),
        )
        obs, _ = env.reset(seed=seed)
        length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            length += 1
            done = terminated or truncated

        env.close()

        if length > best_length:
            best_length = length
            best_seed = seed

    return best_seed


def create_evolution_gif(
    env_config: EnvConfig,
    video_config: VideoConfig,
    checkpoint_dir: str = "checkpoints",
) -> str:
    """Create a GIF showing agent evolution across training stages.

    Records episodes from untrained, half-trained, and fully trained
    checkpoints, then combines them into a single side-by-side GIF.

    Args:
        env_config: Environment configuration.
        video_config: Video settings.
        checkpoint_dir: Path to saved model checkpoints.

    Returns:
        Path to the generated GIF file.
    """
    stages = [
        ("Untrained", os.path.join(checkpoint_dir, "model_untrained")),
        ("Half-Trained", os.path.join(checkpoint_dir, "model_halftrained")),
        ("Fully Trained", os.path.join(checkpoint_dir, "model_trained")),
    ]

    candidate_seeds = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50, 100]
    all_stage_frames: List[List[np.ndarray]] = []

    for stage_name, model_path in stages:
        print(f"  Recording {stage_name} agent...")
        model = load_model(model_path, env_config)

        # Find a seed where the agent survives long enough to show behavior
        best_seed = find_best_seed(model, env_config, candidate_seeds)
        print(f"    Best seed: {best_seed}")

        frames = record_episode_frames(model, env_config, seed=best_seed)
        all_stage_frames.append(frames)
        print(f"    Captured {len(frames)} frames")

    # Equalize frame counts (pad shorter ones with last frame)
    max_frames = max(len(f) for f in all_stage_frames)
    for i, frames in enumerate(all_stage_frames):
        while len(frames) < max_frames:
            frames.append(frames[-1])

    # Limit total frames to keep GIF reasonable
    max_gif_frames = min(max_frames, 150)
    step = max(1, max_frames // max_gif_frames)

    # Create side-by-side frames with labels
    combined_frames: List[Image.Image] = []
    stage_names = [s[0] for s in stages]

    for frame_idx in range(0, max_frames, step):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for col, (ax, stage_name) in enumerate(zip(axes, stage_names)):
            ax.imshow(all_stage_frames[col][frame_idx])
            ax.set_title(stage_name, fontsize=16, fontweight="bold", pad=10)
            ax.axis("off")

        plt.tight_layout()

        # Convert matplotlib figure to PIL image
        fig.canvas.draw()
        img_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        combined_frames.append(Image.fromarray(img_array))
        plt.close(fig)

    # Save as GIF
    os.makedirs(video_config.output_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    gif_path = os.path.join("assets", video_config.gif_filename)

    combined_frames[0].save(
        gif_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=1000 // video_config.fps,
        loop=0,
    )

    print(f"  Evolution GIF saved: {gif_path}")
    return gif_path


def main() -> None:
    """Run evaluation and generate evolution video."""
    env_config = EnvConfig()
    video_config = VideoConfig()

    print("=" * 60)
    print("  Evaluation & Video Generation")
    print("=" * 60)

    # Evaluate final model
    print("\n[1/2] Evaluating final model...")
    model = load_model("checkpoints/model_trained", env_config)
    eval_env = make_env(env_config, seed=99)
    evaluate_agent(model, eval_env, n_episodes=10)
    eval_env.close()

    # Generate evolution GIF
    print("\n[2/2] Generating evolution GIF...")
    gif_path = create_evolution_gif(env_config, video_config)

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  GIF: {gif_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
