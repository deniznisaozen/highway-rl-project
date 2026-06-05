# 🚗 Autonomous Highway Driving with Reinforcement Learning

![Evolution](assets/evolution.gif)

**Student:** DENIZ NISA OZEN - LEVENT SARIOGLU
**Course:** CMP4501 – Applied Reinforcement Learning
**Track:** Option A – Autonomous Driving with Highway
**Github Link:** https://github.com/deniznisaozen/highway-rl-project
---

## 📌 Project Overview

This project trains a **Deep Q-Network (DQN)** agent to drive autonomously on a simulated multi-lane highway. The agent must navigate dense traffic by balancing three competing objectives: **maintaining high speed**, **avoiding collisions**, and **keeping lane discipline**. The environment is provided by [Highway-Env](https://highway-env.farama.org/) and the agent is trained using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

---

## 🎬 Evolution Video

The GIF above shows the agent's progression across three training stages:

| Stage | Behavior |
|-------|----------|
| **Untrained** | The agent takes random actions — it drifts between lanes aimlessly, often colliding with other vehicles immediately. |
| **Half-Trained** | The agent has learned basic collision avoidance but still makes poor lane-change decisions, occasionally crashing. |
| **Fully Trained** | The agent drives smoothly at high speed, changes lanes to overtake slower vehicles, and avoids collisions consistently. |

---

## 🔬 Methodology

### a. The Reward Function

The reward function is a weighted combination of three objectives designed to produce safe, fast, and disciplined driving behavior:

R_t = α × speed_reward + β × collision_penalty + γ × lane_reward

Where:

| Symbol | Value | Description |
|--------|-------|-------------|
| $\alpha$ | 0.4 | **High speed reward weight.** Encourages the agent to maintain speed within the target range [20, 30] m/s. The reward is linearly interpolated: 0 at 20 m/s, maximum at 30 m/s. |
| $\beta$ | -1.0 | **Collision penalty.** A harsh penalty applied when the agent collides with another vehicle. This is the strongest signal, ensuring safety is prioritized over speed. |
| $\gamma$ | 0.1 | **Right lane reward weight.** Provides a small incentive for staying in the rightmost lane, mimicking real-world highway driving conventions. |

**Why this reward function?**

The collision penalty ($\beta = -1.0$) is deliberately set as the most extreme value to ensure the agent learns that crashing is never acceptable. The speed reward ($\alpha = 0.4$) is the second-strongest signal, pushing the agent to drive fast rather than crawl safely. The lane reward ($\gamma = 0.1$) is intentionally small — it guides the agent toward the right lane without overriding the primary objectives of speed and safety.

### b. The Model

**Algorithm: Deep Q-Network (DQN)**

DQN was selected for the following reasons:

1. **Discrete action space compatibility:** Highway-env uses 5 discrete actions (lane left, idle, lane right, faster, slower). DQN is purpose-built for discrete action spaces, unlike actor-critic methods that are more naturally suited for continuous actions.
2. **Experience replay:** DQN stores past experiences in a replay buffer and samples from them randomly during training. This breaks temporal correlations between consecutive observations and leads to more stable learning.
3. **Target network:** DQN uses a separate target network that is updated periodically, which prevents the moving-target problem where the Q-values being learned change with every update.
4. **Proven track record:** DQN is one of the most well-established RL algorithms for discrete control tasks and has been shown to work well with highway-env in the literature.

**Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | $5 \times 10^{-4}$ | Standard rate for DQN — fast enough to learn within 30K steps. |
| Discount Factor ($\gamma$) | 0.99 | High discount factor to encourage the agent to plan ahead and avoid short-sighted decisions. |
| Replay Buffer Size | 50,000 | Large enough to store diverse experiences for stable learning. |
| Learning Starts | 1,000 | Number of random exploration steps before training begins, ensuring the buffer has enough diversity. |
| Batch Size | 64 | Standard mini-batch size for stable gradient estimates. |
| Target Update Interval | 500 | How often the target network is synchronized with the main network. |
| Exploration Fraction | 0.3 | 30% of training is spent transitioning from full exploration to near-greedy behavior. |
| $\varepsilon$ Initial / Final | 1.0 / 0.05 | Starts fully random, decays to 5% randomness for continued exploration. |
| Tau ($\tau$) | 1.0 | Hard target update (full copy) rather than soft update. |

**Neural Network Architecture:**

```
Input (5×5 = 25 features, flattened)
    │
    ▼
Linear(25 → 256) + ReLU
    │
    ▼
Linear(256 → 256) + ReLU
    │
    ▼
Output: Linear(256 → 5)  [Q-value for each action]
```

A two-layer MLP with 256 units per layer was chosen because the observation space is small (5×5 matrix). The network outputs 5 Q-values, one for each possible action. The action with the highest Q-value is selected during inference.

### c. States and Actions

**Observation Space (State):**

The agent observes a **5×5 kinematics matrix** where each row represents a vehicle (the ego vehicle + 4 nearest neighbors), and each column represents a feature:

| Column | Feature | Description |
|--------|---------|-------------|
| 0 | `presence` | Whether a vehicle exists in this slot (1.0 or 0.0) |
| 1 | `x` | Longitudinal position (normalized) |
| 2 | `y` | Lateral position (normalized) |
| 3 | `vx` | Longitudinal velocity (normalized) |
| 4 | `vy` | Lateral velocity (normalized) |

All values are normalized to [0, 1] for stable neural network training.

**Action Space:**

The agent chooses from 5 discrete actions:

| Action ID | Name | Effect |
|-----------|------|--------|
| 0 | `LANE_LEFT` | Change one lane to the left |
| 1 | `IDLE` | Maintain current lane and speed |
| 2 | `LANE_RIGHT` | Change one lane to the right |
| 3 | `FASTER` | Accelerate |
| 4 | `SLOWER` | Decelerate |

---

## 📊 Training Analysis

### a. Reward Graph

![Training Results](assets/reward_plot.png)

### b. Commentary

**Early Training (Exploration Phase):**
During the first ~1,000 steps, the agent takes completely random actions due to the `learning_starts` parameter. Rewards are low and highly variable, as the agent frequently crashes or drives slowly by chance. This phase fills the replay buffer with diverse experiences.

**Mid Training (Epsilon Decay):**
As epsilon decays from 1.0 toward 0.05, the agent gradually shifts from random exploration to learned behavior. The rolling average reward climbs steadily as the agent learns to avoid collisions and maintain speed. The Q-network begins to accurately predict which actions lead to higher cumulative rewards.

**Late Training (Exploitation):**
Once exploration is mostly complete, the agent exploits its learned policy. Rewards stabilize at a higher level with reduced variance, indicating the agent has found a reliable driving strategy. The target network updates help maintain stability during this phase.

**Episode Length Analysis:**
Episode length increases over training, reflecting that the agent survives longer. Short episodes (< 20 steps) indicate crashes, while full-length episodes (200 steps) indicate the agent completed the entire duration without collision.

---

## ⚠️ Challenges and Failures

### Challenge: Policy Collapse with PPO

The first version of this project used **PPO (Proximal Policy Optimization)** instead of DQN. While PPO is generally a strong algorithm, it exhibited **policy collapse** in this environment: the agent would learn reasonably well during the first half of training, then catastrophically forget its learned behavior in the second half. The fully-trained PPO agent sometimes performed worse than the half-trained one.

**Root cause:** PPO is an on-policy algorithm that discards old experience after each update. In highway-env, the traffic scenarios are highly variable — some episodes place the ego vehicle in an easy position, others in a very difficult one. PPO's lack of experience replay meant it would overfit to recent (possibly easy) scenarios and then fail when encountering harder ones.

**Solution:** Switching to **DQN**, which uses a replay buffer to store and reuse past experiences, immediately solved this problem. The replay buffer ensures the agent learns from a diverse mix of scenarios, preventing catastrophic forgetting. DQN's target network also provided additional stability by preventing the Q-value targets from changing too quickly.

---

## 🗂️ Repository Structure

```
highway-rl-project/
│
├── README.md                  # This report
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── src/
│   ├── config.py              # All hyperparameters (separated from logic)
│   ├── model.py               # DQN model creation and loading
│   ├── train.py               # Training pipeline with checkpoints
│   ├── evaluate.py            # Evaluation and video generation
│   └── utils.py               # Environment setup, callbacks, plotting
├── assets/
│   ├── evolution.gif           # Training evolution video
│   └── reward_plot.png         # Training performance graph
└── checkpoints/
    ├── model_untrained.zip     # Before training
    ├── model_halftrained.zip   # Midway checkpoint
    └── model_trained.zip       # Final trained model
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the agent
cd src
python train.py

# Generate evolution video
python evaluate.py
```

---

## 📚 References

- [Highway-Env Documentation](https://highway-env.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [DQN Paper — Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
