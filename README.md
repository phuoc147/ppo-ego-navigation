# PPO + Transformer for Multi-Agent Ego Vehicle Navigation

This project applies deep reinforcement learning (PPO) with a transformer-based module to train an ego vehicle to navigate a 2D environment. The ego car learns to reach a target while avoiding collisions with surrounding vehicles, using temporal state histories of nearby agents.

## 🚗 Project Description

- A custom Pygame environment simulates multiple vehicles in a bounded 2D space.
- The ego vehicle (in red) is trained using Proximal Policy Optimization (PPO).
- A Transformer Encoder is used to model temporal behavior of nearby agents and enhance the ego's decision-making.
- The project explores how attention-based models can improve multi-agent interaction reasoning in simplified autonomous driving scenarios.

## 🧠 Core Features

- **PPO Algorithm**: Stable on-policy reinforcement learning.
- **Transformer-based Agent Modeling**: Encodes sequence histories of other agents to improve context awareness.
- **Reward Shaping**:
  - Positive reward for getting closer to the target.
  - Penalty for collisions and inefficient movements.
- **Visualization**: Real-time rendering of the environment using Pygame.

## 🗂️ Training and render the result
- Sample training command: python train.py  --total_timesteps 50000 --saved_file ppo_trans --cars_num 50
- Sample result command: python render.py --model ppo_trans --cars_num 50
