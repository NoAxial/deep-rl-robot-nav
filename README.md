# Deep RL Autonomous Mobile Robot Navigation 🤖🚗

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-F0DC4E?style=for-the-badge&logo=python&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

A high-performance, custom-built Reinforcement Learning framework for training autonomous mobile robots to navigate complex, dynamic 2D environments. This project features a **100% Native GPU Physics Simulator** supporting over 4,000 simultaneous environments, paired with a custom from-scratch implementation of the **Soft Actor-Critic (SAC)** algorithm.

---

## ✨ Key Features

- **Massively Parallel GPU Physics Engine:** The entire differential drive physics engine and 2D environment (including collision detection and LIDAR raycasting) is written natively in PyTorch tensors (`batched_env.py`). This allows **4,096+ robots** to be simulated simultaneously on a single GPU, achieving speeds upwards of 50,000 FPS.
- **Custom PyTorch SAC Algorithm:** We built Soft Actor-Critic from scratch (`pytorch_sac.py`) specifically tailored for high-speed batched GPU environments, removing the bottleneck of CPU-bound libraries like Stable Baselines 3.
- **Automated Curriculum Learning:** The training pipeline features a dynamic difficulty scaler. Over the first 20 Million steps, the environment automatically transitions from an empty "Kindergarten" map (0 obstacles, close goals) to a highly chaotic environment (5 high-speed dynamic obstacles, distant goals).
- **Strict "Smooth Driving" Penalties:** The reward function (`config.yaml`) enforces strict penalties on jerky steering, maximum angular velocity spinning, and close-proximity obstacle skirting to guarantee safe, human-like, and smooth navigation.
- **Live Pygame Visualization:** Includes a beautiful, live-rendering testing script (`visualize.py`) to watch the trained robot navigate the environment in real-time, complete with a live analytics dashboard and LIDAR ray visualization.

## 🏗️ Architecture

- **Algorithm:** Soft Actor-Critic (SAC) - Continuous Control
- **State Space:**
  - 32-Ray 360° LIDAR (Stacked 3x = 96 channels)
  - 7 Vector State Features (Relative Goal Dist/Angle, Velocity, Previous Action, Urgency)
- **Action Space:** Continuous `[-1.0, 1.0]` (Acceleration, Steering)
- **Neural Network:**
  - Standard multi-layer perceptron (MLP) architecture tailored for massive batch sizes.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/deep-rl-robot-nav.git
   cd deep-rl-robot-nav
   ```

2. **Install dependencies:**
   Ensure you have PyTorch installed with CUDA support for your specific GPU.
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

**1. High-Speed Batched GPU Training**
Trains the agent using 4,096 parallel environments on the GPU. This is entirely headless and will print progress metrics to the console.
```bash
python train_agent_gpu.py
```

**2. Live Visual Evaluation**
Watch the latest trained model navigate the environment in real-time with Pygame graphics.
```bash
python visualize.py
```

## ⚙️ Configuration
You can easily adjust the difficulty of the environment, the radar parameters, or the penalty weights of the robot by tweaking `config.yaml`. 

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
