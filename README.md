# Deep RL Autonomous Mobile Robot Navigation 🤖🚗

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Stable Baselines 3](https://img.shields.io/badge/Stable_Baselines3-20B2AA?style=for-the-badge)
![Pygame](https://img.shields.io/badge/Pygame-F0DC4E?style=for-the-badge&logo=python&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

A high-performance, custom-built Reinforcement Learning framework for training autonomous mobile robots to navigate complex, dynamic 2D environments. This project uses **Soft Actor-Critic (SAC)** paired with a custom **1D Convolutional Neural Network (CNN)** to process LIDAR data and predict dynamic obstacle motion in real-time.

---

## ✨ Key Features

- **Differential Drive Physics Engine:** Built natively in `numpy` using mathematical AABB (Axis-Aligned Bounding Box) and radial collision algorithms coupled with a custom Extended Kalman Filter (EKF) for precise odometry tracking.
- **Temporal Motion Prediction:** Uses a `VecFrameStack` of the last 4 frames fed into a custom PyTorch 1D CNN over the spatial LIDAR dimension. The agent natively calculates velocity and predicts the future positions of dynamic obstacles.
- **Automated Curriculum Learning:** Features a dynamic difficulty scaler (`curriculum.py`) that monitors the agent's rolling success rate. It automatically scales obstacle density, shrinks the goal radius, and activates dynamic moving obstacles as the agent masters early stages.
- **Dual-Hardware Asynchronous Execution:** Fully optimized to run the vectorized environments (Simulator) on the CPU while offloading the SAC neural network gradient backpropagation to a CUDA-enabled GPU.
- **Live Pygame Visualization:** Includes a beautiful, live-rendering training dashboard to watch the robot learn in real-time with comprehensive graphics.

## 🏗️ Architecture

- **Algorithm:** Soft Actor-Critic (SAC) - Continuous Control
- **State Space:** 42 Dimensions
  - 32-Ray 360° LIDAR (Stacked 4x = 128 channels)
  - 8 Vector State Features (Relative Goal Dist/Angle, Velocity, Previous Action, Urgency)
- **Action Space:** Continuous `[-1.0, 1.0]` (Acceleration, Steering)
- **Neural Network:**
  - LIDAR Pathway: 1D CNN `(In: 4, Out: 16) -> (In: 16, Out: 32) -> Flatten`
  - State Pathway: `MLP(64) -> MLP(64)`
  - Merged Pathway: `MLP(256)`

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

The project is governed by a unified command-line interface `run.py`. Configurations and hyperparameter tuning can be found in `config.py` or overridden via YAML configs in the `configs/` directory.

**1. Headless High-Speed Training**
Trains the agent purely in the console at maximum CPU/GPU utilization.
```bash
python run.py train
```

**2. Live Visual Training**
Trains the agent while rendering the environment and live analytics dashboard via Pygame.
```bash
python train_visual.py
```

**3. Evaluate / Visualize a Trained Model**
Watch a trained model navigate the environment. (Requires `best_model/` or a saved `.zip` file).
```bash
python run.py view
```

## ⚙️ Configuration
You can easily adjust the difficulty of the environment or the parameters of the robot by tweaking `config.py`. The project comes with built-in presets:
```bash
python run.py train --config configs/hard.yaml
```

## 📊 Analytics
Run the analytics dashboard to generate charts and evaluate the convergence of your saved models.
```bash
python run.py analyze
```

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
