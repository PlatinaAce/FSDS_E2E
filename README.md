# FSDS_E2E

An End-to-End (E2E) autonomous driving project for Formula Student racing in the **Formula Student Driverless Simulator (FSDS)**.  
This repository focuses on **Behavioral Cloning (Imitation Learning)**, where a neural network learns steering/throttle/brake behavior directly from expert demonstrations.

## Project Overview

FSDS_E2E aims to train and deploy deep learning driving policies that can complete laps quickly and reliably in simulation.  
The core workflow is:

1. Collect expert driving data in FSDS via ROS 2 topics.
2. Train E2E models with PyTorch using recorded observations and control commands.
3. Evaluate each model in closed-loop simulation for lap time and driving stability.

## Branching Strategy & Architectures

To evaluate architectures in a clean and reproducible way, this repository is **branch-structured by model type**:

- `cnn/*` branches: convolution-based baselines
- `rnn/*` or `cnn-rnn/*` branches: temporal models (e.g., CNN + LSTM/GRU)
- `vit/*` branches: Vision Transformer variants

> **Important:** the `main` branch is reserved for the current best-performing model (state-of-the-art for this project), selected by evaluation metrics such as lap time and consistency.

To inspect a specific architecture implementation:

```bash
git checkout <branch-name>
```

Examples:

```bash
git checkout cnn/baseline
git checkout cnn-rnn/lstm_policy
git checkout vit/patch16_policy
```

## Prerequisites

Ensure the following are installed:

- **Ubuntu** (recommended for ROS 2 + FSDS workflows)
- **ROS 2 Humble** (recommended/tested baseline; newer compatible releases may also work)
- **FSDS** (Formula Student Driverless Simulator)
- **Python 3** with **PyTorch**

## Installation & Setup

Clone and build the ROS 2 workspace:

Replace `<ros2-distro>` in the commands below with your ROS 2 distribution (for example, `humble`).

```bash
mkdir -p ~/fsds_ws/src
cd ~/fsds_ws/src
git clone https://github.com/PlatinaAce/FSDS_E2E.git
cd ..
source /opt/ros/<ros2-distro>/setup.bash
colcon build
source install/setup.bash
```

## Data Collection Pipeline

Expert driving data is collected inside FSDS using ROS 2:

1. Run FSDS with the target track/scenario.
2. Launch an expert controller (human teleoperation or scripted controller).
3. Subscribe to sensor topics (e.g., camera images, vehicle state).
4. Record synchronized control commands (steering/throttle/brake).
5. Save samples into a training dataset format for PyTorch.

Typical ROS 2 tools used:

- `ros2 topic echo` / custom subscribers for live inspection
- `ros2 bag record` for synchronized logging and replay

## Training & Evaluation

General model development loop:

1. **Prepare dataset** from collected expert runs.
2. **Train** a selected architecture branch with PyTorch.
3. **Validate** on held-out data to monitor loss and overfitting.
4. **Deploy** the trained policy node in FSDS.
5. **Evaluate** closed-loop performance (lap time, completion rate, smoothness, off-track events).
6. Promote the best model branch to `main` once it is verified as top-performing.

---

For branch-specific commands, hyperparameters, and launch files, check the README/docs within each architecture branch.
