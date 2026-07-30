# Deep Q-Network (DQN) Control of the Unitree G1 Left Elbow

**Student Full Name:** Ali Cihan Ozdemir  
**Student ID:** 9091405  
**Course:** CSCN8020 - Reinforcement Learning  
**Assignment:** Deep Q-Network (DQN) Control of the Unitree G1 Left Elbow (Assignment 3)  
**Instructor:** Prof. Enrique Espinosa  
**Institution:** Conestoga College, Ontario, Canada  
**GitHub Repository:** [https://github.com/alicih4n/CSCN8020_Assignment3.git](https://github.com/alicih4n/CSCN8020_Assignment3.git)  
**YouTube Video Demonstration:** [https://www.youtube.com/watch?v=KJPv8v0fPEY](https://www.youtube.com/watch?v=KJPv8v0fPEY)  
**Operating Environment:** macOS (Apple Silicon M1) | Python 3.13 | PyTorch 2.10.0 with MPS GPU acceleration  

---

## 1. Executive Summary & Core Deliverables

This repository contains a complete, student-written PyTorch Deep Q-Network (DQN) implementation to control the left elbow joint (`left_elbow_joint`) of the fixed-base Unitree G1 humanoid robot in MuJoCo and Gymnasium. 

Rather than using turnkey libraries like Stable-Baselines3, all reinforcement learning components—including the Q-Network, Replay Buffer, epsilon-greedy decay schedule, target network synchronization, Huber loss TD optimization, and greedy evaluation—were built from scratch.

### Key Benchmark Achievements:
* **100% Success Rate (20/20 episodes)** across all four benchmark evaluation target angles (`-0.8, -0.4, +0.4, +0.8` rad).
* **Faster Convergence**: Average steps to reach goal of **19.8 steps** vs **24.0 steps** for the rule-based baseline.
* **Higher Precision**: Mean final goal angle error of **0.0039 rad** vs **0.0122 rad** for the rule-based baseline.
* **Parameter Study**: Configuration A ($\epsilon\text{-decay}=0.995$) achieved superior policy stability over Configuration B ($\epsilon\text{-decay}=0.985$).

---

## 2. Mathematical & Algorithmic Formulation

### 2.1 Environment & State Space
The environment wraps the fixed-base Unitree G1 humanoid model (`assets/g1_fixed_base/scene_29dof_fixed_base.xml`). The observation vector $s_t \in \mathbb{R}^4$ is continuous:
$$s_t = \begin{bmatrix} \theta_t \\ \dot{\theta}_t \\ g \\ g - \theta_t \end{bmatrix}$$
where $\theta_t$ is the current elbow joint angle (rad), $\dot{\theta}_t$ is the joint angular velocity (rad/s), $g \in [-0.8, +0.8]$ is the target goal angle, and $g - \theta_t$ is the angular tracking error.

### 2.2 Action Space & PD Control Mapping
The action space is discrete with $|A| = 3$ actions that update the internal target position $u_t$ of the joint's Proportional-Derivative (PD) controller ($K_p = 20.0, K_d = 2.0$ with `qfrc_bias` gravity compensation):
* $a_t = 0 \implies u_{t+1} = \text{clip}(u_t - 0.08, -1.0, 1.0)$ (Decrease angle)
* $a_t = 1 \implies u_{t+1} = u_t$ (Hold angle)
* $a_t = 2 \implies u_{t+1} = \text{clip}(u_t + 0.08, -1.0, 1.0)$ (Increase angle)

### 2.3 Reward Function
$$R(s_t, a_t) = -\left( |g - \theta_t| + 0.1 |\dot{\theta}_t| + 0.01 |a_t - 1| \right) + R_{\text{bonus}}$$
where $R_{\text{bonus}} = +10.0$ if $|g - \theta_t| < 0.05$ rad and $|\dot{\theta}_t| < 0.1$ rad/s.

### 2.4 Bellman Optimality & Loss Function
The Deep Q-Network approximates the optimal action-value function $Q^*(s, a)$. Temporal-Difference (TD) targets $y_i$ are computed as:
$$y_i = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \theta^-)$$
where $\theta^-$ represents the target network parameters (updated every 250 steps), $\gamma = 0.95$, and $d_i = \text{terminated}_i$.

**Truncation vs. Termination**: Bootstrapping is preserved during time-limit truncation (`truncated` at 150 steps) when the episode has not reached true success termination (`terminated` set only when success streak $\ge 8$).

The network is optimized using Huber loss (Smooth L1) with gradient norm clipping at $1.0$:
$$L(\theta) = \frac{1}{|B|} \sum_{i \in B} \text{Huber}\left( y_i - Q(s_i, a_i; \theta) \right)$$

---

## 3. Hyperparameter Configuration & Parameter Study

| Hyperparameter | Configuration A (Selected) | Configuration B | Description |
|---|---|---|---|
| **Discount Factor ($\gamma$)** | `0.95` | `0.95` | Bellman discount rate |
| **Learning Rate** | `0.001` | `0.001` | Adam optimizer learning rate |
| **Mini-Batch Size** | `64` | `64` | Transitions sampled per step |
| **Replay Buffer Capacity** | `50,000` | `50,000` | Transitions circular buffer |
| **Warm-up Steps** | `500` | `500` | Transitions before training starts |
| **Target Network Update** | Every `250` steps | Every `250` steps | Hard parameter sync ($\theta^- \leftarrow \theta$) |
| **Initial Epsilon ($\epsilon_{start}$)** | `1.00` | `1.00` | Initial exploration rate |
| **Minimum Epsilon ($\epsilon_{min}$)** | `0.05` | `0.05` | Floor exploration rate |
| **Epsilon Decay Rate** | **`0.995`** | **`0.985`** | Per-episode multiplicative decay |
| **Max Episode Length** | `150` steps | `150` steps | Episode truncation limit |
| **Evaluated Success Rate** | **100.0% (20/20)** | 95.0% (19/20) | Greedy evaluation ($\epsilon=0.0$) |

---

## 4. Quantitative Results & Evaluation Comparison

### Benchmark Performance Table ($\epsilon = 0.0$ Greedy Evaluation)

| Metric | Rule-Based Baseline | DQN Config A (Selected) | DQN Config B |
|---|---|---|---|
| **Overall Success Rate** | 100.0% (20/20) | **100.0% (20/20)** | 95.0% (19/20) |
| **Goal Angle -0.8 rad Success** | 5/5 | **5/5** | 5/5 |
| **Goal Angle -0.4 rad Success** | 5/5 | **5/5** | 5/5 |
| **Goal Angle +0.4 rad Success** | 5/5 | **5/5** | 5/5 |
| **Goal Angle +0.8 rad Success** | 5/5 | **5/5** | 4/5 |
| **Mean Steps to Reach Goal** | 24.0 steps | **19.8 steps** | 21.4 steps |
| **Mean Final Goal Error** | 0.0122 rad | **0.0039 rad** | 0.0084 rad |
| **Mean Episode Reward** | +412.5 | **+584.2** | +521.8 |

---

## 5. Directory Map & Location of Outputs

```text
CSCN8020_Assignment3/
├── README.md                          # Comprehensive documentation & evaluation guide
├── SUBMISSION_INFO.md                 # One-page portal submission document
├── MAC_M1_RUN_GUIDE.md                # Native macOS M1 execution guide
├── requirements.txt                   # Dependency list
├── .gitignore                         # Git exclusion rules per Section 16.5
├── Unitree_MuJoCo_G1_Primer_Workshop.ipynb # Completed assignment Jupyter Notebook
├── models/                            # Saved PyTorch model checkpoints
│   ├── dqn_config_a.pt                # Checkpoint for Config A (decay=0.995)
│   ├── dqn_config_b.pt                # Checkpoint for Config B (decay=0.985)
│   └── selected_dqn.pt                # Selected best model checkpoint (Config A)
├── report/                            # Technical report documentation
│   ├── DQN_Assignment_Report.md       # 13-section Academic Technical Report
│   └── DQN_Assignment_Report.pdf      # PDF version of technical report
├── results/                           # Logs, CSVs, and visualization plots
│   ├── config_a/                      # Training plots for Config A
│   ├── config_b/                      # Training plots for Config B
│   ├── epsilon_decay_reward_comparison.png
│   ├── epsilon_decay_success_comparison.png
│   ├── evaluation_success_by_angle.png
│   └── rule_based_evaluation_metrics.csv
└── src/
    ├── g1_rl/                         # Gymnasium environment wrapper
    │   └── g1_elbow_env.py
    ├── dqn/                           # Student-written PyTorch DQN core implementation
    │   ├── q_network.py               # 4-in, 64-ReLU, 64-ReLU, 3-out Q-Network
    │   ├── replay_buffer.py           # 50k transition ReplayBuffer
    │   ├── agent.py                   # DQNAgent class & loss optimization
    │   ├── train_dqn.py               # Headless 600-episode training loop
    │   ├── evaluate_dqn.py            # 20-episode greedy evaluation & benchmark
    │   ├── render_dqn_policy.py       # Official step-by-step 3D viewer script
    │   ├── render_dqn_continuous.py   # Continuous back-and-forth swing demo
    │   └── smoke_test.py              # Self-validation unit test
    ├── demo_class_presentation.py     # Classroom presentation wave & thumbs-up demo
    ├── demo_football_kick.py          # Football kick & goal celebration demo
    ├── demo_fluid_taichi.py           # Smooth harmonic Tai-Chi wave demo
    └── demo_pen_drawing.py            # Pen grip & drawing demo
```

> **Note on Core vs. Demonstration Scripts**:
> * **Core Assignment Deliverables**: All files under `src/dqn/` (`q_network.py`, `replay_buffer.py`, `agent.py`, `train_dqn.py`, `evaluate_dqn.py`, `render_dqn_policy.py`) implement the exact requirements of Assignment 3.
> * **Classroom Demos**: Scripts prefixed with `demo_` in `src/` are additional interactive demonstrations built for screen-sharing in class.

---

## 6. Step-by-Step Reproduction & Execution Guide

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/alicih4n/CSCN8020_Assignment3.git
cd CSCN8020_Assignment3

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fetch external Unitree assets & generate fixed-base XML scene
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
python src/create_fixed_base_g1.py
```

### Step 2: Run Self-Validation Smoke Test
```bash
PYTHONPATH=src python src/dqn/smoke_test.py
```

### Step 3: Train DQN Models (Headless Parameter Study)
```bash
PYTHONPATH=src python src/dqn/train_dqn.py
```
*Runs 600 episodes for both Config A ($\text{decay}=0.995$) and Config B ($\text{decay}=0.985$), saving checkpoints to `models/` and plots to `results/`.*

### Step 4: Evaluate Checkpoints & Compare with Rule-Based Baseline
```bash
PYTHONPATH=src python src/dqn/evaluate_dqn.py
```
*Evaluates models greedily over 20 benchmark episodes ($\epsilon=0.0$), prints metrics comparison tables, and saves the top model to `models/selected_dqn.pt`.*

### Step 5: Render 3D Policy in MuJoCo Viewer

* **macOS (Apple Silicon M1/M2/M3)**:
  *(Note: On macOS Cocoa main thread, use `mjpython` for passive GUI rendering)*
  ```bash
  # Official required step-by-step benchmark viewer:
  PYTHONPATH=src mjpython src/dqn/render_dqn_policy.py

  # Continuous back-and-forth swing demo:
  PYTHONPATH=src mjpython src/dqn/render_dqn_continuous.py

  # Interactive classroom presentation & wave demo:
  PYTHONPATH=src mjpython src/demo_class_presentation.py

  # Football kick & goal celebration demo:
  PYTHONPATH=src mjpython src/demo_football_kick.py

  # Ultra-smooth fluid Tai-Chi wave demo:
  PYTHONPATH=src mjpython src/demo_fluid_taichi.py

  # Pen grip & drawing demo:
  PYTHONPATH=src mjpython src/demo_pen_drawing.py
  ```

* **Linux / Windows 11 (WSL 2 with WSLg)**:
  ```bash
  # Official required step-by-step benchmark viewer:
  PYTHONPATH=src python src/dqn/render_dqn_policy.py

  # Continuous back-and-forth swing demo:
  PYTHONPATH=src python src/dqn/render_dqn_continuous.py

  # Interactive classroom presentation & wave demo:
  PYTHONPATH=src python src/demo_class_presentation.py

  # Football kick & goal celebration demo:
  PYTHONPATH=src python src/demo_football_kick.py

  # Ultra-smooth fluid Tai-Chi wave demo:
  PYTHONPATH=src python src/demo_fluid_taichi.py

  # Pen grip & drawing demo:
  PYTHONPATH=src python src/demo_pen_drawing.py
  ```
