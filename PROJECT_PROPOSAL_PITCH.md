# CSCN8020 Team Project Proposal & Pitch Guide (5-Minute Presentation)

**Course:** CSCN8020 - Reinforcement Learning  
**Presenter:** Ali Cihan Ozdemir & Team  
**Instructor:** Prof. Enrique Espinosa  
**Institution:** Conestoga College  

---

## ⏱️ 5-Minute Pitch Outline & Time Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 0:00 - 1:30 | PART 1: Assignment 3 Summary (DQN Left Elbow Control)         │
│ 1:30 - 3:00 | PART 2: Mathematical & Algorithmic Foundations (MDP, Bellman) │
│ 3:00 - 4:30 | PART 3: Final Project Proposal (DQN -> Actor-Critic / PPO/SAC) │
│ 4:30 - 5:00 | PART 4: Summary & Q&A                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎤 Part 1: Assignment 3 Overview (1.5 Minutes)

### Speaker Notes / Script:
> *"Hello Professor Espinosa and class. Today, our team is presenting our results for Assignment 3—Deep Q-Network control of the Unitree G1 left elbow—and pitching our Final Project Proposal.*
> 
> *For A3, we built the entire PyTorch DQN pipeline from scratch without relying on turnkey libraries like Stable-Baselines3. We modeled the G1 left elbow as a Gymnasium environment with a 4-dimensional continuous state vector $[\theta, \dot{\theta}, g, g - \theta]$ and a 3-action discrete control space.*
> 
> *Our trained DQN agent achieved a **100% success rate (20/20 episodes)** across four target angles, reaching the goal in **19.8 steps** with an accuracy error of just **0.0039 radians**."*

---

## 📐 Part 2: Mathematical & Algorithmic Foundations (1.5 Minutes)

### Whiteboard Topics Covered:
1. **Markov Decision Process (MDP)**:
   - Current state $s_t$ depends on previous state $s_{t-1}$ and action $a_{t-1}$.
   - Agent maximizes cumulative discounted reward: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$.
2. **Action-Value Approximation ($Q^*$)**:
   - Optimal Bellman target: $y_i = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \theta^-)$.
   - Target network updated every 250 steps to prevent moving-target divergence.
3. **Exploration Study**:
   - Parameter decay study ($\text{decay}=0.995$ vs $0.985$) showed that gradual exploration prevents premature convergence.

---

## 🚀 Part 3: Final Project Proposal (1.5 Minutes)

### Proposal Core Concept: **From Discrete DQN to Continuous Actor-Critic Control on Unitree G1**

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Assignment 3 (DQN)                    │ Final Project Proposal (Actor-Critic) │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Discrete Actions (3 choices: -0.08,0,+0.08)│ Continuous Torque Control ($a_t \in \mathbb{R}^n$)│
│ Single Joint (Left Elbow)             │ Multi-Joint Bimanual / Leg Control    │
│ Value-Based ($Q(s, a)$ estimation)    │ Policy Gradient ($\pi_{\theta}(a|s)$) + Value $V_{\phi}(s)$ │
│ Fixed Target Goal Angles              │ Dynamic Trajectory Tracking & Locomotion│
└───────────────────────────────────────┴───────────────────────────────────────┘
```

### Key Technical Improvements Planned:
1. **DQN vs. Actor-Critic (Robotics Context)**:
   - Discrete action DQNs are great for games, but real humanoid robots operate with continuous joint torques.
   - We will implement an **Actor-Critic architecture (PPO / SAC)** where the **Actor** outputs a continuous action probability distribution $\pi_{\theta}(a|s) \sim \mathcal{N}(\mu(s), \sigma(s))$, and the **Critic** approximates state values $V_{\phi}(s)$.
2. **Multi-Joint Coordination**:
   - Expanding from single elbow control to **multi-joint bimanual arm control** and **whole-body balance**.
3. **Advanced Reward Design ($R$)**:
   - Multi-objective reward penalizing joint acceleration, energy consumption, and tracking error.

---

## 🏁 Part 4: Conclusion & Wrap-Up (0.5 Minutes)

### Speaker Notes / Script:
> *"In summary, our A3 DQN foundation is fully verified with 100% benchmark success. For our final project, we will bridge the gap to continuous robotic control using Actor-Critic policy gradients on the Unitree G1 model.*
> 
> *Thank you, and we welcome any questions!"*
