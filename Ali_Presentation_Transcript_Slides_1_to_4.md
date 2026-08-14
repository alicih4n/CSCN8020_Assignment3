# CSCN8020 Presentation Transcript: Slides 1 to 4

**Presenter:** Ali Cihan Ozdemir  
**Course:** CSCN8020 - Reinforcement Learning  
**Project:** Namaste-G1: Teaching a Humanoid Robot to Greet  
**Team:** Group 4 (Ali Cihan Ozdemir, Lohith Reddy Danda, Sumanth Reddy Konannagari, Muthuraj Jayakumar)  
**Target Duration per Slide:** 50 Seconds (~110 words per slide)  
**Total Duration for Slides 1–4:** 3 Minutes 20 Seconds  

---

## 📽️ SLIDE 1: Title & Project Overview
* **Target Time:** 50 Seconds
* **Word Count:** 110 words
* **On-Screen Action:** Point to the title *Namaste-G1* and your team members' names.

### Spoken Transcript:
> "Good morning Professor Espinosa and classmates. I'm Ali Cihan Ozdemir, and together with my team members Lohith, Sumanth, and Muthuraj, we represent Group 4.
> 
> Today, we're taking reinforcement learning beyond basic single-joint control to tackle a complex, high-dimensional robotics challenge: **Namaste-G1: Teaching a Humanoid Robot to Greet**.
> 
> Our main goal is to train a simulated 14-degree-of-freedom Unitree G1 humanoid robot to execute a full two-arm Namaste greeting pose—reaching the pose, holding it steadily within strict joint tolerances, and returning gracefully to rest.
> 
> Here is how we designed and evaluated multi-joint humanoid control here at Conestoga College."

---

## 📽️ SLIDE 2: Today’s Plan (Agenda)
* **Target Time:** 50 Seconds
* **Word Count:** 110 words
* **On-Screen Action:** Point to the roadmap items listed on the slide.

### Spoken Transcript:
> "Here is the roadmap for our presentation today.
> 
> We will begin by reviewing our initial 14-joint Dueling DQN baseline, and explain why we transitioned from discrete step control to continuous Actor-Critic.
> 
> Next, we will reveal our major headline finding and the critical PyTorch gradient bug we discovered along the way.
> 
> We will then analyze what happened after fixing the bug—where the robot demonstrated real learning movement but experienced policy instability—followed by explaining the underlying mathematical principles in plain terms, running a live simulation demonstration, and detailing our next steps for the project."

---

## 📽️ SLIDE 3: Before Actor-Critic: The DQN Baseline
* **Target Time:** 50 Seconds
* **Word Count:** 110 words
* **On-Screen Action:** Point to the 14-joint Dueling Q-Network stats (3.1 to 5.2 joints, peak 7/14).

### Spoken Transcript:
> "To establish our baseline, we first designed a 14-joint branching Dueling Q-Network. Each joint had three discrete actions: hold the current position, decrease the angle, or increase the angle.
> 
> We ran a fully logged 200-episode diagnostic experiment to evaluate its learning capability. While the network successfully learned to increase the average number of joints in tolerance from 3.1 up to 5.2—reaching a peak moment where 7 out of 14 joints met the criteria—it achieved zero full 14-joint successes across the entire run.
> 
> This proved that coarse, stepped discrete control made fine multi-joint coordination extremely difficult."

---

## 📽️ SLIDE 4: Why We Moved to Actor-Critic
* **Target Time:** 50 Seconds
* **Word Count:** 110 words
* **On-Screen Action:** Point to the side-by-side box (*DQN Stepped* vs *Actor-Critic Smooth*).

### Spoken Transcript:
> "That key limitation directly motivated our shift to Actor-Critic.
> 
> While DQN could only move each joint in fixed, stepped increments, Actor-Critic outputs a continuous action value for every joint simultaneously. Achieving finer, smoother control is essential when coordinating 14 joints at the exact same time.
> 
> However, moving to continuous control significantly expanded the search space, making the task harder rather than easier.
> 
> Consequently, our refined goal expanded into three distinct phases: reaching the Namaste pose, holding it steadily for a consecutive streak, and returning smoothly back to a resting posture."
