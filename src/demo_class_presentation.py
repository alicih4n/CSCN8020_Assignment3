from __future__ import annotations

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

SCENE_PATH = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"


def main() -> None:
    print("\n==================================================")
    print("   UNITREE G1 - CLASSROOM PRESENTATION DEMO       ")
    print("   CSCN8020 Reinforcement Learning | Ali Cihan    ")
    print("==================================================")
    print("Loading Unitree G1 29-DOF fixed-base model...")

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Left Arm Joint Addresses
    l_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")]
    l_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")]
    l_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_yaw_joint")]
    l_elbow_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint")]
    l_w_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_roll_joint")]
    l_w_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_pitch_joint")]

    # Right Arm Joint Addresses
    r_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")]
    r_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint")]
    r_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_yaw_joint")]
    r_elbow_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow_joint")]
    r_w_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_roll_joint")]
    r_w_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_pitch_joint")]

    # Waist Joint Address
    waist_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")]

    # Initialize posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0, 0, 0.80])  # Anchored pelvis
    data.qpos[3:7] = np.array([1, 0, 0, 0])
    mujoco.mj_forward(model, data)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        
        # -------------------------------------------------------------
        # PHASE 1: FRIENDLY WAVE TO THE CLASS
        # -------------------------------------------------------------
        print("\n🤖 [ROBOT]: 'Hello Class & Professor Espinosa! Welcome to the Demo!'")
        
        # Raise Right Arm into High Wave Pose
        for i in range(100):
            t = i / 100.0
            data.qpos[r_pitch_q] = 0.80 * t
            data.qpos[r_roll_q] = -0.50 * t
            data.qpos[r_elbow_q] = -0.80 * t
            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        # Wave hand side-to-side
        for i in range(160):
            wave = math.sin(i * 0.15) * 0.40
            data.qpos[r_w_roll_q] = wave
            data.qpos[waist_yaw_q] = wave * 0.20
            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.015)

        # -------------------------------------------------------------
        # PHASE 2: POINTING TO THE PRESENTATION SCREEN
        # -------------------------------------------------------------
        print("\n🤖 [ROBOT]: 'Pointing to our Assignment 3 Results on screen...'")
        time.sleep(0.5)

        # Lower right arm, extend left arm to point left
        for i in range(120):
            t = i / 120.0
            # Lower Right Arm
            data.qpos[r_pitch_q] = 0.80 * (1 - t)
            data.qpos[r_roll_q] = -0.50 * (1 - t)
            data.qpos[r_elbow_q] = -0.80 * (1 - t)
            data.qpos[r_w_roll_q] = 0.0

            # Extend Left Arm to Point
            data.qpos[l_pitch_q] = 0.60 * t
            data.qpos[l_roll_q] = 0.65 * t
            data.qpos[l_elbow_q] = -0.15 * t
            data.qpos[waist_yaw_q] = 0.35 * t  # Turn body toward slides

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        time.sleep(1.2)

        # -------------------------------------------------------------
        # PHASE 3: DOUBLE THUMBS-UP APPROVAL
        # -------------------------------------------------------------
        print("\n🤖 [ROBOT]: '100% Success Rate Benchmark Verified! Double Thumbs-Up!'")
        time.sleep(0.5)

        # Bring both arms to chest height for Double Thumbs-Up
        for i in range(120):
            t = i / 120.0
            data.qpos[waist_yaw_q] = 0.35 * (1 - t)  # Center body

            # Left Arm Thumbs-Up
            data.qpos[l_pitch_q] = 0.60 + (0.40 - 0.60) * t
            data.qpos[l_roll_q] = 0.65 + (0.15 - 0.65) * t
            data.qpos[l_elbow_q] = -0.15 + (-1.20 - (-0.15)) * t
            data.qpos[l_w_pitch_q] = 0.50 * t

            # Right Arm Thumbs-Up
            data.qpos[r_pitch_q] = 0.40 * t
            data.qpos[r_roll_q] = -0.15 * t
            data.qpos[r_elbow_q] = -1.20 * t
            data.qpos[r_w_pitch_q] = 0.50 * t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        time.sleep(1.2)

        # -------------------------------------------------------------
        # PHASE 4: DYNAMIC CELEBRATION WAVE
        # -------------------------------------------------------------
        print("\n🤖 [ROBOT]: 'Grand Celebration Wave for the Class!'")
        
        # Raise both arms overhead and sway
        for i in range(200):
            t = i * 0.05
            sway = math.sin(t) * 0.35

            data.qpos[waist_yaw_q] = sway * 0.5
            data.qpos[l_pitch_q] = 0.80 + sway * 0.2
            data.qpos[l_roll_q] = 0.50 + sway * 0.3
            data.qpos[l_elbow_q] = -0.50

            data.qpos[r_pitch_q] = 0.80 - sway * 0.2
            data.qpos[r_roll_q] = -0.50 + sway * 0.3
            data.qpos[r_elbow_q] = -0.50

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.015)

        # -------------------------------------------------------------
        # PHASE 5: RESPECTFUL BOW TO PROFESSOR & CLASS
        # -------------------------------------------------------------
        print("\n🤖 [ROBOT]: 'Thank you Professor Espinosa & Class!' (Bow)")
        
        # Lower arms and bow torso
        for i in range(120):
            t = i / 120.0
            data.qpos[waist_yaw_q] = 0.0

            # Right arm across chest for bow
            data.qpos[r_pitch_q] = 0.80 + (0.30 - 0.80) * t
            data.qpos[r_roll_q] = -0.50 * (1 - t)
            data.qpos[r_elbow_q] = -0.50 + (-1.40 - (-0.50)) * t

            # Left arm down at side
            data.qpos[l_pitch_q] = 0.80 * (1 - t)
            data.qpos[l_roll_q] = 0.50 * (1 - t)
            data.qpos[l_elbow_q] = -0.50 * (1 - t)

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n==================================================")
        print("    CLASSROOM PRESENTATION DEMO COMPLETED!        ")
        print("==================================================")
        print("Close the viewer window to finish.")

        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
