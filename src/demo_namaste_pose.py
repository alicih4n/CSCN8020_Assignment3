from __future__ import annotations

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

SCENE_PATH = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"


def main() -> None:
    print("==================================================")
    print("   UNITREE G1 - NAMASTE GREETING POSE DEMO        ")
    print("   Namaste-G1: Multi-Joint Gesture Demonstration  ")
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
    l_w_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_yaw_joint")]

    # Right Arm Joint Addresses
    r_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")]
    r_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint")]
    r_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_yaw_joint")]
    r_elbow_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow_joint")]
    r_w_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_roll_joint")]
    r_w_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_pitch_joint")]
    r_w_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_yaw_joint")]

    # Waist Joint Address
    waist_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")]

    # Initialize posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0, 0, 0.80])  # Anchored pelvis
    data.qpos[3:7] = np.array([1, 0, 0, 0])
    mujoco.mj_forward(model, data)

    # Target Namaste Pose Joint Angles
    # Left Arm
    l_pitch_target = 0.40
    l_roll_target = 0.35
    l_yaw_target = 0.25
    l_elbow_target = -1.45
    l_w_roll_target = 0.60
    l_w_pitch_target = 0.30

    # Right Arm (Mirrored)
    r_pitch_target = 0.40
    r_roll_target = -0.35
    r_yaw_target = -0.25
    r_elbow_target = -1.45
    r_w_roll_target = -0.60
    r_w_pitch_target = 0.30

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)

        # -------------------------------------------------------------
        # STAGE 1: IDLE POSITION
        # -------------------------------------------------------------
        print("\n[Stage 1: Neutral Standing Idle Position]")
        for i in range(60):
            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.015)

        # -------------------------------------------------------------
        # STAGE 2: ARMS RISING
        # -------------------------------------------------------------
        print("[Stage 2: Arms & Elbows Raising Smoothly]")
        for i in range(120):
            t = i / 120.0
            smooth_t = 0.5 * (1.0 - math.cos(t * math.pi))  # S-curve interpolation

            data.qpos[l_pitch_q] = l_pitch_target * smooth_t
            data.qpos[l_elbow_q] = -0.70 * smooth_t
            data.qpos[r_pitch_q] = r_pitch_target * smooth_t
            data.qpos[r_elbow_q] = -0.70 * smooth_t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.012)

        # -------------------------------------------------------------
        # STAGE 3: HANDS ALIGNING & PALMS TOUCHING
        # -------------------------------------------------------------
        print("[Stage 3: Hands Aligning & Palms Joining in Center]")
        for i in range(150):
            t = i / 150.0
            smooth_t = 0.5 * (1.0 - math.cos(t * math.pi))

            # Left Arm transition
            data.qpos[l_pitch_q] = l_pitch_target
            data.qpos[l_roll_q] = l_roll_target * smooth_t
            data.qpos[l_yaw_q] = l_yaw_target * smooth_t
            data.qpos[l_elbow_q] = -0.70 + (l_elbow_target - (-0.70)) * smooth_t
            data.qpos[l_w_roll_q] = l_w_roll_target * smooth_t
            data.qpos[l_w_pitch_q] = l_w_pitch_target * smooth_t

            # Right Arm transition (Mirrored)
            data.qpos[r_pitch_q] = r_pitch_target
            data.qpos[r_roll_q] = r_roll_target * smooth_t
            data.qpos[r_yaw_q] = r_yaw_target * smooth_t
            data.qpos[r_elbow_q] = -0.70 + (r_elbow_target - (-0.70)) * smooth_t
            data.qpos[r_w_roll_q] = r_w_roll_target * smooth_t
            data.qpos[r_w_pitch_q] = r_w_pitch_target * smooth_t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.012)

        # -------------------------------------------------------------
        # STAGE 4: NAMASTE POSE HOLD & GENTLE BOW
        # -------------------------------------------------------------
        print("\n>>> [Stage 4: NAMASTE POSE ACHIEVED - Holding Steady with Bow] <<<")
        for i in range(250):
            t = i * 0.04
            bow_sway = math.sin(t * 0.5) * 0.05  # Subtle breathing motion

            data.qpos[l_pitch_q] = l_pitch_target + bow_sway
            data.qpos[r_pitch_q] = r_pitch_target + bow_sway
            data.qpos[waist_yaw_q] = math.sin(t * 0.3) * 0.04  # Gentle center sway

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()

            if i % 60 == 0:
                print(f"Holding Namaste Pose | Step {i:3d}/250 | Left Elbow: {data.qpos[l_elbow_q]:.3f} rad | Right Elbow: {data.qpos[r_elbow_q]:.3f} rad")
            time.sleep(0.015)

        print("\n==================================================")
        print("      NAMASTE GREETING POSE DEMO COMPLETED!       ")
        print("==================================================")
        print("Close the viewer window to finish.")

        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
