from __future__ import annotations

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

SCENE_PATH = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"


def main() -> None:
    print("==================================================")
    print("   UNITREE G1 - DUAL-ARM SALUTE & TRACKING DEMO   ")
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

    # Waist / Torso Joint Address
    waist_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")]

    # Initialize posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0, 0, 0.80])  # Anchored pelvis
    data.qpos[3:7] = np.array([1, 0, 0, 0])
    mujoco.mj_forward(model, data)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n[Phase 1: Humanoid System Initialization & Ready Stance]")
        time.sleep(1.0)

        # Phase 1: Dual-arm smooth rise into ready stance (120 steps)
        for i in range(120):
            t = i / 120.0
            # Left Arm
            data.qpos[l_pitch_q] = 0.20 * t
            data.qpos[l_roll_q] = 0.15 * t
            data.qpos[l_elbow_q] = -0.30 * t
            # Right Arm
            data.qpos[r_pitch_q] = 0.20 * t
            data.qpos[r_roll_q] = -0.15 * t
            data.qpos[r_elbow_q] = -0.30 * t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("[Phase 2: Precision Futuristic Robot Salute]")
        time.sleep(0.5)

        # Phase 2: Left arm moves into a crisp salute position (150 steps)
        for i in range(150):
            t = i / 150.0
            # Left arm crisp salute pose
            data.qpos[l_pitch_q] = 0.20 + (1.20 - 0.20) * t
            data.qpos[l_roll_q] = 0.15 + (0.45 - 0.15) * t
            data.qpos[l_elbow_q] = -0.30 + (-1.40 - (-0.30)) * t
            data.qpos[l_w_pitch_q] = 0.35 * t
            data.qpos[l_w_roll_q] = 0.50 * t

            # Right arm balances posture
            data.qpos[r_pitch_q] = 0.20 + (0.10 - 0.20) * t
            data.qpos[r_roll_q] = -0.15 + (-0.30 - (-0.15)) * t
            data.qpos[r_elbow_q] = -0.30 + (-0.60 - (-0.30)) * t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n>>> [Phase 3: Bimanual Synchronized Wave & Multi-Joint Tracking] <<<")
        time.sleep(0.5)

        # Phase 3: Dynamic bimanual wave and smooth tracking (300 steps)
        for i in range(300):
            t = i / 300.0
            wave_sine = math.sin(i * 0.1) * 0.25
            waist_sine = math.sin(i * 0.05) * 0.15

            # Waist rotation tracking
            data.qpos[waist_yaw_q] = waist_sine

            # Left arm holding salute with subtle breathing motion
            data.qpos[l_pitch_q] = 1.20 + wave_sine * 0.1
            data.qpos[l_elbow_q] = -1.40 + wave_sine * 0.08

            # Right arm dynamic wave motion (sweeping overhead)
            data.qpos[r_pitch_q] = 0.10 + wave_sine * 0.3
            data.qpos[r_roll_q] = -0.30 - (0.50 + wave_sine * 0.4)
            data.qpos[r_elbow_q] = -0.60 + wave_sine * 0.2
            data.qpos[r_w_roll_q] = wave_sine * 0.8

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()

            if i % 60 == 0:
                print(f"Tracking Progress: Step {i:3d}/300 | Waist Yaw: {waist_sine:+.3f} rad | Wave Amplitude: {wave_sine:+.3f} rad")
            time.sleep(0.015)

        print("\n[Phase 4: Respectful Humanoid Bow & Return to Rest]")
        time.sleep(0.5)

        # Phase 4: Respectful return to neutral resting pose (150 steps)
        for i in range(150):
            t = i / 150.0
            # Return waist
            data.qpos[waist_yaw_q] = data.qpos[waist_yaw_q] * (1 - t)

            # Lower Left Arm
            data.qpos[l_pitch_q] = 1.20 * (1 - t)
            data.qpos[l_roll_q] = 0.45 * (1 - t)
            data.qpos[l_elbow_q] = -1.40 * (1 - t)
            data.qpos[l_w_pitch_q] = 0.35 * (1 - t)
            data.qpos[l_w_roll_q] = 0.50 * (1 - t)

            # Lower Right Arm
            data.qpos[r_pitch_q] = 0.10 * (1 - t)
            data.qpos[r_roll_q] = -0.30 * (1 - t)
            data.qpos[r_elbow_q] = -0.60 * (1 - t)
            data.qpos[r_w_roll_q] = 0.0

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n==================================================")
        print("    HUMANOID DUAL-ARM SALUTE DEMO COMPLETED!      ")
        print("==================================================")
        print("Close the viewer window to finish.")

        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
