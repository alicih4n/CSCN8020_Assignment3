from __future__ import annotations

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

SCENE_PATH = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"


def main() -> None:
    print("==================================================")
    print("     UNITREE G1 - FLUID TAI-CHI & HARMONIC FLOW   ")
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
        print("\n[Phase 1: Opening Breath & Fluid Arm Float]")
        time.sleep(1.0)

        # Phase 1: Smooth opening float (150 steps)
        for i in range(150):
            t = i / 150.0
            smooth_t = 0.5 * (1.0 - math.cos(t * math.pi))  # S-curve interpolation

            data.qpos[l_pitch_q] = 0.35 * smooth_t
            data.qpos[l_roll_q] = 0.20 * smooth_t
            data.qpos[l_elbow_q] = -0.50 * smooth_t

            data.qpos[r_pitch_q] = 0.35 * smooth_t
            data.qpos[r_roll_q] = -0.20 * smooth_t
            data.qpos[r_elbow_q] = -0.50 * smooth_t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.012)

        print("\n>>> [Phase 2: Tai-Chi Flowing Water Wave Routine] <<<")
        time.sleep(0.5)

        # Phase 2: Silky harmonic Tai-Chi waves (350 steps)
        for i in range(350):
            t = i * 0.03
            
            # Harmonic wave components
            sin_1 = math.sin(t)
            cos_1 = math.cos(t)
            sin_half = math.sin(t * 0.5)

            # Torso sway
            data.qpos[waist_yaw_q] = 0.20 * sin_half

            # Left Arm Flow
            data.qpos[l_pitch_q] = 0.35 + 0.25 * sin_1
            data.qpos[l_roll_q] = 0.25 + 0.15 * cos_1
            data.qpos[l_elbow_q] = -0.60 - 0.25 * sin_1
            data.qpos[l_w_pitch_q] = 0.20 * cos_1
            data.qpos[l_w_roll_q] = 0.30 * sin_1

            # Right Arm Flow (phase shifted for fluid elegance)
            data.qpos[r_pitch_q] = 0.35 - 0.25 * sin_1
            data.qpos[r_roll_q] = -0.25 - 0.15 * cos_1
            data.qpos[r_elbow_q] = -0.60 + 0.25 * sin_1
            data.qpos[r_w_pitch_q] = -0.20 * cos_1
            data.qpos[r_w_roll_q] = -0.30 * sin_1

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()

            if i % 70 == 0:
                print(f"Fluid Wave Progress: Step {i:3d}/350 | Torso Sway: {data.qpos[waist_yaw_q]:+.3f} rad")
            time.sleep(0.015)

        print("\n[Phase 3: Majestic Orchestral Expansion]")
        time.sleep(0.5)

        # Phase 3: Wide majestic expansion (200 steps)
        for i in range(200):
            t = i * 0.04
            expand = math.sin(t * 0.5)

            data.qpos[waist_yaw_q] = 0.10 * math.sin(t * 0.3)
            data.qpos[l_pitch_q] = 0.20 + 0.15 * expand
            data.qpos[l_roll_q] = 0.40 + 0.30 * expand
            data.qpos[l_elbow_q] = -0.40 - 0.20 * expand
            data.qpos[l_w_roll_q] = 0.40 * expand

            data.qpos[r_pitch_q] = 0.20 + 0.15 * expand
            data.qpos[r_roll_q] = -0.40 - 0.30 * expand
            data.qpos[r_elbow_q] = -0.40 - 0.20 * expand
            data.qpos[r_w_roll_q] = -0.40 * expand

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.015)

        print("\n[Phase 4: Return to Harmonious Balance]")
        time.sleep(0.5)

        # Phase 4: Gentle return to rest (150 steps)
        for i in range(150):
            t = i / 150.0
            smooth_t = 1.0 - (0.5 * (1.0 - math.cos(t * math.pi)))

            data.qpos[waist_yaw_q] = data.qpos[waist_yaw_q] * smooth_t
            data.qpos[l_pitch_q] = 0.20 * smooth_t
            data.qpos[l_roll_q] = 0.40 * smooth_t
            data.qpos[l_elbow_q] = -0.40 * smooth_t

            data.qpos[r_pitch_q] = 0.20 * smooth_t
            data.qpos[r_roll_q] = -0.40 * smooth_t
            data.qpos[r_elbow_q] = -0.40 * smooth_t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.012)

        print("\n==================================================")
        print("      FLUID TAI-CHI DEMO COMPLETED SUCCESSFULLY!  ")
        print("==================================================")
        print("Close the viewer window to finish.")

        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
