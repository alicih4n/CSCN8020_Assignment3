from __future__ import annotations

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

SCENE_PATH = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"


def main() -> None:
    print("==================================================")
    print("  UNITREE G1 - CYBERPUNK MARTIAL ARTS & BOXING   ")
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
        print("\n[Phase 1: Raising Fists into Boxing Guard Stance]")
        time.sleep(1.0)

        # Stage 1: Raise both arms into tight boxing guard
        for i in range(100):
            t = i / 100.0
            data.qpos[l_pitch_q] = 0.50 * t
            data.qpos[l_roll_q] = 0.20 * t
            data.qpos[l_elbow_q] = -1.30 * t
            data.qpos[l_w_pitch_q] = 0.30 * t

            data.qpos[r_pitch_q] = 0.50 * t
            data.qpos[r_roll_q] = -0.20 * t
            data.qpos[r_elbow_q] = -1.30 * t
            data.qpos[r_w_pitch_q] = 0.30 * t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n>>> [Phase 2: EXECUTING HIGH-SPEED PUNCH COMBO] <<<")
        time.sleep(0.5)

        # Combo 1: Rapid Left Jab
        print("  -> Strike 1: Rapid Left Jab")
        for i in range(35):
            t = i / 35.0
            strike = math.sin(t * math.pi)
            data.qpos[l_pitch_q] = 0.50 + 0.30 * strike
            data.qpos[l_elbow_q] = -1.30 + 0.90 * strike  # Rapid elbow extension
            data.qpos[waist_yaw_q] = 0.10 * strike

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.008)

        # Combo 2: Power Right Cross (with waist rotation)
        print("  -> Strike 2: Power Right Cross")
        for i in range(40):
            t = i / 40.0
            strike = math.sin(t * math.pi)
            data.qpos[r_pitch_q] = 0.50 + 0.40 * strike
            data.qpos[r_elbow_q] = -1.30 + 1.00 * strike  # Full right extension
            data.qpos[waist_yaw_q] = -0.35 * strike      # Torso rotation into punch

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.008)

        # Combo 3: Left Upper Hook
        print("  -> Strike 3: Left Upper-Hook")
        for i in range(40):
            t = i / 40.0
            strike = math.sin(t * math.pi)
            data.qpos[l_pitch_q] = 0.50 + 0.60 * strike
            data.qpos[l_roll_q] = 0.20 + 0.35 * strike
            data.qpos[l_elbow_q] = -1.30 + 0.50 * strike
            data.qpos[waist_yaw_q] = 0.30 * strike

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.008)

        print("\n[Phase 3: Defensive Cross-Shield Block]")
        time.sleep(0.3)

        # Stage 3: Cross arms in front of head for defensive block
        for i in range(80):
            t = i / 80.0
            data.qpos[waist_yaw_q] = 0.0
            data.qpos[l_pitch_q] = 0.50 + (0.90 - 0.50) * t
            data.qpos[l_roll_q] = 0.20 + (0.45 - 0.20) * t
            data.qpos[l_elbow_q] = -1.30 + (-1.55 - (-1.30)) * t

            data.qpos[r_pitch_q] = 0.50 + (0.90 - 0.50) * t
            data.qpos[r_roll_q] = -0.20 + (-0.45 - (-0.20)) * t
            data.qpos[r_elbow_q] = -1.30 + (-1.55 - (-1.30)) * t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n[Phase 4: Wide Victory Pose & Audience Sweep]")
        time.sleep(0.5)

        # Stage 4: Open arms wide into T-pose / Victory Pose and sweep torso
        for i in range(250):
            t = i / 250.0
            sweep = math.sin(i * 0.06) * 0.25

            data.qpos[waist_yaw_q] = sweep
            data.qpos[l_pitch_q] = 0.10 + math.sin(i * 0.05) * 0.08
            data.qpos[l_roll_q] = 0.70                      # Arms wide open
            data.qpos[l_elbow_q] = -0.40

            data.qpos[r_pitch_q] = 0.10 + math.sin(i * 0.05) * 0.08
            data.qpos[r_roll_q] = -0.70                     # Arms wide open
            data.qpos[r_elbow_q] = -0.40

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.015)

        print("\n==================================================")
        print("     CYBERPUNK MARTIAL ARTS DEMO COMPLETED!       ")
        print("==================================================")
        print("Close the viewer window to finish.")

        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
