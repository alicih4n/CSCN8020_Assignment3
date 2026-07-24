from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

SCENE_BASE_PATH = Path("assets/g1_fixed_base/scene_29dof_fixed_base.xml")
SCENE_BALL_PATH = Path("assets/g1_fixed_base/scene_football_kick.xml")


def ensure_football_scene() -> Path:
    """Create scene_football_kick.xml with floor and a soccer ball in front of G1's right leg."""
    if not SCENE_BASE_PATH.is_file():
        raise FileNotFoundError(f"Base scene not found: {SCENE_BASE_PATH}")

    tree = ET.parse(SCENE_BASE_PATH)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError("Worldbody not found in base scene.")

    # Remove existing ball if present to avoid duplication
    for child in list(worldbody):
        if child.get("name") == "soccer_ball":
            worldbody.remove(child)

    # Add Soccer Ball sitting in front of right leg
    ball_body = ET.SubElement(worldbody, "body", {"name": "soccer_ball", "pos": "0.32 -0.08 0.12"})
    ET.SubElement(ball_body, "freejoint", {"name": "ball_joint"})
    # Soccer Ball (Bright White with Dark Patterns)
    ET.SubElement(ball_body, "geom", {
        "name": "ball_geom",
        "type": "sphere",
        "size": "0.11",
        "rgba": "0.95 0.95 0.95 1",
        "mass": "0.45",
        "friction": "0.8 0.005 0.0001",
    })

    tree.write(SCENE_BALL_PATH, encoding="utf-8", xml_declaration=True)
    return SCENE_BALL_PATH


def main() -> None:
    print("==================================================")
    print("       UNITREE G1 - FOOTBALL KICK DEMONSTRATION   ")
    print("==================================================")
    print("Preparing MuJoCo scene with soccer ball...")

    scene_path = ensure_football_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Right Leg Joint Addresses
    r_hip_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch_joint")]
    r_knee_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint")]
    r_ankle_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_pitch_joint")]

    # Upper Body Joint Addresses for Arm Balance & Celebration
    l_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")]
    l_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")]
    r_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")]
    r_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_roll_joint")]
    waist_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")]

    ball_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")]

    # Initialize posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0, 0, 0.80])  # Anchored pelvis
    data.qpos[3:7] = np.array([1, 0, 0, 0])
    mujoco.mj_forward(model, data)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)

        # -------------------------------------------------------------
        # PHASE 1: LEG WIND-UP & ARM BALANCE
        # -------------------------------------------------------------
        print("\n⚽ [Phase 1: Shifting Weight & Pulling Right Leg Back for Kick]")
        
        for i in range(100):
            t = i / 100.0
            # Pull right leg back
            data.qpos[r_hip_pitch_q] = -0.45 * t
            data.qpos[r_knee_q] = 0.65 * t
            data.qpos[r_ankle_pitch_q] = -0.20 * t

            # Balance upper body with arms
            data.qpos[l_roll_q] = 0.35 * t
            data.qpos[r_roll_q] = -0.35 * t
            data.qpos[waist_yaw_q] = 0.15 * t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        time.sleep(0.3)

        # -------------------------------------------------------------
        # PHASE 2: HIGH-SPEED KICK STRIKE & BALL LAUNCH
        # -------------------------------------------------------------
        print("\n💥 >>> [Phase 2: EXECUTING POWER FOOTBALL KICK!] <<< 💥")
        
        for i in range(40):
            t = i / 40.0
            strike = math.sin(t * math.pi * 0.5)

            # Snap right leg forward violently
            data.qpos[r_hip_pitch_q] = -0.45 + (0.75 - (-0.45)) * strike
            data.qpos[r_knee_q] = 0.65 + (-0.10 - 0.65) * strike
            data.qpos[r_ankle_pitch_q] = -0.20 + (0.35 - (-0.20)) * strike

            # Launch ball into ballistic flight path upon impact
            if t > 0.4:
                ball_t = (t - 0.4) / 0.6
                ball_x = 0.32 + 2.5 * ball_t
                ball_y = -0.08 + 0.15 * ball_t
                ball_z = 0.12 + 1.20 * math.sin(ball_t * math.pi * 0.7) - 0.30 * (ball_t ** 2)
                data.qpos[ball_qpos_adr : ball_qpos_adr + 3] = np.array([ball_x, ball_y, max(0.11, ball_z)])

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.008)

        # -------------------------------------------------------------
        # PHASE 3: BALL FLIGHT & LEG FOLLOW-THROUGH
        # -------------------------------------------------------------
        print("\n🚀 [Phase 3: Ball Flight & Follow-Through]")
        
        for i in range(80):
            t = i / 80.0
            ball_t = 1.0 + (i / 80.0) * 1.5
            ball_x = 0.32 + 2.5 * (1.0 + t)
            ball_y = -0.08 + 0.15 * (1.0 + t)
            ball_z = max(0.11, 0.90 - 0.80 * (t ** 1.5))

            data.qpos[ball_qpos_adr : ball_qpos_adr + 3] = np.array([ball_x, ball_y, ball_z])

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        # -------------------------------------------------------------
        # PHASE 4: GOAL CELEBRATION (ARMS OVERHEAD)
        # -------------------------------------------------------------
        print("\n🎉 [Phase 4: GOAL CELEBRATION! Arms Overhead!]")
        
        for i in range(150):
            t = i / 150.0
            smooth_t = 0.5 * (1.0 - math.cos(t * math.pi))
            cheer_sway = math.sin(i * 0.1) * 0.15

            # Retract kicking leg to stance
            data.qpos[r_hip_pitch_q] = 0.75 * (1 - smooth_t)
            data.qpos[r_knee_q] = 0.0

            # Raise arms overhead in goal celebration
            data.qpos[l_pitch_q] = 1.20 * smooth_t + cheer_sway
            data.qpos[l_roll_q] = 0.40 * smooth_t
            data.qpos[r_pitch_q] = 1.20 * smooth_t - cheer_sway
            data.qpos[r_roll_q] = -0.40 * smooth_t
            data.qpos[waist_yaw_q] = cheer_sway

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.012)

        print("\n==================================================")
        print("      G1 FOOTBALL KICK & GOAL DEMO COMPLETED!     ")
        print("==================================================")
        print("Close the viewer window to finish.")

        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
