from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

SCENE_BASE_PATH = Path("assets/g1_fixed_base/scene_29dof_fixed_base.xml")
SCENE_PEN_PATH = Path("assets/g1_fixed_base/scene_pen_drawing.xml")


def ensure_pen_drawing_scene() -> Path:
    """Create scene_pen_drawing.xml with table and pen placed in front of G1."""
    if not SCENE_BASE_PATH.is_file():
        raise FileNotFoundError(f"Base scene not found: {SCENE_BASE_PATH}")

    tree = ET.parse(SCENE_BASE_PATH)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise RuntimeError("Worldbody not found in base scene.")

    # Remove existing table/pen if present to avoid duplication
    for child in list(worldbody):
        if child.get("name") in {"table", "pen"}:
            worldbody.remove(child)

    # Add Wooden Table in front of G1
    table_body = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0.30 0.15 0.35"})
    ET.SubElement(table_body, "geom", {"type": "box", "size": "0.20 0.25 0.02", "rgba": "0.45 0.28 0.15 1", "mass": "10"})
    ET.SubElement(table_body, "geom", {"type": "cylinder", "size": "0.02 0.17", "pos": "0.15 0.20 -0.17", "rgba": "0.2 0.2 0.2 1"})
    ET.SubElement(table_body, "geom", {"type": "cylinder", "size": "0.02 0.17", "pos": "-0.15 0.20 -0.17", "rgba": "0.2 0.2 0.2 1"})
    ET.SubElement(table_body, "geom", {"type": "cylinder", "size": "0.02 0.17", "pos": "0.15 -0.20 -0.17", "rgba": "0.2 0.2 0.2 1"})
    ET.SubElement(table_body, "geom", {"type": "cylinder", "size": "0.02 0.17", "pos": "-0.15 -0.20 -0.17", "rgba": "0.2 0.2 0.2 1"})

    # Add Pen sitting on the table
    pen_body = ET.SubElement(worldbody, "body", {"name": "pen", "pos": "0.28 0.15 0.385"})
    ET.SubElement(pen_body, "freejoint", {"name": "pen_joint"})
    # Pen Barrel (Navy Blue)
    ET.SubElement(pen_body, "geom", {"name": "pen_body", "type": "cylinder", "size": "0.007 0.07", "rgba": "0.1 0.2 0.85 1", "mass": "0.03"})
    # Pen Nib/Tip (Bright Red)
    ET.SubElement(pen_body, "geom", {"name": "pen_tip", "type": "cylinder", "size": "0.003 0.02", "pos": "0 0 0.08", "rgba": "1 0.1 0.1 1", "mass": "0.005"})
    ET.SubElement(pen_body, "site", {"name": "pen_nib_site", "pos": "0 0 0.09", "size": "0.004", "rgba": "1 0 0 1"})

    tree.write(SCENE_PEN_PATH, encoding="utf-8", xml_declaration=True)
    return SCENE_PEN_PATH


def generate_number_6_trajectory(steps: int = 300) -> list[tuple[float, float]]:
    """Generate (y, z) coordinates for drawing the number '6'."""
    trajectory = []
    stem_steps = int(steps * 0.45)

    # Top arc swooping down-left
    for i in range(stem_steps):
        t = i / stem_steps
        angle = math.pi * 0.2 + t * math.pi * 0.8
        y = 0.15 + 0.06 * math.cos(angle)
        z = 0.66 + 0.08 * math.sin(angle) - t * 0.10
        trajectory.append((y, z))

    # Circular bottom loop
    loop_steps = steps - stem_steps
    center_y = 0.15
    center_z = 0.56
    radius = 0.05
    for i in range(loop_steps):
        t = i / loop_steps
        angle = math.pi * 0.5 - t * 2.0 * math.pi
        y = center_y + radius * math.cos(angle)
        z = center_z + radius * math.sin(angle)
        trajectory.append((y, z))

    return trajectory


def set_pen_in_hand(
    data: mujoco.MjData,
    hand_id: int,
    pen_qpos_adr: int,
    offset_in_hand: np.ndarray = np.array([0.04, 0.0, -0.01]),
) -> None:
    """Rigidly attach pen inside robot hand's palm/fingers."""
    hand_pos = data.xpos[hand_id]
    hand_mat = data.xmat[hand_id].reshape(3, 3)

    # Compute global 3D position of pen held in hand
    held_pen_pos = hand_pos + hand_mat @ offset_in_hand

    # Set pen position
    data.qpos[pen_qpos_adr : pen_qpos_adr + 3] = held_pen_pos

    # Quaternion matching hand orientation (pen tip pointing forward)
    data.qpos[pen_qpos_adr + 3 : pen_qpos_adr + 7] = np.array([0.707, 0.707, 0, 0])


def main() -> None:
    print("==================================================")
    print("      UNITREE G1 - PEN DRAWING DEMONSTRATION      ")
    print("==================================================")
    print("Preparing MuJoCo scene with table and pen...")

    scene_path = ensure_pen_drawing_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Joint addresses
    left_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")]
    left_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")]
    left_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_yaw_joint")]
    left_elbow_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint")]
    left_wrist_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_roll_joint")]
    left_wrist_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_pitch_joint")]

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    pen_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pen_joint")]

    # Initialize posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0, 0, 0.80])  # Anchored pelvis
    data.qpos[3:7] = np.array([1, 0, 0, 0])
    mujoco.mj_forward(model, data)

    trajectory = generate_number_6_trajectory(steps=300)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n[Phase 1: Reaching Down to Hand Level on Table]")
        time.sleep(1.0)

        # Stage 1: Lower hand directly over table & pen (120 steps)
        for i in range(120):
            t = i / 120.0
            data.qpos[left_pitch_q] = 0.45 * t
            data.qpos[left_roll_q] = 0.20 * t
            data.qpos[left_elbow_q] = -0.45 * t
            data.qpos[left_wrist_pitch_q] = 0.20 * t

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("[Phase 2: Enclosing Fingers around Pen & Lifting to Drawing Height]")
        time.sleep(0.5)

        # Stage 2: Grasp pen into palm and lift up to drawing position
        for i in range(120):
            t = i / 120.0
            data.qpos[left_pitch_q] = 0.45 + (0.15 - 0.45) * t
            data.qpos[left_roll_q] = 0.20 + (0.10 - 0.20) * t
            data.qpos[left_elbow_q] = -0.45 + (-0.75 - (-0.45)) * t
            data.qpos[left_yaw_q] = 0.25 * t
            data.qpos[left_wrist_roll_q] = 0.30 * t
            data.qpos[left_wrist_pitch_q] = 0.20 + (-0.30 - 0.20) * t

            # Clamp pen firmly inside left hand's palm
            set_pen_in_hand(data, hand_id, pen_qpos_adr)

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n>>> [Phase 3: Drawing Number '6' in the Air with Pen Tip] <<<")
        time.sleep(0.5)

        # Stage 3: Perform arm movements to draw "6" in the air
        for idx, (target_y, target_z) in enumerate(trajectory):
            # Smooth humanoid arm inverse-kinematics approximation
            data.qpos[left_roll_q] = (target_y - 0.15) * 2.5 + 0.10
            data.qpos[left_pitch_q] = 0.15 + (0.65 - target_z) * 1.8
            data.qpos[left_elbow_q] = -0.75 + (target_z - 0.55) * 0.9
            data.qpos[left_wrist_pitch_q] = -0.30 + (target_z - 0.55) * 0.5

            # Pen remains rigidly locked in hand's grip
            set_pen_in_hand(data, hand_id, pen_qpos_adr)

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()

            if idx % 50 == 0:
                print(f"Drawing Progress: Step {idx:3d}/{len(trajectory)} | Pen Tip Y: {target_y:+.3f}, Z: {target_z:+.3f}")
            time.sleep(0.015)

        print("\n[Phase 4: Returning Pen to Table Surface]")
        time.sleep(0.5)

        # Stage 4: Lower pen back onto the table surface and release
        for i in range(120):
            t = i / 120.0
            data.qpos[left_pitch_q] = 0.15 + (0.45 - 0.15) * t
            data.qpos[left_roll_q] = 0.10 * (1 - t)
            data.qpos[left_elbow_q] = -0.75 + (-0.25 - (-0.75)) * t
            data.qpos[left_wrist_pitch_q] = -0.30 * (1 - t)

            # Lower pen onto table
            if t < 0.8:
                set_pen_in_hand(data, hand_id, pen_qpos_adr)
            else:
                # Release pen onto table surface
                data.qpos[pen_qpos_adr : pen_qpos_adr + 3] = np.array([0.28, 0.15, 0.385])
                data.qpos[pen_qpos_adr + 3 : pen_qpos_adr + 7] = np.array([1, 0, 0, 0])

            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n==================================================")
        print("          PEN DRAWING DEMO COMPLETED!            ")
        print("==================================================")
        print("Close the viewer window to finish.")

        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
