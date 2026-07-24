from __future__ import annotations

import math
import time
import numpy as np
import mujoco
import mujoco.viewer

# MuJoCo XML with a table and pen in front of the fixed-base G1 robot
SCENE_WITH_PEN_XML = """
<mujoco model="g1_pen_drawing_scene">
  <include file="assets/g1_fixed_base/scene_29dof_fixed_base.xml"/>
  <worldbody>
    <!-- Table in front of G1 robot -->
    <body name="table" pos="0.35 0 0.35">
      <geom type="box" size="0.2 0.35 0.02" rgba="0.4 0.25 0.15 1" mass="10"/>
      <geom type="cylinder" size="0.02 0.17" pos="0.15 0.3 -0.17" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.02 0.17" pos="-0.15 0.3 -0.17" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.02 0.17" pos="0.15 -0.3 -0.17" rgba="0.2 0.2 0.2 1"/>
      <geom type="cylinder" size="0.02 0.17" pos="-0.15 -0.3 -0.17" rgba="0.2 0.2 0.2 1"/>
    </body>

    # Pen on the table
    <body name="pen" pos="0.32 0.12 0.395">
      <freejoint name="pen_joint"/>
      <!-- Body of the pen (Navy Blue) -->
      <geom name="pen_body" type="cylinder" size="0.008 0.08" rgba="0.1 0.2 0.8 1" mass="0.05"/>
      <!-- Tip/Nib of the pen (Bright Red) -->
      <geom name="pen_tip" type="cone" size="0.008 0.02" pos="0 0 0.09" rgba="1 0.1 0.1 1" mass="0.01"/>
      <site name="pen_nib_site" pos="0 0 0.10" size="0.005" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def generate_number_6_trajectory(steps: int = 250) -> list[tuple[float, float]]:
    """
    Generate (y, z) coordinates for drawing the number "6".
    The number '6' starts at top right, curves left and down, then forms a bottom loop.
    """
    trajectory = []
    
    # 1. Top curve down to bottom loop (approx 40% of steps)
    stem_steps = int(steps * 0.45)
    for i in range(stem_steps):
        t = i / stem_steps
        # Top right arc swooping down-left
        angle = math.pi * 0.2 + t * math.pi * 0.8
        y = 0.06 * math.cos(angle) + 0.02
        z = 0.68 + 0.08 * math.sin(angle) - t * 0.10
        trajectory.append((y, z))
        
    # 2. Circular bottom loop (approx 55% of steps)
    loop_steps = steps - stem_steps
    center_y = 0.00
    center_z = 0.58
    radius = 0.05
    for i in range(loop_steps):
        t = i / loop_steps
        angle = math.pi * 0.5 - t * 2.0 * math.pi  # Full clockwise circle
        y = center_y + radius * math.cos(angle)
        z = center_z + radius * math.sin(angle)
        trajectory.append((y, z))
        
    return trajectory


def main() -> None:
    print("==================================================")
    print("      UNITREE G1 - PEN DRAWING DEMONSTRATION      ")
    print("==================================================")
    print("Initializing MuJoCo scene with table and pen...")

    # Load XML scene directly
    model = mujoco.MjModel.from_xml_string(SCENE_WITH_PEN_XML)
    data = mujoco.MjData(model)

    # Get joint and actuator IDs
    left_pitch_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch")
    left_roll_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_roll")
    left_yaw_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_yaw")
    left_elbow_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow")

    # Joint qpos addresses
    left_pitch_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")]
    left_roll_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")]
    left_yaw_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_yaw_joint")]
    left_elbow_q = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint")]

    pen_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
    pen_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pen_joint")]

    # Initialize model forward pass
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0, 0, 0.80])  # Fix pelvis
    data.qpos[3:7] = np.array([1, 0, 0, 0])
    mujoco.mj_forward(model, data)

    trajectory = generate_number_6_trajectory(steps=300)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n[Phase 1: Reaching Down to Pick Up Pen]")
        time.sleep(1.0)

        # Stage 1: Lower arm to table (100 steps)
        for i in range(120):
            t = i / 120.0
            data.qpos[left_pitch_q] = 0.35 * t
            data.qpos[left_roll_q] = 0.25 * t
            data.qpos[left_elbow_q] = -0.40 * t
            
            # Keep gravity bias compensated
            data.ctrl[:] = 0.0
            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("[Phase 2: Grasping Pen and Pointing Tip to Viewer]")
        time.sleep(0.5)

        # Stage 2: Lift pen up to chest height and orient tip
        for i in range(120):
            t = i / 120.0
            data.qpos[left_pitch_q] = 0.35 + (0.15 - 0.35) * t
            data.qpos[left_roll_q] = 0.25 + (0.10 - 0.25) * t
            data.qpos[left_elbow_q] = -0.40 + (-0.75 - (-0.40)) * t
            data.qpos[left_yaw_q] = 0.30 * t
            
            # Snap pen to hand end-effector during grasp
            data.qpos[pen_qpos_adr : pen_qpos_adr + 3] = np.array([0.30, 0.06, 0.60 + 0.05 * t])
            # Quaternion for pointing tip forward
            data.qpos[pen_qpos_adr + 3 : pen_qpos_adr + 7] = np.array([0.707, 0.707, 0, 0])
            
            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        print("\n>>> [Phase 3: Drawing Number '6' in the Air] <<<")
        time.sleep(0.5)

        # Stage 3: Follow trajectory drawing "6"
        for idx, (target_y, target_z) in enumerate(trajectory):
            # IK approximation for arm posture while drawing
            data.qpos[left_roll_q] = target_y * 2.0
            data.qpos[left_pitch_q] = 0.20 + (0.65 - target_z) * 1.5
            data.qpos[left_elbow_q] = -0.75 + (target_z - 0.60) * 0.8
            
            # Move pen tip smoothly along trajectory
            data.qpos[pen_qpos_adr : pen_qpos_adr + 3] = np.array([0.32, target_y, target_z])
            data.qpos[pen_qpos_adr + 3 : pen_qpos_adr + 7] = np.array([0.707, 0.707, 0, 0])
            
            data.qfrc_applied[:] = data.qfrc_bias[:]
            mujoco.mj_step(model, data)
            viewer.sync()
            
            if idx % 50 == 0:
                print(f"Drawing Progress: Step {idx}/{len(trajectory)} | Pen Tip Y: {target_y:+.3f}, Z: {target_z:+.3f}")
            time.sleep(0.015)

        print("\n[Phase 4: Returning Pen to Table]")
        time.sleep(0.5)

        # Stage 4: Lower pen back onto the table
        for i in range(120):
            t = i / 120.0
            data.qpos[left_pitch_q] = 0.20 + (0.35 - 0.20) * t
            data.qpos[left_roll_q] = 0.20 * (1 - t)
            data.qpos[left_elbow_q] = -0.75 + (-0.20 - (-0.75)) * t
            
            # Lower pen to table
            data.qpos[pen_qpos_adr : pen_qpos_adr + 3] = np.array([0.32, 0.12 * (1 - t), 0.58 + (0.395 - 0.58) * t])
            data.qpos[pen_qpos_adr + 3 : pen_qpos_adr + 7] = np.array([1, 0, 0, 0])  # Flat on table
            
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
