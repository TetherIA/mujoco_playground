import time
import numpy as np
import mujoco
import mujoco.viewer
import glfw

MODEL = "/home/nan/CodeRepos/tetheria_rl/mujoco_playground/mujoco_playground/_src/manipulation/tetheria_hand/xmls/PreGen1_RightHand_Tendon_thumb.xml"

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)

# Sensor slice (<tendonpos name="len_t1" tendon="pf_tendon"/>)
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "len_t1")
sadr = m.sensor_adr[sid]
sdim = m.sensor_dim[sid]

# Tendon id
tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "th_tendon2")

with mujoco.viewer.launch_passive(m, d) as viewer:
    dt = m.opt.timestep
    last_update = 0.0

    # Initialize min/max
    min_len, max_len = float("inf"), float("-inf")

    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()

        now = time.time()
        if now - last_update > 0.1:
            len_sensor = float(d.sensordata[sadr : sadr + sdim][0])
            len_direct = float(d.ten_length[tid])
            vel_direct = float(d.ten_velocity[tid])

            # Update min/max
            min_len = min(min_len, len_direct)
            max_len = max(max_len, len_direct)

            text = (
                f"pf_tendon  len(sensor)={len_sensor:.6f}  "
                f"len(data)={len_direct:.6f}  vel={vel_direct:.6f}\n"
                f"min={min_len:.6f}  max={max_len:.6f}"
            )

            try:
                viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Tendon", text)
            except Exception:
                try:
                    glfw.set_window_title(viewer.window, text)
                except Exception:
                    print(text)

            last_update = now

    # Print final results after window closes
    print(f"Final tendon range: min={min_len:.6f}, max={max_len:.6f}")
