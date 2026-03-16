import time
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import mujoco
import mujoco.viewer


class G1ActuatorController(Node):
    """
    Actuator-driven upper-body controller for G1.

    Input topic:
        /g1_upperbody_q_des   Float32MultiArray
        order:
        [waist_roll, waist_pitch, left_shoulder_roll, left_elbow, right_shoulder_roll, right_elbow]

    Control:
        Uses MuJoCo position actuators:
            data.ctrl[actuator_ids] = q_cmd

    Notes:
        - Designed for g1_mjx.xml first.
        - In both g1_mjx.xml and g1.xml, actuator names for these joints
          are the same as the joint names.
    """

    def __init__(self):
        super().__init__("g1_actuator_controller")

        # ---------------- params ----------------
        self.declare_parameter("mjcf_path", "/third_party/mujoco_menagerie/unitree_g1/g1_mjx.xml")
        self.declare_parameter("qdes_topic", "/g1_upperbody_q_des")
        self.declare_parameter("qdes_in_degrees", False)

        self.declare_parameter("sim_dt", 1.0 / 250.0)   # g1_mjx.xml timestep = 0.004
        self.declare_parameter("ctrl_dt", 1.0 / 60.0)

        self.declare_parameter("log_output", "both")    # qdes | q | ctrl | both
        self.declare_parameter("ema_alpha", 0.25)
        self.declare_parameter("max_rate_deg", 120.0)

        # joint names = actuator names in your XML
        self.declare_parameter(
            "joint_names",
            [
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_roll_joint",
                "left_elbow_joint",
                "right_shoulder_roll_joint",
                "right_elbow_joint",
            ],
        )

        # startup pose for the 6 controlled joints
        self.declare_parameter(
            "q_home",
            [0.0, 0.0, 0.0, 0.55, 0.0, 0.55]
        )

        # controller-side clipping
        # ranges chosen to match your XML upper-body joint ranges conservatively:
        # waist_roll   [-0.52, 0.52]
        # waist_pitch  [-0.52, 0.52]
        # left_sh_roll [-1.5882, 2.2515]
        # left_elbow   [-1.0472, 2.0944]
        # right_sh_roll[-2.2515, 1.5882]
        # right_elbow  [-1.0472, 2.0944]
        self.declare_parameter(
            "q_min",
            [-0.52, -0.52, -1.5882, -1.0472, -2.2515, -1.0472]
        )
        self.declare_parameter(
            "q_max",
            [0.52, 0.52,  2.2515,  2.0944,  1.5882,  2.0944]
        )

        self.declare_parameter("apply_q_home_on_start", True)

        # ---------------- read params ----------------
        self.mjcf_path = str(self.get_parameter("mjcf_path").value)
        if not self.mjcf_path:
            raise RuntimeError("mjcf_path is required")

        self.qdes_topic = str(self.get_parameter("qdes_topic").value)
        self.qdes_in_degrees = bool(self.get_parameter("qdes_in_degrees").value)

        self.sim_dt = float(self.get_parameter("sim_dt").value)
        self.ctrl_dt = float(self.get_parameter("ctrl_dt").value)

        self.log_output = str(self.get_parameter("log_output").value).strip().lower()
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.max_rate_deg = float(self.get_parameter("max_rate_deg").value)
        self.max_rate_rad = np.deg2rad(self.max_rate_deg)

        self.joint_names: List[str] = list(self.get_parameter("joint_names").value)
        self.q_home = np.array(self.get_parameter("q_home").value, dtype=np.float64)
        self.q_min = np.array(self.get_parameter("q_min").value, dtype=np.float64)
        self.q_max = np.array(self.get_parameter("q_max").value, dtype=np.float64)
        self.apply_q_home_on_start = bool(self.get_parameter("apply_q_home_on_start").value)

        self.n = len(self.joint_names)
        for arr_name, arr in [
            ("q_home", self.q_home),
            ("q_min", self.q_min),
            ("q_max", self.q_max),
        ]:
            if arr.shape[0] != self.n:
                raise ValueError(f"{arr_name} must have length {self.n}")

        # ---------------- ROS ----------------
        self.latest_q_des: Optional[np.ndarray] = None
        self.q_cmd: Optional[np.ndarray] = None
        self.last_log_t = time.time()

        self.create_subscription(
            Float32MultiArray, self.qdes_topic, self._on_qdes, 10
        )

        # ---------------- MuJoCo ----------------
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        # overwrite model timestep if desired
        self.model.opt.timestep = self.sim_dt

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.get_logger().info("MuJoCo viewer launched (passive).")

        # resolve joint ids / qpos ids / actuator ids
        self.joint_ids = []
        self.qpos_ids = []
        self.actuator_ids = []

        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint '{name}' not found in MJCF")

            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(
                    f"Joint '{name}' is not a hinge joint. "
                    "This controller assumes hinge joints only."
                )

            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(
                    f"Actuator '{name}' not found in MJCF. "
                    "In your XML, actuator names should match joint names."
                )

            self.joint_ids.append(jid)
            self.qpos_ids.append(int(self.model.jnt_qposadr[jid]))
            self.actuator_ids.append(aid)

        self.qpos_ids = np.asarray(self.qpos_ids, dtype=np.int32)
        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)

        self.get_logger().info(f"Controlled joints: {self.joint_names}")
        self.get_logger().info(f"Controlled qpos ids: {self.qpos_ids.tolist()}")
        self.get_logger().info(f"Controlled actuator ids: {self.actuator_ids.tolist()}")

        # initialize a sane whole-body actuator ctrl
        self._initialize_full_ctrl()

        # initialize controlled joint command
        q_now = self.data.qpos[self.qpos_ids].copy()
        if self.apply_q_home_on_start:
            q0 = np.clip(self.q_home.copy(), self.q_min, self.q_max)
            self.q_cmd = q0.copy()
            self.data.ctrl[self.actuator_ids] = self.q_cmd
            # let the actuators pull the robot toward q_home for a few steps
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
        else:
            self.q_cmd = q_now.copy()

        self.get_logger().info(
            f"Initial controlled q = {np.array2string(self.data.qpos[self.qpos_ids], precision=3)}"
        )

        self.timer = self.create_timer(self.ctrl_dt, self._loop)

        self.get_logger().info(
            f"Started G1ActuatorController: qdes_topic={self.qdes_topic}, "
            f"ctrl_dt={self.ctrl_dt:.4f}, sim_dt={self.sim_dt:.4f}"
        )

    def _initialize_full_ctrl(self):
        """
        Initialize all position actuators to current joint qpos so that
        uncontrolled joints do not get strange actuator targets.
        """
        self.data.ctrl[:] = 0.0

        for aid in range(self.model.nu):
            trnid = int(self.model.actuator_trnid[aid, 0])
            if trnid < 0:
                continue

            jtype = self.model.jnt_type[trnid]
            if jtype != mujoco.mjtJoint.mjJNT_HINGE:
                continue

            qpos_adr = int(self.model.jnt_qposadr[trnid])
            self.data.ctrl[aid] = self.data.qpos[qpos_adr]

        mujoco.mj_forward(self.model, self.data)

    def _on_qdes(self, msg: Float32MultiArray):
        if len(msg.data) != self.n:
            self.get_logger().warn(
                f"Expected q_des dim={self.n}, got {len(msg.data)}. Skip."
            )
            return

        q_des = np.array(msg.data, dtype=np.float64)
        if self.qdes_in_degrees:
            q_des = np.deg2rad(q_des)

        q_des = np.clip(q_des, self.q_min, self.q_max)
        self.latest_q_des = q_des

    def _loop(self):
        if self.latest_q_des is None:
            self._step_only()
            return

        q_target = self.latest_q_des.copy()

        # rate limit
        max_step = self.max_rate_rad * self.ctrl_dt
        dq = q_target - self.q_cmd
        dq = np.clip(dq, -max_step, max_step)
        q_limited = self.q_cmd + dq

        # EMA smoothing
        q_next = self.ema_alpha * self.q_cmd + (1.0 - self.ema_alpha) * q_limited

        # final clip
        q_next = np.clip(q_next, self.q_min, self.q_max)
        self.q_cmd = q_next

        # write only the 6 controlled actuators
        self.data.ctrl[self.actuator_ids] = self.q_cmd

        # step physics
        mujoco.mj_step(self.model, self.data)

        if self.viewer.is_running():
            self.viewer.sync()

        # logging
        now = time.time()
        if now - self.last_log_t > 1.0:
            if self.log_output in ["qdes", "both"]:
                qd_deg = np.rad2deg(self.q_cmd)
                self.get_logger().info(
                    "[q_des_cmd_deg] "
                    f"waist_roll={qd_deg[0]:.2f}, waist_pitch={qd_deg[1]:.2f}, "
                    f"l_sh_roll={qd_deg[2]:.2f}, l_elbow={qd_deg[3]:.2f}, "
                    f"r_sh_roll={qd_deg[4]:.2f}, r_elbow={qd_deg[5]:.2f}"
                )

            if self.log_output in ["q", "both"]:
                q_deg = np.rad2deg(self.data.qpos[self.qpos_ids])
                self.get_logger().info(
                    "[q_now_deg] "
                    f"waist_roll={q_deg[0]:.2f}, waist_pitch={q_deg[1]:.2f}, "
                    f"l_sh_roll={q_deg[2]:.2f}, l_elbow={q_deg[3]:.2f}, "
                    f"r_sh_roll={q_deg[4]:.2f}, r_elbow={q_deg[5]:.2f}"
                )

            if self.log_output in ["ctrl", "both"]:
                c_deg = np.rad2deg(self.data.ctrl[self.actuator_ids])
                self.get_logger().info(
                    "[ctrl_position_deg] "
                    f"waist_roll={c_deg[0]:.2f}, waist_pitch={c_deg[1]:.2f}, "
                    f"l_sh_roll={c_deg[2]:.2f}, l_elbow={c_deg[3]:.2f}, "
                    f"r_sh_roll={c_deg[4]:.2f}, r_elbow={c_deg[5]:.2f}"
                )

            self.last_log_t = now

    def _step_only(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()


def main():
    rclpy.init()
    node = None
    try:
        node = G1ActuatorController()
        rclpy.spin(node)
    finally:
        if node is not None:
            try:
                if hasattr(node, "viewer") and node.viewer is not None:
                    node.viewer.close()
            except Exception:
                pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()