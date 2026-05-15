import time
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from .components.mujoco_29dof_urdf_indices import FULL_JOINT_MAP

import mujoco
import mujoco.viewer


class G1ActuatorController(Node):
    """
    Fixed-base actuator-driven upper-body controller for G1.

    Input topic:
        /g1_upperbody_q_des_safe   Float32MultiArray
        order:
        [waist_roll, waist_pitch, left_shoulder_pitch, left_shoulder_roll, left_elbow, right_shoulder_pitch, right_shoulder_roll, right_elbow]

    Control:
        Uses MuJoCo position actuators:
            data.ctrl[actuator_ids] = q_cmd

    Special behavior:
        - base is locked every step (good for upper-body-only debugging)
        - still uses data.ctrl + mj_step(), so this is actuator-driven, not direct qpos teleport
    """

    def __init__(self):
        super().__init__("g1_actuator_controller")

        # ---------------- params ----------------
        self.declare_parameter("mjcf_path", "/repos/unitree_g1/g1_mjx.xml")
        self.declare_parameter("qdes_topic", "/g1_upperbody_q_des_safe")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("qdes_in_degrees", False)

        self.declare_parameter("sim_dt", 1.0 / 250.0)
        self.declare_parameter("ctrl_dt", 1.0 / 100.0)

        self.declare_parameter("log_output", "both")    # qdes | q | ctrl | both
        self.declare_parameter("ema_alpha", 0.25)
        self.declare_parameter("max_rate_deg", 120.0)

        # fix root/base in simulation
        self.declare_parameter("lock_base", True)

        # show MuJoCo viewer window
        self.declare_parameter("show_viewer", True)

        # joint names = actuator names in your XML
        self.declare_parameter(
            "joint_names",
            [
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_elbow_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_elbow_joint",
            ],
        )

        # startup pose for the 8 controlled joints
        # home position: stand up straight with arms down
        self.declare_parameter(
            "q_home",
            [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708]
        )

        # controller-side clipping
        # (-30-30, -30-30, -177-153, -90-130, -60-120, -177-153, -130,90, -60-120)
        self.declare_parameter(
            "q_min",
            [-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472]
        )
        self.declare_parameter(
            "q_max",
            [0.52, 0.52, 2.6704, 2.2515, 2.0944, 2.6704, 1.5882, 2.0944]
        )

        self.declare_parameter("apply_q_home_on_start", True)

        self.declare_parameter("debug_log", False)
        self.declare_parameter("debug_log_period_sec", 1.0)
        self.debug_log = bool(self.get_parameter("debug_log").value)
        self.debug_log_period_sec = float(self.get_parameter("debug_log_period_sec").value)

        # ---------------- read params ----------------
        self.mjcf_path = str(self.get_parameter("mjcf_path").value)
        if not self.mjcf_path:
            raise RuntimeError("mjcf_path is required")

        self.qdes_topic = str(self.get_parameter("qdes_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.qdes_in_degrees = bool(self.get_parameter("qdes_in_degrees").value)

        self.sim_dt = float(self.get_parameter("sim_dt").value)
        self.ctrl_dt = float(self.get_parameter("ctrl_dt").value)

        self.log_output = str(self.get_parameter("log_output").value).strip().lower()
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.max_rate_deg = float(self.get_parameter("max_rate_deg").value)
        self.max_rate_rad = np.deg2rad(self.max_rate_deg)

        self.lock_base = bool(self.get_parameter("lock_base").value)
        self.show_viewer = bool(self.get_parameter("show_viewer").value)

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

        # topic for nominal q_des array
        self.create_subscription(
            Float32MultiArray, self.qdes_topic, self._on_qdes, 10
        )
        # topic for joint states as JointState for CBF node
        self.pub_joint_states = self.create_publisher(
            JointState, self.joint_state_topic, 10
        )

        # ---------------- MuJoCo ----------------
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = self.sim_dt

        self.viewer = None
        if self.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.get_logger().info("MuJoCo viewer launched (passive).")
        else:
            self.get_logger().info("MuJoCo viewer disabled by show_viewer parameter.")

        # detect floating base
        self.has_free_root = self._detect_free_root()

        # save initial base state for locking
        self.base_qpos0 = None
        if self.has_free_root:
            self.base_qpos0 = self.data.qpos[:7].copy()
            self.get_logger().info(
                f"Detected free root joint. lock_base={self.lock_base}, "
                f"base_qpos0={np.array2string(self.base_qpos0, precision=3)}"
            )
        else:
            self.get_logger().info(
                f"No free root joint detected. lock_base={self.lock_base} has no effect."
            )

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

        # ---------------- full 29-DoF joint-state publishing ----------------
        self.full_joint_names = [
            name for name, _ in sorted(FULL_JOINT_MAP.items(), key=lambda kv: kv[1])
        ]

        self.full_joint_ids = []
        self.full_qpos_ids = []
        self.full_dof_ids = []

        for name in self.full_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Full joint '{name}' not found in MJCF")

            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(
                    f"Full joint '{name}' is not a hinge joint; "
                    "full joint-state publishing assumes hinge joints only."
                )

            self.full_joint_ids.append(jid)
            self.full_qpos_ids.append(int(self.model.jnt_qposadr[jid]))
            self.full_dof_ids.append(int(self.model.jnt_dofadr[jid]))

        self.full_qpos_ids = np.asarray(self.full_qpos_ids, dtype=np.int32)
        self.full_dof_ids = np.asarray(self.full_dof_ids, dtype=np.int32)

        self.get_logger().info(
            f"Full joint-state publishing enabled with {len(self.full_joint_names)} joints."
        )
        self.get_logger().info(
            f"Full joint-state names: {self.full_joint_names}"
        )

        # initialize all actuators to current qpos to avoid weird targets
        self._initialize_full_ctrl()

        # initialize controlled joint command
        q_now = self.data.qpos[self.qpos_ids].copy()
        if self.apply_q_home_on_start:
            q0 = np.clip(self.q_home.copy(), self.q_min, self.q_max)
            self.q_cmd = q0.copy()
            self.data.ctrl[self.actuator_ids] = self.q_cmd

            # let the actuators settle for a short while
            for _ in range(50):
                self._apply_base_lock()
                mujoco.mj_step(self.model, self.data)
        else:
            self.q_cmd = q_now.copy()

        self.get_logger().info(
            f"Initial controlled q = {np.array2string(self.data.qpos[self.qpos_ids], precision=3)}"
        )

        self.timer = self.create_timer(self.ctrl_dt, self._loop)
        # publish one full joint-state immediately so downstream nodes
        # (CBF / robot_state_publisher / RViz) do not wait for first timer tick
        self._publish_joint_states()

        self.get_logger().info(
            f"Started G1ActuatorController: qdes_topic={self.qdes_topic}, "
            f"ctrl_dt={self.ctrl_dt:.4f}, sim_dt={self.sim_dt:.4f}, "
            f"lock_base={self.lock_base}, show_viewer={self.show_viewer}"
        )

    def _detect_free_root(self) -> bool:
        if self.model.njnt <= 0:
            return False
        root_type = self.model.jnt_type[0]
        return root_type == mujoco.mjtJoint.mjJNT_FREE

    def _apply_base_lock(self):
        if not self.lock_base:
            return
        if not self.has_free_root:
            return

        # lock free base pose and velocity
        self.data.qpos[:7] = self.base_qpos0
        self.data.qvel[:6] = 0.0

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

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.full_joint_names)

        # Publish full 29-DoF current MuJoCo joint positions
        q_now = self.data.qpos[self.full_qpos_ids].copy()
        msg.position = [float(x) for x in q_now]

        # Publish full 29-DoF current MuJoCo joint velocities
        v_now = self.data.qvel[self.full_dof_ids].copy()
        msg.velocity = [float(x) for x in v_now]

        msg.effort = []
        self.pub_joint_states.publish(msg)

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
        q_next = (1.0 - self.ema_alpha) * self.q_cmd + self.ema_alpha * q_limited

        # final clip
        q_next = np.clip(q_next, self.q_min, self.q_max)
        self.q_cmd = q_next

        # write only the 6 controlled actuators
        self.data.ctrl[self.actuator_ids] = self.q_cmd

        # lock base before stepping
        self._apply_base_lock()

        # step physics
        mujoco.mj_step(self.model, self.data)

        # lock base again after stepping, for extra stability
        self._apply_base_lock()
        mujoco.mj_forward(self.model, self.data)

        # publish JointState for CBF node
        self._publish_joint_states()

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        # logging
        now = time.time()
        if now - self.last_log_t > 1.0:
            if self.log_output in ["qdes", "both"]:
                qd_deg = np.rad2deg(self.q_cmd)
                if self.debug_log:
                    self.get_logger().info(
                        "[q_des_cmd_deg] "
                        f"waist_roll={qd_deg[0]:.2f}, waist_pitch={qd_deg[1]:.2f}, "
                        f"l_sh_pitch={qd_deg[2]:.2f}, l_sh_roll={qd_deg[3]:.2f}, l_elbow={qd_deg[4]:.2f}, "
                        f"r_sh_pitch={qd_deg[5]:.2f}, r_sh_roll={qd_deg[6]:.2f}, r_elbow={qd_deg[7]:.2f}",
                        throttle_duration_sec=self.debug_log_period_sec,
                    )

            if self.log_output in ["q", "both"]:
                q_deg = np.rad2deg(self.data.qpos[self.qpos_ids])
                if self.debug_log:
                    self.get_logger().info(
                        "[q_now_deg] "
                    f"waist_roll={q_deg[0]:.2f}, waist_pitch={q_deg[1]:.2f}, "
                    f"l_sh_pitch={q_deg[2]:.2f}, l_sh_roll={q_deg[3]:.2f}, l_elbow={q_deg[4]:.2f}, "
                    f"r_sh_pitch={q_deg[5]:.2f}, r_sh_roll={q_deg[6]:.2f}, r_elbow={q_deg[7]:.2f}",
                    throttle_duration_sec=self.debug_log_period_sec,
                )

            if self.log_output in ["ctrl", "both"]:
                c_deg = np.rad2deg(self.data.ctrl[self.actuator_ids])
                if self.debug_log:
                    self.get_logger().info(
                        "[ctrl_position_deg] "
                    f"waist_roll={c_deg[0]:.2f}, waist_pitch={c_deg[1]:.2f}, "
                    f"l_sh_pitch={c_deg[2]:.2f}, l_sh_roll={c_deg[3]:.2f}, l_elbow={c_deg[4]:.2f}, "
                    f"r_sh_pitch={c_deg[5]:.2f}, r_sh_roll={c_deg[6]:.2f}, r_elbow={c_deg[7]:.2f}",
                    throttle_duration_sec=self.debug_log_period_sec,
                )

            self.last_log_t = now

    def _step_only(self):
        self._apply_base_lock()
        mujoco.mj_step(self.model, self.data)
        self._apply_base_lock()
        mujoco.mj_forward(self.model, self.data)
        self._publish_joint_states()
        if self.viewer is not None and self.viewer.is_running():
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