import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from sensor_msgs_py import point_cloud2

import mujoco
import mujoco.viewer

from mink import Configuration, FrameTask, SE3, solve_ik


@dataclass
class SkeletonFrame:
    points_xyz: np.ndarray  # shape (N, 3)
    confidence: int
    stamp_sec: float


class G1IKController(Node):
    """
    MVP controller:
      - Subscribes to /skeleton/points (PointCloud2) and /skeleton/confidence (UInt8)
      - Loads Unitree G1 MJCF (Menagerie)
      - Uses mink differential IK to move a hand site toward a target position.
      - Target is defined as relative motion: ee_target = ee_ref + gain * (p - p_ref) (after mm->m)
    """

    def __init__(self):
        super().__init__("g1_ik_controller")

        # -------- Parameters (easy to tune later) --------
        self.declare_parameter("mjcf_path", "")
        self.declare_parameter("ee_site", "")  # site name in MJCF
        self.declare_parameter("target_joint_index", 15)  # which skeleton point to track
        self.declare_parameter("skeleton_unit_scale", 0.001)  # mm -> m
        self.declare_parameter("motion_gain", 0.6)  # how much of human delta to apply
        self.declare_parameter("ik_dt", 1.0 / 120.0)  # IK timestep
        self.declare_parameter("sim_dt", 1.0 / 240.0)  # MuJoCo physics timestep
        self.declare_parameter("solver", "daqp")  # mink QP solver backend (fallback to default if not installed)

        self.mjcf_path = self.get_parameter("mjcf_path").get_parameter_value().string_value
        self.ee_site = self.get_parameter("ee_site").get_parameter_value().string_value
        self.target_joint_index = int(self.get_parameter("target_joint_index").value)
        self.skel_scale = float(self.get_parameter("skeleton_unit_scale").value)
        self.motion_gain = float(self.get_parameter("motion_gain").value)
        self.ik_dt = float(self.get_parameter("ik_dt").value)
        self.sim_dt = float(self.get_parameter("sim_dt").value)
        self.solver = self.get_parameter("solver").get_parameter_value().string_value

        if not self.mjcf_path:
            raise RuntimeError("Parameter 'mjcf_path' is required.")
        if not self.ee_site:
            raise RuntimeError("Parameter 'ee_site' is required.")

        # -------- ROS subscriptions --------
        self._latest_cloud: Optional[PointCloud2] = None
        self._latest_conf: Optional[int] = None
        self._latest_frame: Optional[SkeletonFrame] = None

        self.create_subscription(PointCloud2, "/skeleton/points", self._on_points, 10)
        self.create_subscription(UInt8, "/skeleton/confidence", self._on_conf, 10)

        # -------- MuJoCo + mink setup --------
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Use fixed timestep
        self.model.opt.timestep = self.sim_dt

        self.cfg = Configuration(self.model)

        # If model has a "home" keyframe you can use it; otherwise just sync default qpos.
        if self.model.nkey > 0:
            # Try to use keyframe 0
            try:
                self.cfg.update_from_keyframe(self.model.keyframe(0).name)
            except Exception:
                # fallback to qpos
                self.cfg.q[:] = self.data.qpos.copy()
        else:
            self.cfg.q[:] = self.data.qpos.copy()

        # FrameTask: position-only by setting orientation_cost=0.0
        self.ee_task = FrameTask(
            frame_name=self.ee_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            gain=0.8,  # smoothness (smaller => smoother)
        )

        # Reference poses for relative tracking
        self._p_ref: Optional[np.ndarray] = None
        self._ee_ref: Optional[SE3] = None

        # Viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.get_logger().info("MuJoCo viewer launched (passive).")

        # Print some debugging info once
        self._print_model_debug()

        # ---- Allow only right-arm DoFs (freeze base/legs/torso/left arm) ----
        self.dof_mask = np.zeros(self.model.nv, dtype=np.float64)

        # Helper: how many DoFs a joint type has
        JT = mujoco.mjtJoint
        dof_count = {
            JT.mjJNT_FREE: 6,
            JT.mjJNT_BALL: 3,
            JT.mjJNT_SLIDE: 1,
            JT.mjJNT_HINGE: 1,
        }

        # Allow these joints (prefix match)
        allow_prefix = ("right_shoulder", "right_elbow", "right_wrist")

        allowed_joint_names = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            if name.startswith(allow_prefix):
                allowed_joint_names.append(name)
                adr = int(self.model.jnt_dofadr[j])
                n = int(dof_count[self.model.jnt_type[j]])
                self.dof_mask[adr:adr+n] = 1.0

        self.get_logger().info(f"Allowed joints: {allowed_joint_names}")
        self.get_logger().info(f"nv={self.model.nv}, allowed dofs={int(self.dof_mask.sum())}")

        # Main loop timer
        self.timer = self.create_timer(1.0/60.0, self._loop)

        self.get_logger().info(
            f"Started G1 IK controller. Tracking joint index={self.target_joint_index}, "
            f"ee_site='{self.ee_site}', mjcf='{self.mjcf_path}'"
        )

    def _print_model_debug(self):
        # Show a few sites/bodies to help you pick correct ee_site
        site_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.model.nsite)]
        site_names = [s for s in site_names if s is not None]
        self.get_logger().info(f"Model has {len(site_names)} sites. Sample: {site_names[:20]}")

        # Validate ee_site exists
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)
        if ee_id < 0:
            raise RuntimeError(f"ee_site '{self.ee_site}' not found in MJCF sites.")

    def _on_conf(self, msg: UInt8):
        self._latest_conf = int(msg.data)

    def _on_points(self, msg: PointCloud2):
        self._latest_cloud = msg
        pts = pc2_to_xyz_array(msg)
        if pts.size == 0:
            return
        conf = self._latest_conf if self._latest_conf is not None else -1
        self._latest_frame = SkeletonFrame(points_xyz=pts, confidence=conf, stamp_sec=time.time())

    def _get_target_point_world(self, frame: SkeletonFrame) -> Optional[np.ndarray]:
        idx = self.target_joint_index
        if idx < 0 or idx >= frame.points_xyz.shape[0]:
            return None
        p = frame.points_xyz[idx].astype(np.float64)

        # If point is invalid, skip
        if not np.all(np.isfinite(p)):
            return None

        # Convert to meters
        p_m = p * self.skel_scale

        # Coordinate mapping (simple default):
        # ZED RIGHT_HANDED_Y_UP is typically: X right, Y up, Z forward.
        # MuJoCo is typically: X forward, Y left, Z up.
        # A reasonable mapping (can be tuned later):
        #   mujoco_x = zed_z
        #   mujoco_y = -zed_x
        #   mujoco_z = zed_y
        p_mj = np.array([p_m[2], -p_m[0], p_m[1]], dtype=np.float64)

        return p_mj

    def _loop(self):
        # Need a frame before doing anything
        frame = self._latest_frame
        if frame is None:
            self._step_sim_only()
            return

        target_p = self._get_target_point_world(frame)
        if target_p is None:
            self._step_sim_only()
            return

        # Initialize references on first valid frame
        if self._p_ref is None or self._ee_ref is None:
            self._p_ref = target_p.copy()
            self._ee_ref = self.cfg.get_transform_frame_to_world(self.ee_site, "site")
            self.ee_task.set_target(self._ee_ref)
            self.get_logger().info("Initialized reference skeleton point and end-effector pose.")
            self._step_sim_only()
            return

        # Relative motion: delta from reference
        delta = (target_p - self._p_ref) * self.motion_gain

        # Debugging
        if not hasattr(self, "_dbg_t"):
            self._dbg_t = time.time()
        elif time.time() - self._dbg_t > 1.0:
            self.get_logger().info(f"delta_norm={float(np.linalg.norm(delta)):.4f}, conf={frame.confidence}")
            self._dbg_t = time.time()

        # Target pose = ee_ref translated by delta in world frame
        ee_target = SE3.from_translation(delta) @ self._ee_ref
        self.ee_task.set_target(ee_target)

        # Solve IK (differential)
        try:
            vel = solve_ik(self.cfg, [self.ee_task], self.ik_dt, solver=self.solver, damping=1e-4)
        except TypeError:
            # Some installs may not expose solver kwarg; fallback to default
            vel = solve_ik(self.cfg, [self.ee_task], self.ik_dt, damping=1e-4)
        except Exception as e:
            self.get_logger().warn(f"IK failed: {e}")
            self._step_sim_only()
            return
        
        # Freeze all DoFs except right arm
        vel = vel * self.dof_mask

        # Integrate IK result into configuration
        self.cfg.integrate_inplace(vel, self.ik_dt)

        # Apply to MuJoCo:
        # simplest: overwrite qpos and step physics.
        # More "actuator correct" later: set data.ctrl for position actuators.
        self.data.qpos[:] = self.cfg.q.copy()

        # Step and render
        mujoco.mj_forward(self.model, self.data)
        # mujoco.mj_step(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def _step_sim_only(self):
        # step physics and render without IK
        mujoco.mj_step(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

def pc2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """
    Robust PointCloud2 -> (N,3) float32 conversion for different ROS2/sensor_msgs_py behaviors.
    """
    # Try numpy path first
    try:
        arr = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        if arr is None:
            raise ValueError("read_points_numpy returned None")

        # Case 1: structured array with fields
        if hasattr(arr, "dtype") and arr.dtype.fields is not None:
            pts = np.empty((arr.shape[0], 3), dtype=np.float32)
            pts[:, 0] = arr["x"]
            pts[:, 1] = arr["y"]
            pts[:, 2] = arr["z"]
            return pts

        # Case 2: plain Nx3 numeric array
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(np.float32, copy=False)

        # Case 3: something else (fallback)
        raise ValueError(f"Unexpected numpy shape/dtype: shape={arr.shape}, dtype={arr.dtype}")

    except Exception:
        # Fallback: iterate points and force tuples
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
        pts = np.array([(float(p[0]), float(p[1]), float(p[2])) for p in gen], dtype=np.float32)
        return pts


def main():
    rclpy.init()
    node = None
    try:
        node = G1IKController()
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