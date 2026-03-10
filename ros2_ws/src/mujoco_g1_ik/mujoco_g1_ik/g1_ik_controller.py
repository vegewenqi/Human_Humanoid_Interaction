import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8

import mujoco
import mujoco.viewer

from mink import Configuration, SE3, solve_ik

from .components.utils import pc2_to_xyz_array, se3_translation
from .components.retargeting import RetargetingConfig, Retargeter
from .components.tasks import TaskConfig, TaskSet
from .components import zed_indices as zi


@dataclass
class SkeletonFrame:
    points_xyz: np.ndarray
    confidence: int
    stamp_sec: float


class G1IKController(Node):
    def __init__(self):
        super().__init__("g1_ik_controller")

        # ---- Params ----
        self.declare_parameter("mjcf_path", "")
        self.declare_parameter("ee_site", "right_palm")
        self.declare_parameter("elbow_body", "right_elbow_link")

        self.declare_parameter("wrist_index", zi.RIGHT_WRIST)   # 17
        self.declare_parameter("elbow_index", zi.RIGHT_ELBOW)   # 15
        self.declare_parameter("pelvis_index", zi.PELIVS)       # 0

        self.declare_parameter("use_pelvis_relative", True)
        self.declare_parameter("skeleton_unit_scale", 0.001)
        self.declare_parameter("motion_gain", 0.6)
        self.declare_parameter("max_delta_m", 0.25)

        self.declare_parameter("ema_alpha", 0.25)
        self.declare_parameter("max_jump_m", 0.12)

        self.declare_parameter("ik_dt", 1.0/60.0)
        self.declare_parameter("sim_dt", 1.0/60.0)
        self.declare_parameter("solver", "daqp")

        # task weights
        self.declare_parameter("wrist_pos_cost", 1.0)
        self.declare_parameter("elbow_pos_cost", 0.35)
        self.declare_parameter("posture_cost", 0.02)
        self.declare_parameter("task_gain", 0.8)

        self.mjcf_path = self.get_parameter("mjcf_path").get_parameter_value().string_value
        if not self.mjcf_path:
            raise RuntimeError("mjcf_path is required")

        self.ee_site = self.get_parameter("ee_site").value
        self.elbow_body = self.get_parameter("elbow_body").value

        self.wrist_index = int(self.get_parameter("wrist_index").value)
        self.elbow_index = int(self.get_parameter("elbow_index").value)
        self.pelvis_index = int(self.get_parameter("pelvis_index").value)

        self.ik_dt = float(self.get_parameter("ik_dt").value)
        self.sim_dt = float(self.get_parameter("sim_dt").value)
        self.solver = self.get_parameter("solver").get_parameter_value().string_value

        # retargeting config
        r_cfg = RetargetingConfig(
            skeleton_unit_scale=float(self.get_parameter("skeleton_unit_scale").value),
            motion_gain=float(self.get_parameter("motion_gain").value),
            use_pelvis_relative=bool(self.get_parameter("use_pelvis_relative").value),
            max_delta_m=float(self.get_parameter("max_delta_m").value),
        )
        ema_alpha = float(self.get_parameter("ema_alpha").value)
        max_jump_m = float(self.get_parameter("max_jump_m").value)
        self.retargeter = Retargeter(r_cfg, ema_alpha=ema_alpha, max_jump_m=max_jump_m)

        # tasks config
        t_cfg = TaskConfig(
            ee_site=self.ee_site,
            elbow_body=self.elbow_body,
            wrist_pos_cost=float(self.get_parameter("wrist_pos_cost").value),
            elbow_pos_cost=float(self.get_parameter("elbow_pos_cost").value),
            posture_cost=float(self.get_parameter("posture_cost").value),
            task_gain=float(self.get_parameter("task_gain").value),
        )
        self.tasks = TaskSet(t_cfg)

        # ---- ROS subscriptions ----
        self._latest_conf: Optional[int] = None
        self._latest_frame: Optional[SkeletonFrame] = None
        self.create_subscription(PointCloud2, "/skeleton/points", self._on_points, 10)
        self.create_subscription(UInt8, "/skeleton/confidence", self._on_conf, 10)

        # ---- MuJoCo + mink ----
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt

        self.cfg = Configuration(self.model)
        # init config from qpos
        self.cfg.q[:] = self.data.qpos.copy()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.get_logger().info("MuJoCo viewer launched (passive).")

        self._print_model_debug()

        # ---- DoF mask: allow only right arm ----
        self.dof_mask = np.zeros(self.model.nv, dtype=np.float64)
        JT = mujoco.mjtJoint
        dof_count = {JT.mjJNT_FREE: 6, JT.mjJNT_BALL: 3, JT.mjJNT_SLIDE: 1, JT.mjJNT_HINGE: 1}
        allow_prefix = ("right_shoulder", "right_elbow", "right_wrist")
        allowed_joint_names = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith(allow_prefix):
                allowed_joint_names.append(name)
                adr = int(self.model.jnt_dofadr[j])
                n = int(dof_count[self.model.jnt_type[j]])
                self.dof_mask[adr:adr+n] = 1.0
        self.get_logger().info(f"Allowed joints: {allowed_joint_names}")
        self.get_logger().info(f"nv={self.model.nv}, allowed dofs={int(self.dof_mask.sum())}")

        # refs init flags
        self._refs_inited = False

        # main loop
        self.timer = self.create_timer(1.0/60.0, self._loop)

        self.get_logger().info(
            f"Started controller: wrist={self.wrist_index}, elbow={self.elbow_index}, pelvis={self.pelvis_index}, "
            f"use_pelvis_relative={r_cfg.use_pelvis_relative}, ee_site={self.ee_site}, elbow_body={self.elbow_body}"
        )

    def _print_model_debug(self):
        # sites
        site_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.model.nsite)]
        site_names = [s for s in site_names if s is not None]
        self.get_logger().info(f"Model has {len(site_names)} sites. Sample: {site_names[:20]}")

        # validate ee_site
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site)
        if ee_id < 0:
            raise RuntimeError(f"ee_site '{self.ee_site}' not found")

        # validate elbow body
        eb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.elbow_body)
        if eb_id < 0:
            raise RuntimeError(f"elbow_body '{self.elbow_body}' not found (try 'right_elbow_link')")

    def _on_conf(self, msg: UInt8):
        self._latest_conf = int(msg.data)

    def _on_points(self, msg: PointCloud2):
        pts = pc2_to_xyz_array(msg)
        if pts.size == 0:
            return
        conf = self._latest_conf if self._latest_conf is not None else -1
        self._latest_frame = SkeletonFrame(points_xyz=pts, confidence=conf, stamp_sec=time.time())

    def _loop(self):
        frame = self._latest_frame
        if frame is None:
            self._render_only()
            return

        # get joints (filtered + mapped)
        wrist = self.retargeter.joint_to_mujoco_world(frame.points_xyz, self.wrist_index)
        elbow = self.retargeter.joint_to_mujoco_world(frame.points_xyz, self.elbow_index)
        if wrist is None or elbow is None:
            self._render_only()
            return

        pelvis = None
        if self.retargeter.cfg.use_pelvis_relative:
            pelvis = self.retargeter.joint_to_mujoco_world(frame.points_xyz, self.pelvis_index)
            if pelvis is None:
                self._render_only()
                return

        # init refs once
        if not self._refs_inited:
            q_home = self.cfg.q.copy()
            self.tasks.initialize_refs(self.cfg, q_home=q_home)
            self.retargeter.set_refs(wrist=wrist, elbow=elbow, pelvis=pelvis)
            self._refs_inited = True
            self.get_logger().info("Initialized refs (ee/elbow + wrist/elbow/pelvis).")
            self._render_only()
            return

        # retarget to deltas
        dw, de = self.retargeter.compute_delta(wrist=wrist, elbow=elbow, pelvis=pelvis)
        self.tasks.set_targets_from_deltas(dw=dw, de=de)

        # solve IK for tasks
        task_list = self.tasks.build_tasks()
        try:
            vel = solve_ik(self.cfg, task_list, self.ik_dt, solver=self.solver, damping=1e-4)
        except TypeError:
            vel = solve_ik(self.cfg, task_list, self.ik_dt, damping=1e-4)

        # velocity damping in nv space
        # vel = vel + self.tasks.posture_damping(self.data.qvel.copy())

        # freeze non-right-arm dofs
        vel = vel * self.dof_mask

        # integrate
        self.cfg.integrate_inplace(vel, self.ik_dt)

        # apply kinematically
        self.data.qpos[:] = self.cfg.q.copy()
        mujoco.mj_forward(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

        # debug once per second
        if not hasattr(self, "_dbg_t"):
            self._dbg_t = time.time()
        elif time.time() - self._dbg_t > 1.0:
            ee_now = se3_translation(self.cfg.get_transform_frame_to_world(self.ee_site, "site"))
            ee_tgt = se3_translation(SE3.from_translation(dw) @ self.tasks.ee_ref)
            ee_err = float(np.linalg.norm(ee_tgt - ee_now))
            self.get_logger().info(f"ee_err={ee_err:.3f} m, |dw|={float(np.linalg.norm(dw)):.3f}, |de|={float(np.linalg.norm(de)):.3f}, conf={frame.confidence}")
            self._dbg_t = time.time()

    def _render_only(self):
        mujoco.mj_forward(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()


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