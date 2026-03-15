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

        self.declare_parameter("init_avg_frames", 15)
        self.declare_parameter("init_min_conf", 85)

        # task weights
        self.declare_parameter("wrist_pos_cost", 1.0)
        self.declare_parameter("elbow_pos_cost", 0.35)
        self.declare_parameter("posture_cost", 0.02)
        self.declare_parameter("task_gain", 0.8)
        self.declare_parameter("wrist_ori_cost", 0.08)
        self.declare_parameter("posture_max_vel", 0.8)
        self.declare_parameter("elbow_avoid_gain", 0.8)
        self.declare_parameter("elbow_avoid_margin_y", 0.18)
        self.declare_parameter("elbow_avoid_margin_x", 0.02)

        self.declare_parameter("home_shoulder_pitch", 0.20)
        self.declare_parameter("home_shoulder_roll", -0.35)
        self.declare_parameter("home_shoulder_yaw", 0.10)
        self.declare_parameter("home_elbow", 0.55)
        self.declare_parameter("home_wrist_roll", 0.00)
        self.declare_parameter("home_wrist_pitch", 0.00)
        self.declare_parameter("home_wrist_yaw", 0.00)

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

        self.init_avg_frames = int(self.get_parameter("init_avg_frames").value)
        self.init_min_conf = int(self.get_parameter("init_min_conf").value)

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
            wrist_ori_cost=float(self.get_parameter("wrist_ori_cost").value),
            posture_cost=float(self.get_parameter("posture_cost").value),
            posture_max_vel=float(self.get_parameter("posture_max_vel").value),
            task_gain=float(self.get_parameter("task_gain").value),
            elbow_avoid_gain=float(self.get_parameter("elbow_avoid_gain").value),
            elbow_avoid_margin_y=float(self.get_parameter("elbow_avoid_margin_y").value),
            elbow_avoid_margin_x=float(self.get_parameter("elbow_avoid_margin_x").value),
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
        dof_count = {
            JT.mjJNT_FREE: 6,
            JT.mjJNT_BALL: 3,
            JT.mjJNT_SLIDE: 1,
            JT.mjJNT_HINGE: 1,
        }

        allow_prefix = ("right_shoulder", "right_elbow", "right_wrist")
        allowed_joint_names = []

        self.arm_dof_ids = []
        self.arm_qpos_ids = []

        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith(allow_prefix):
                allowed_joint_names.append(name)

                dof_adr = int(self.model.jnt_dofadr[j])
                qpos_adr = int(self.model.jnt_qposadr[j])
                n = int(dof_count[self.model.jnt_type[j]])

                self.dof_mask[dof_adr:dof_adr+n] = 1.0

                if n == 1:
                    self.arm_dof_ids.append(dof_adr)
                    self.arm_qpos_ids.append(qpos_adr)

        self.arm_dof_ids = np.asarray(self.arm_dof_ids, dtype=np.int32)
        self.arm_qpos_ids = np.asarray(self.arm_qpos_ids, dtype=np.int32)
        self.arm_q_home = None
        self.get_logger().info(f"Allowed joints: {allowed_joint_names}")
        self.get_logger().info(f"nv={self.model.nv}, allowed dofs={int(self.dof_mask.sum())}")

        # refs init flags
        self._refs_inited = False
        self._init_wrist_buf = []
        self._init_elbow_buf = []
        self._init_pelvis_buf = []

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
            if self._try_initialize_refs(wrist=wrist, elbow=elbow, pelvis=pelvis, conf=frame.confidence):
                ee_ref_xyz = se3_translation(self.tasks.ee_ref)
                elbow_ref_xyz_robot = se3_translation(self.tasks.elbow_ref)

                self.get_logger().info(
                    f"[INIT] conf={frame.confidence}, "
                    f"human_wrist_ref={np.array2string(self.retargeter.wrist_ref, precision=3)}, "
                    f"human_elbow_ref={np.array2string(self.retargeter.elbow_ref, precision=3)}, "
                    f"human_pelvis_ref={np.array2string(self.retargeter.pelvis_ref, precision=3) if self.retargeter.pelvis_ref is not None else 'None'}"
                )
                self.get_logger().info(
                    f"[INIT] robot_ee_ref={np.array2string(ee_ref_xyz, precision=3)}, "
                    f"robot_elbow_ref={np.array2string(elbow_ref_xyz_robot, precision=3)}"
                )

                self._render_only()
                return
            self._render_only()
            return

        # retarget to deltas
        dw, de = self.retargeter.compute_delta(wrist=wrist, elbow=elbow, pelvis=pelvis)
        self.tasks.set_targets_from_deltas(dw=dw, de=de)

        self.tasks._tmp_elbow_avoid_task = None
        self.tasks.elbow_avoid_velocity(
            cfg_obj=self.cfg,
            model=self.model,
            elbow_body=self.elbow_body,
            torso_body="torso_link",
        )

        # solve IK for tasks
        task_list = self.tasks.build_tasks()
        try:
            vel = solve_ik(self.cfg, task_list, self.ik_dt, solver=self.solver, damping=1e-4)
        except TypeError:
            vel = solve_ik(self.cfg, task_list, self.ik_dt, damping=1e-4)

        # posture regularization
        if self.arm_q_home is not None:
            vel = vel + self.tasks.posture_velocity(
                q=self.cfg.q,
                arm_q_home=self.arm_q_home,
                arm_qpos_ids=self.arm_qpos_ids,
                arm_dof_ids=self.arm_dof_ids,
                nv=self.model.nv,
            )

        # freeze non-right-arm dofs
        vel = vel * self.dof_mask

        # integrate
        self.cfg.integrate_inplace(vel, self.ik_dt)

        # apply kinematically
        self.data.qpos[:] = self.cfg.q.copy()
        mujoco.mj_forward(self.model, self.data)
        # mujoco.mj_step(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

        # debug once per second
        if not hasattr(self, "_dbg_t"):
            self._dbg_t = time.time()
        elif time.time() - self._dbg_t > 1.0:
            ee_now = se3_translation(self.cfg.get_transform_frame_to_world(self.ee_site, "site"))
            ee_tgt = se3_translation(SE3.from_translation(dw) @ self.tasks.ee_ref)
            ee_err = float(np.linalg.norm(ee_tgt - ee_now))

            dbg = getattr(self.retargeter, "_last_debug", {})

            vel_norm_before_mask = float(np.linalg.norm(vel))
            vel_norm_after_mask = float(np.linalg.norm(vel * self.dof_mask))

            self.get_logger().info(
                "[DBG] "
                f"conf={frame.confidence}, "
                f"ee_err={ee_err:.3f} m, "
                f"|dw_raw|={dbg.get('dw_raw_norm', -1.0):.3f}, "
                f"|de_raw|={dbg.get('de_raw_norm', -1.0):.3f}, "
                f"|dw|={dbg.get('dw_norm', -1.0):.3f}, "
                f"|de|={dbg.get('de_norm', -1.0):.3f}, "
                f"dw_clamped={dbg.get('dw_clamped', False)}, "
                f"de_clamped={dbg.get('de_clamped', False)}, "
                f"|vel|={vel_norm_before_mask:.3f}, "
                f"|vel_masked|={vel_norm_after_mask:.3f}"
            )

            self.get_logger().info(
                "[DBG_VEC] "
                f"w_now={np.array2string(dbg.get('w_now', np.zeros(3)), precision=3)}, "
                f"e_now={np.array2string(dbg.get('e_now', np.zeros(3)), precision=3)}, "
                f"w_ref={np.array2string(dbg.get('w_ref', np.zeros(3)), precision=3)}, "
                f"e_ref={np.array2string(dbg.get('e_ref', np.zeros(3)), precision=3)}"
            )

            self.get_logger().info(
                "[DBG_VEC] "
                f"dw_raw={np.array2string(dbg.get('dw_raw', np.zeros(3)), precision=3)}, "
                f"de_raw={np.array2string(dbg.get('de_raw', np.zeros(3)), precision=3)}, "
                f"dw={np.array2string(dbg.get('dw', np.zeros(3)), precision=3)}, "
                f"de={np.array2string(dbg.get('de', np.zeros(3)), precision=3)}"
            )

            self._dbg_t = time.time()

    def _render_only(self):
        mujoco.mj_forward(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def _try_initialize_refs(self, wrist: np.ndarray, elbow: np.ndarray, pelvis: Optional[np.ndarray], conf: int) -> bool:
        if conf < self.init_min_conf:
            return False

        self._init_wrist_buf.append(wrist.copy())
        self._init_elbow_buf.append(elbow.copy())
        if pelvis is not None:
            self._init_pelvis_buf.append(pelvis.copy())

        n = len(self._init_wrist_buf)
        if n < self.init_avg_frames:
            if n == 1 or n % 5 == 0:
                self.get_logger().info(f"[INIT_BUF] collecting init samples: {n}/{self.init_avg_frames}")
            return False

        wrist_ref = np.mean(np.stack(self._init_wrist_buf, axis=0), axis=0)
        elbow_ref = np.mean(np.stack(self._init_elbow_buf, axis=0), axis=0)
        pelvis_ref = None
        if self.retargeter.cfg.use_pelvis_relative:
            pelvis_ref = np.mean(np.stack(self._init_pelvis_buf, axis=0), axis=0)

        q_home = self.cfg.q.copy()
        self.tasks.initialize_refs(self.cfg, q_home=q_home)
        self.retargeter.set_refs(wrist=wrist_ref, elbow=elbow_ref, pelvis=pelvis_ref)

        # posture regularization: nominal arm pose
        self.arm_q_home = self._build_manual_arm_q_home()
        self.get_logger().info(
        f"[INIT_DONE] manual arm_q_home={np.array2string(self.arm_q_home, precision=3)}"
    )

        self._refs_inited = True

        self.get_logger().info(
            f"[INIT_DONE] averaged over {n} frames, conf>={self.init_min_conf}, "
            f"w_ref={np.array2string(wrist_ref, precision=3)}, "
            f"e_ref={np.array2string(elbow_ref, precision=3)}, "
            f"p_ref={np.array2string(pelvis_ref, precision=3) if pelvis_ref is not None else 'None'}"
        )
        self.get_logger().info(
            f"[INIT_DONE] arm_q_home={np.array2string(self.arm_q_home, precision=3)}"
        )
        return True

    def _build_manual_arm_q_home(self) -> np.ndarray:
        """
        人工指定一个更自然的右臂 nominal pose。
        顺序必须与 allowed joints / arm_qpos_ids 保持一致：
        [right_shoulder_pitch_joint,
        right_shoulder_roll_joint,
        right_shoulder_yaw_joint,
        right_elbow_joint,
        right_wrist_roll_joint,
        right_wrist_pitch_joint,
        right_wrist_yaw_joint]
        """
        q_home = np.array([
            float(self.get_parameter("home_shoulder_pitch").value),
            float(self.get_parameter("home_shoulder_roll").value),
            float(self.get_parameter("home_shoulder_yaw").value),
            float(self.get_parameter("home_elbow").value),
            float(self.get_parameter("home_wrist_roll").value),
            float(self.get_parameter("home_wrist_pitch").value),
            float(self.get_parameter("home_wrist_yaw").value),
        ], dtype=np.float64)

        if q_home.shape[0] != self.arm_qpos_ids.shape[0]:
            self.get_logger().warn(
                f"Manual arm_q_home len={q_home.shape[0]} but arm_qpos_ids len={self.arm_qpos_ids.shape[0]}, "
                "falling back to current cfg.q"
            )
            return self.cfg.q[self.arm_qpos_ids].copy()

        return q_home

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