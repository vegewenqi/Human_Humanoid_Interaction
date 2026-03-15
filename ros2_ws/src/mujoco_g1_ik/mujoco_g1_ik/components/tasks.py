from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from mink import FrameTask, SE3


@dataclass
class TaskConfig:
    ee_site: str = "right_palm"
    elbow_body: str = "right_elbow_link"

    # task weights
    wrist_pos_cost: float = 1.0
    elbow_pos_cost: float = 0.35
    wrist_ori_cost: float = 0.08
    posture_cost: float = 0.15

    task_gain: float = 0.8
    posture_max_vel: float = 0.8

    elbow_avoid_gain: float = 0.8
    elbow_avoid_margin_y: float = 0.18
    elbow_avoid_margin_x: float = 0.02


class TaskSet:
    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        self.ee_ref: Optional[SE3] = None
        self.elbow_ref: Optional[SE3] = None
        self.q_home: Optional[np.ndarray] = None

        self.wrist_task = FrameTask(
            frame_name=self.cfg.ee_site,
            frame_type="site",
            position_cost=self.cfg.wrist_pos_cost,
            orientation_cost=self.cfg.wrist_ori_cost,
            gain=self.cfg.task_gain,
        )

        self.elbow_task = FrameTask(
            frame_name=self.cfg.elbow_body,
            frame_type="body",
            position_cost=self.cfg.elbow_pos_cost,
            orientation_cost=0.0,
            gain=self.cfg.task_gain,
        )

        self._tmp_elbow_avoid_task = None

    def initialize_refs(self, cfg_obj, q_home: np.ndarray):
        self.ee_ref = cfg_obj.get_transform_frame_to_world(self.cfg.ee_site, "site")
        self.elbow_ref = cfg_obj.get_transform_frame_to_world(self.cfg.elbow_body, "body")

        self.wrist_task.set_target(self.ee_ref)
        self.elbow_task.set_target(self.elbow_ref)

        self.q_home = q_home.copy()

    def set_targets_from_deltas(self, dw: np.ndarray, de: np.ndarray):
        assert self.ee_ref is not None and self.elbow_ref is not None
        self.wrist_task.set_target(SE3.from_translation(dw) @ self.ee_ref)
        self.elbow_task.set_target(SE3.from_translation(de) @ self.elbow_ref)

    def build_tasks(self) -> List[FrameTask]:
        tasks = [self.wrist_task, self.elbow_task]
        if hasattr(self, "_tmp_elbow_avoid_task") and self._tmp_elbow_avoid_task is not None:
            tasks.append(self._tmp_elbow_avoid_task)
        return tasks

    def posture_velocity(
        self,
        q: np.ndarray,
        arm_q_home: np.ndarray,
        arm_qpos_ids: np.ndarray,
        arm_dof_ids: np.ndarray,
        nv: int,
    ) -> np.ndarray:
        """
        posture regularization:
            v_bias[dof_i] = -k * (q_i - q_home_i)
        """
        v = np.zeros(nv, dtype=np.float64)
        if arm_q_home is None or arm_qpos_ids.size == 0:
            return v

        q_err = q[arm_qpos_ids] - arm_q_home
        v_arm = -self.cfg.posture_cost * q_err
        v_arm = np.clip(v_arm, -self.cfg.posture_max_vel, self.cfg.posture_max_vel)

        v[arm_dof_ids] = v_arm
        return v
    
    def elbow_avoid_velocity(
        self,
        cfg_obj,
        model,
        elbow_body: str,
        shoulder_body: str = "right_shoulder_pitch_link",
        torso_body: str = "torso_link",
    ) -> np.ndarray:
        """
        生成一个很轻量的软偏置：
        - 如果右肘太靠近身体中线/胸前内侧，就把它往身体右外侧推
        - 输出是 nv 维 velocity bias

        做法：
        1. 读 torso / elbow 世界坐标
        2. 构造一个简单的“希望肘在 torso 右外侧”的位置修正
        3. 用 elbow task 的局部线性化思想近似成一个目标位移 bias
        这里不走完整 Jacobian 显式求逆，而是借用一个临时 FrameTask 更稳妥
        """
        nv = model.nv
        v = np.zeros(nv, dtype=np.float64)

        try:
            elbow_tf = cfg_obj.get_transform_frame_to_world(elbow_body, "body")
            torso_tf = cfg_obj.get_transform_frame_to_world(torso_body, "body")
        except Exception:
            return v

        # 取平移
        elbow_p = elbow_tf.translation()
        torso_p = torso_tf.translation()

        # 当前相对 torso 的位置
        rel = elbow_p - torso_p

        # 目标：
        # 右肘应该在 torso 的右外侧 => rel[1] 应该更负一些
        # 同时别太贴 torso 前中线 => rel[0] 稍微保持在前方或至少别太穿进去
        delta = np.zeros(3, dtype=np.float64)

        # 对右臂，要求 y <= -margin_y
        if rel[1] > -self.cfg.elbow_avoid_margin_y:
            delta[1] = -(rel[1] + self.cfg.elbow_avoid_margin_y)

        # 如果 x 太小，说明肘容易往身体中间/后侧缩，给一点前推
        if rel[0] < self.cfg.elbow_avoid_margin_x:
            delta[0] = (self.cfg.elbow_avoid_margin_x - rel[0])

        if np.linalg.norm(delta) < 1e-6:
            return v

        # 临时构造一个“把肘轻微推开”的 task 目标
        tmp_task = FrameTask(
            frame_name=elbow_body,
            frame_type="body",
            position_cost=self.cfg.elbow_avoid_gain,
            orientation_cost=0.0,
            gain=0.6,
        )
        tmp_task.set_target(SE3.from_translation(delta) @ elbow_tf)

        # 返回 task，交给外面 solve_ik 一起算更合理
        # 这里为了兼容现有结构，直接把 task 存出来
        self._tmp_elbow_avoid_task = tmp_task
        return v