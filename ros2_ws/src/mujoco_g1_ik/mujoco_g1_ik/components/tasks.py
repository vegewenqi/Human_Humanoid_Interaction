# /mujoco_g1_ik/components/tasks.py
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
    elbow_pos_cost: float = 0.35  # elbow weights, less than wrist but enough to influence posture
    posture_cost: float = 0.02    # posture regularization (the larger, the more "home" it will be)

    task_gain: float = 0.8        # mink task gain (response speed)


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
            orientation_cost=0.0,
            gain=self.cfg.task_gain,
        )

        self.elbow_task = FrameTask(
            frame_name=self.cfg.elbow_body,
            frame_type="body",
            position_cost=self.cfg.elbow_pos_cost,
            orientation_cost=0.0,
            gain=self.cfg.task_gain,
        )

    def initialize_refs(self, cfg_obj, q_home: np.ndarray):
        # cfg_obj is mink.Configuration
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
        return [self.wrist_task, self.elbow_task]

    def posture_damping(self, qvel: np.ndarray) -> np.ndarray:
        """
        Velocity-space damping (nv-dim):  v_bias = -k * qvel
        This stabilizes motion and helps avoid "stuck" postures without nq/nv mismatch.
        """
        return -qvel * self.cfg.posture_cost