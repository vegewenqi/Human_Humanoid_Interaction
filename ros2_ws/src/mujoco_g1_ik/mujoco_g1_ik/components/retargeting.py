# /mujoco_g1_ik/components/retargeting.py
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from .filtering import EMAJumpFilter


@dataclass
class RetargetingConfig:
    skeleton_unit_scale: float = 0.001   # mm -> m
    motion_gain: float = 0.6
    use_pelvis_relative: bool = True
    max_delta_m: float = 0.25

    # ZED->MuJoCo axis mapping (default guess, works for many setups)
    # ZED: X right, Y up, Z forward
    # MuJoCo: X forward, Y left, Z up
    def map_axes(self, p_m: np.ndarray) -> np.ndarray:
        return np.array([p_m[2], -p_m[0], p_m[1]], dtype=np.float64)


class Retargeter:
    def __init__(self, cfg: RetargetingConfig, ema_alpha: float, max_jump_m: float):
        self.cfg = cfg
        self.filters: Dict[int, EMAJumpFilter] = {}
        self.ema_alpha = ema_alpha
        self.max_jump_m = max_jump_m

        # refs
        self.wrist_ref: Optional[np.ndarray] = None
        self.elbow_ref: Optional[np.ndarray] = None
        self.pelvis_ref: Optional[np.ndarray] = None

        self._last_debug = {}

    def _get_filter(self, idx: int) -> EMAJumpFilter:
        if idx not in self.filters:
            self.filters[idx] = EMAJumpFilter(self.ema_alpha, self.max_jump_m)
        return self.filters[idx]

    def joint_to_mujoco_world(self, pts_xyz: np.ndarray, idx: int) -> Optional[np.ndarray]:
        if idx < 0 or idx >= pts_xyz.shape[0]:
            return None
        p = pts_xyz[idx].astype(np.float64)
        if not np.all(np.isfinite(p)):
            return None
        p_m = p * self.cfg.skeleton_unit_scale
        p_mj = self.cfg.map_axes(p_m)
        p_f = self._get_filter(idx).update(p_mj)
        return p_f

    def set_refs(self, wrist: np.ndarray, elbow: np.ndarray, pelvis: Optional[np.ndarray]):
        self.wrist_ref = wrist.copy()
        self.elbow_ref = elbow.copy()
        if pelvis is not None:
            self.pelvis_ref = pelvis.copy()

    def compute_delta(self, wrist: np.ndarray, elbow: np.ndarray, pelvis: Optional[np.ndarray]):
        """
        Returns:
          delta_wrist (3,), delta_elbow (3,)
        Both are in mujoco-world, scaled by motion_gain and clamped.
        """
        assert self.wrist_ref is not None and self.elbow_ref is not None

        if self.cfg.use_pelvis_relative:
            assert pelvis is not None and self.pelvis_ref is not None
            w_now = wrist - pelvis
            e_now = elbow - pelvis
            w_ref = self.wrist_ref - self.pelvis_ref
            e_ref = self.elbow_ref - self.pelvis_ref
            dw_raw = (w_now - w_ref) * self.cfg.motion_gain
            de_raw = (e_now - e_ref) * self.cfg.motion_gain
        else:
            w_now = wrist.copy()
            e_now = elbow.copy()
            w_ref = self.wrist_ref.copy()
            e_ref = self.elbow_ref.copy()
            dw_raw = (wrist - self.wrist_ref) * self.cfg.motion_gain
            de_raw = (elbow - self.elbow_ref) * self.cfg.motion_gain

        dw = self._clamp(dw_raw, self.cfg.max_delta_m)
        de = self._clamp(de_raw, self.cfg.max_delta_m)

        self._last_debug = {
            "w_now": w_now.copy(),
            "e_now": e_now.copy(),
            "w_ref": w_ref.copy(),
            "e_ref": e_ref.copy(),
            "dw_raw": dw_raw.copy(),
            "de_raw": de_raw.copy(),
            "dw": dw.copy(),
            "de": de.copy(),
            "dw_raw_norm": float(np.linalg.norm(dw_raw)),
            "de_raw_norm": float(np.linalg.norm(de_raw)),
            "dw_norm": float(np.linalg.norm(dw)),
            "de_norm": float(np.linalg.norm(de)),
            "dw_clamped": float(np.linalg.norm(dw_raw)) > self.cfg.max_delta_m,
            "de_clamped": float(np.linalg.norm(de_raw)) > self.cfg.max_delta_m,
        }

        return dw, de

    @staticmethod
    def _clamp(v: np.ndarray, max_norm: float) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > max_norm:
            return v * (max_norm / (n + 1e-9))
        return v