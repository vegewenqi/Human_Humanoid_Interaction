import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class PointFilterState:
    value: Optional[np.ndarray] = None


class EMAJumpFilter:
    def __init__(self, alpha: float = 0.25, max_jump: float = 0.15):
        self.alpha = float(alpha)
        self.max_jump = float(max_jump)
        self.state = PointFilterState()

    def reset(self):
        self.state = PointFilterState()

    def update(self, x: np.ndarray) -> Optional[np.ndarray]:
        if x is None:
            return self.state.value

        x = np.asarray(x, dtype=np.float64)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            return self.state.value

        if self.state.value is None:
            self.state.value = x.copy()
            return self.state.value

        jump = np.linalg.norm(x - self.state.value)
        if jump > self.max_jump:
            return self.state.value

        self.state.value = self.alpha * x + (1.0 - self.alpha) * self.state.value
        return self.state.value


class HumanPosePreprocessor:
    def __init__(self, alpha: float = 0.25, max_jump: float = 0.15):
        self.alpha = float(alpha)
        self.max_jump = float(max_jump)
        self.filters: Dict[int, EMAJumpFilter] = {}

    def _get_filter(self, idx: int) -> EMAJumpFilter:
        if idx not in self.filters:
            self.filters[idx] = EMAJumpFilter(alpha=self.alpha, max_jump=self.max_jump)
        return self.filters[idx]

    def filter_point(self, pts_xyz: np.ndarray, idx: int) -> Optional[np.ndarray]:
        if pts_xyz is None:
            return None
        if idx < 0 or idx >= pts_xyz.shape[0]:
            return None

        p = np.asarray(pts_xyz[idx], dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return None

        return self._get_filter(idx).update(p)

    def extract_points(self, pts_xyz: np.ndarray, index_map: dict) -> Dict[str, Optional[np.ndarray]]:
        out = {}
        for name, idx in index_map.items():
            out[name] = self.filter_point(pts_xyz, idx)
        return out