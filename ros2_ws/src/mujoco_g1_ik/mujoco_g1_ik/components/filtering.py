# /mujoco_g1_ik/components/filtering.py
from dataclasses import dataclass
from typing import Optional
import numpy as np


# jump gate + EMA
@dataclass
class EMAJumpFilter:
    ema_alpha: float = 0.25
    max_jump_m: float = 0.12

    last: Optional[np.ndarray] = None
    filt: Optional[np.ndarray] = None

    def update(self, p: np.ndarray) -> np.ndarray:
        """p is (3,) in mujoco world meters."""
        if self.last is not None:
            if np.linalg.norm(p - self.last) > self.max_jump_m:
                # reject outlier
                return self.filt if self.filt is not None else self.last

        self.last = p

        if self.filt is None:
            self.filt = p.copy()
        else:
            a = self.ema_alpha
            self.filt = a * p + (1.0 - a) * self.filt
        return self.filt