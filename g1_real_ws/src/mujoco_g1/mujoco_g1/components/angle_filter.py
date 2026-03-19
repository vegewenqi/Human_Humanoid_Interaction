from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class AngleLimits:
    min_val: float
    max_val: float


class ScalarEMA:
    def __init__(self, alpha: float = 0.25):
        self.alpha = float(alpha)
        self.value = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return float(self.value)


class AngleFilter:
    def __init__(self, alpha: float = 0.25, max_rate_deg: float = 180.0, dt: float = 1.0 / 30.0):
        self.alpha = float(alpha)
        self.max_rate = np.deg2rad(max_rate_deg)
        self.dt = float(dt)
        self.ema: Dict[str, ScalarEMA] = {}
        self.prev: Dict[str, float] = {}

        self.limits = {
            "torso_roll": AngleLimits(np.deg2rad(-60.0), np.deg2rad(60.0)),
            "torso_pitch": AngleLimits(np.deg2rad(-60.0), np.deg2rad(60.0)),
            "l_sh_roll": AngleLimits(np.deg2rad(0.0), np.deg2rad(140.0)),
            "r_sh_roll": AngleLimits(np.deg2rad(0.0), np.deg2rad(140.0)),
            "l_el_pitch": AngleLimits(np.deg2rad(0.0), np.deg2rad(180.0)),
            "r_el_pitch": AngleLimits(np.deg2rad(0.0), np.deg2rad(180.0)),
        }

    def _ema(self, name: str, x: float) -> float:
        if name not in self.ema:
            self.ema[name] = ScalarEMA(alpha=self.alpha)
        return self.ema[name].update(x)

    def _rate_limit(self, name: str, x: float) -> float:
        if name not in self.prev:
            self.prev[name] = x
            return x

        max_step = self.max_rate * self.dt
        dx = x - self.prev[name]
        dx = np.clip(dx, -max_step, max_step)
        y = self.prev[name] + dx
        self.prev[name] = float(y)
        return float(y)

    def _clip(self, name: str, x: float) -> float:
        lim = self.limits[name]
        return float(np.clip(x, lim.min_val, lim.max_val))

    def update_dict(self, angles: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for k, v in angles.items():
            y = self._ema(k, v)
            y = self._rate_limit(k, y)
            y = self._clip(k, y)
            out[k] = float(y)
        return out