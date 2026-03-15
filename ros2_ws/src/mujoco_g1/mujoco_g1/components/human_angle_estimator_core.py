from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .human_math import build_torso_frame, express_in_torso, angle_between


@dataclass
class HumanUpperBodyAngles:
    torso_roll: float
    torso_pitch: float
    l_sh_roll: float
    l_el_pitch: float
    r_sh_roll: float
    r_el_pitch: float


class HumanAngleEstimatorCore:
    def estimate(self, pts: Dict[str, np.ndarray]) -> Optional[HumanUpperBodyAngles]:
        required = ["pelvis", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]
        for k in required:
            if pts.get(k) is None:
                return None

        pelvis = pts["pelvis"]
        l_sh = pts["l_shoulder"]
        r_sh = pts["r_shoulder"]
        l_el = pts["l_elbow"]
        r_el = pts["r_elbow"]
        l_wr = pts["l_wrist"]
        r_wr = pts["r_wrist"]

        R_ZT = build_torso_frame(pelvis, l_sh, r_sh)
        if R_ZT is None:
            return None

        e_tx = R_ZT[:, 0]
        e_tz = R_ZT[:, 2]

        torso_roll = self._estimate_torso_roll(e_tx)
        torso_pitch = self._estimate_torso_pitch(e_tz, torso_roll)

        u_l = l_el - l_sh
        u_r = r_el - r_sh
        f_l = l_wr - l_el
        f_r = r_wr - r_el

        u_l_t = express_in_torso(R_ZT, u_l)
        u_r_t = express_in_torso(R_ZT, u_r)

        l_sh_roll = self._estimate_left_shoulder_roll(u_l_t)
        r_sh_roll = self._estimate_right_shoulder_roll(u_r_t)

        l_el_pitch = self._estimate_elbow_pitch(u_l, f_l)
        r_el_pitch = self._estimate_elbow_pitch(u_r, f_r)

        if None in [torso_roll, torso_pitch, l_sh_roll, r_sh_roll, l_el_pitch, r_el_pitch]:
            return None

        return HumanUpperBodyAngles(
            torso_roll=float(torso_roll),
            torso_pitch=float(torso_pitch),
            l_sh_roll=float(l_sh_roll),
            l_el_pitch=float(l_el_pitch),
            r_sh_roll=float(r_sh_roll),
            r_el_pitch=float(r_el_pitch),
        )

    def _estimate_torso_roll(self, e_tx: np.ndarray) -> float:
        ex, ey, ez = e_tx
        return float(np.arctan2(ez, np.sqrt(ex * ex + ey * ey)))

    def _estimate_torso_pitch(self, e_tz: np.ndarray, torso_roll: float) -> float:
        c = np.cos(-torso_roll)
        s = np.sin(-torso_roll)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s,  c],
        ], dtype=np.float64)

        e_deroll = Rx @ e_tz
        return float(np.arctan2(e_deroll[0], e_deroll[2]))

    def _estimate_left_shoulder_roll(self, u_l_t: np.ndarray) -> float:
        ux, uy, uz = u_l_t
        return float(np.arctan2(ux, -uz))

    def _estimate_right_shoulder_roll(self, u_r_t: np.ndarray) -> float:
        ux, uy, uz = u_r_t
        return float(np.arctan2(-ux, -uz))

    def _estimate_elbow_pitch(self, u: np.ndarray, f: np.ndarray) -> Optional[float]:
        # flexion-like form: 0 near fully extended, increases when bending
        ang = angle_between(u, f)
        if ang is None:
            return None
        return float(np.pi - ang)