import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

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
        required = [
            "pelvis",
            "l_shoulder",
            "r_shoulder",
            "l_elbow",
            "r_elbow",
            "l_wrist",
            "r_wrist",
        ]
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

        # torso frame
        R_ZT = build_torso_frame(pelvis, l_sh, r_sh)
        if R_ZT is None:
            return None

        e_tx = R_ZT[:, 0]  # torso right
        e_ty = R_ZT[:, 1]  # torso forward
        e_tz = R_ZT[:, 2]  # torso up

        # torso angles: unified formula style
        torso_roll = self._estimate_torso_roll_from_torso_axes(e_tx, e_ty, e_tz)
        torso_pitch = self._estimate_torso_pitch_from_torso_axes(e_tx, e_ty, e_tz)

        # arm segment vectors in global frame
        u_l = l_el - l_sh
        u_r = r_el - r_sh
        f_l = l_wr - l_el
        f_r = r_wr - r_el

        # express upper-arm vectors in torso frame
        u_l_t = express_in_torso(R_ZT, u_l)
        u_r_t = express_in_torso(R_ZT, u_r)

        # shoulder / elbow angles
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

    def _estimate_torso_roll_from_torso_axes(
        self,
        e_tx: np.ndarray,
        e_ty: np.ndarray,
        e_tz: np.ndarray,
    ) -> float:
        """
        Torso roll from torso local axes and global vertical.

        We use:
            roll = atan2( e_ty^T (k x e_tz), k^T e_tz )

        If the sign is opposite to your desired convention
        (e.g. right shoulder higher should be positive),
        flip the sign of `num`.
        """
        k = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        num = -float(np.dot(e_ty, np.cross(k, e_tz)))
        den = float(np.dot(k, e_tz))

        roll = float(np.arctan2(num, den))

        # keep roll in small-angle branch around upright
        if roll < -np.pi / 2:
            roll += np.pi
        elif roll > np.pi / 2:
            roll -= np.pi

        return roll

    def _estimate_torso_pitch_from_torso_axes(
        self,
        e_tx: np.ndarray,
        e_ty: np.ndarray,
        e_tz: np.ndarray,
    ) -> float:
        """
        Torso pitch from torso local axes and global vertical.

        We use:
            pitch = atan2( - e_tx^T (k x e_tz), k^T e_tz )

        Sign convention:
            forward lean  -> positive
            backward lean -> negative
        """
        k = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        num = -float(np.dot(e_tx, np.cross(k, e_tz)))
        den = float(np.dot(k, e_tz))

        pitch = float(np.arctan2(num, den))

        # choose the equivalent branch closest to zero
        candidates = [pitch, pitch + np.pi, pitch - np.pi]
        pitch = min(candidates, key=lambda x: abs(x))

        return pitch

    def _estimate_left_shoulder_roll(self, u_l_t: np.ndarray) -> float:
        """
        Left shoulder roll:
        hanging down -> ~0
        side raise   -> increases toward +90 deg
        """
        ux, uy, uz = u_l_t
        return float(np.arctan2(-ux, -uz))

    def _estimate_right_shoulder_roll(self, u_r_t: np.ndarray) -> float:
        """
        Right shoulder roll:
        hanging down -> ~0
        side raise   -> increases toward +90 deg
        """
        ux, uy, uz = u_r_t
        return float(np.arctan2(ux, -uz))

    def _estimate_elbow_pitch(self, u: np.ndarray, f: np.ndarray) -> Optional[float]:
        """
        Flexion-like convention:
        fully extended -> near 0
        bending        -> larger angle
        """
        ang = angle_between(u, f)
        if ang is None:
            return None
        return float(ang)