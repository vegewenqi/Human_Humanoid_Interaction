from typing import Optional, Tuple

import numpy as np


EPS = 1e-8


def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < EPS or not np.isfinite(n):
        return None
    return v / n


def safe_acos(x: float) -> float:
    return float(np.arccos(np.clip(x, -1.0, 1.0)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    a = normalize(v1)
    b = normalize(v2)
    if a is None or b is None:
        return None
    return safe_acos(float(np.dot(a, b)))


def build_torso_frame(
    pelvis: np.ndarray,
    l_shoulder: np.ndarray,
    r_shoulder: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Return R_ZT = [e_tx e_ty e_tz], columns expressed in Z frame.
    x_T: torso right
    y_T: torso forward
    z_T: torso up
    """
    shoulder_center = 0.5 * (l_shoulder + r_shoulder)

    s = shoulder_center - pelvis
    e_tz = normalize(s)
    if e_tz is None:
        return None

    r = r_shoulder - l_shoulder
    r_proj = r - np.dot(r, e_tz) * e_tz
    e_tx = normalize(r_proj)
    if e_tx is None:
        return None

    e_ty = normalize(np.cross(e_tz, e_tx))
    if e_ty is None:
        return None

    R_ZT = np.column_stack([e_tx, e_ty, e_tz])
    return R_ZT


def express_in_torso(R_ZT: np.ndarray, v_z: np.ndarray) -> np.ndarray:
    """
    Convert vector from Z frame to torso frame.
    """
    return R_ZT.T @ v_z