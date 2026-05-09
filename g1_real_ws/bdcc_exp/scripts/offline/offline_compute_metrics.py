#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    from g1_cbf.kinematics import G1Kinematics, CONTROLLED_JOINTS, COLLISION_PAIRS
except Exception as e:
    raise RuntimeError(
        "Failed to import g1_cbf.kinematics. "
        "Run this inside the ROS2 workspace environment after sourcing setup.bash."
    ) from e


HUMAN_CAPSULE_NAMES = [
    "torso",
    "left_upper_arm",
    "left_forearm_hand",
    "right_upper_arm",
    "right_forearm_hand",
    "left_thigh",
    "right_thigh",
    "left_shin",
    "right_shin",
    "head_sphere",
]


ROBOT_HUMAN_COLLISION_PAIRS = [
    ("left_arm", "right_upper_arm"),
    ("left_arm", "right_forearm_hand"),
    ("right_arm", "left_upper_arm"),
    ("right_arm", "left_forearm_hand"),
    ("left_upper_arm", "right_upper_arm"),
    ("left_upper_arm", "right_forearm_hand"),
    ("right_upper_arm", "left_upper_arm"),
    ("right_upper_arm", "left_forearm_hand"),
    ("left_arm", "torso"),
    ("right_arm", "torso"),
    ("left_upper_arm", "torso"),
    ("right_upper_arm", "torso"),
    ("left_arm", "right_thigh"),
    ("right_arm", "left_thigh"),
]


# Correct BODY_38 indices from your zed_indices.py
ZED = {
    "pelvis": 0,
    "spine_1": 1,
    "spine_2": 2,
    "spine_3": 3,
    "neck": 4,
    "nose": 5,
    "left_clavicle": 10,
    "right_clavicle": 11,
    "left_shoulder": 12,
    "right_shoulder": 13,
    "left_elbow": 14,
    "right_elbow": 15,
    "left_wrist": 16,
    "right_wrist": 17,
    "left_middle_tip": 34,
    "right_middle_tip": 35,
}


def unit(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.shape[0] != 3 or not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def clip_acos_dot(a: np.ndarray, b: np.ndarray) -> float:
    aa = unit(a)
    bb = unit(b)
    if aa is None or bb is None:
        return math.nan
    return float(np.arccos(np.clip(np.dot(aa, bb), -1.0, 1.0)))


def segment_segment_distance(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    p1 = np.asarray(p1, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))

    D = a * c - b * b
    small = 1e-12

    sN = 0.0
    sD = D
    tN = 0.0
    tD = D

    if D < small:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d

        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a

    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < small else sN / sD
    tc = 0.0 if abs(tN) < small else tN / tD

    c1 = p1 + sc * u
    c2 = p2 + tc * v
    return float(np.linalg.norm(c1 - c2)), c1, c2


def as_2d_float_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in arr], axis=0)
    else:
        arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(f"{name} should be 2D, got shape {arr.shape}")
    return arr


def interpolate_matrix(src_t: np.ndarray, src_x: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    src_t = np.asarray(src_t, dtype=np.float64)
    src_x = np.asarray(src_x, dtype=np.float64)
    target_t = np.asarray(target_t, dtype=np.float64)

    if len(src_t) < 2:
        raise ValueError("Need at least 2 source samples for interpolation.")

    out = np.zeros((len(target_t), src_x.shape[1]), dtype=np.float64)
    for j in range(src_x.shape[1]):
        out[:, j] = np.interp(target_t, src_t, src_x[:, j])
    return out


def make_common_time(time_arrays: List[np.ndarray], rate_hz: float) -> np.ndarray:
    starts = [float(np.min(t)) for t in time_arrays if len(t) > 0]
    ends = [float(np.max(t)) for t in time_arrays if len(t) > 0]

    t0 = max(starts)
    t1 = min(ends)

    if t1 <= t0:
        raise ValueError(f"No overlap between time ranges: starts={starts}, ends={ends}")

    dt = 1.0 / max(rate_hz, 1e-6)
    return np.arange(t0, t1 + 0.5 * dt, dt, dtype=np.float64)


def build_q_full_from_controlled(
    kin: G1Kinematics,
    q_ctrl: np.ndarray,
    q_template: np.ndarray,
) -> np.ndarray:
    q_full = q_template.copy()
    q_full[kin.controlled_q_indices] = q_ctrl
    return q_full


def compute_collision_count(unsafe: np.ndarray) -> int:
    unsafe = np.asarray(unsafe, dtype=bool)
    if len(unsafe) == 0:
        return 0
    if len(unsafe) == 1:
        return int(unsafe[0])

    count = 0
    for i in range(1, len(unsafe)):
        if unsafe[i] and not unsafe[i - 1]:
            count += 1
    return int(count)


def safety_summary(prefix: str, global_min: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(global_min, dtype=np.float64)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return {
            f"{prefix}_M_clear_m": math.nan,
            f"{prefix}_M_ctr": math.nan,
            f"{prefix}_M_cc": math.nan,
        }

    arr_v = arr[valid]
    unsafe = arr_v < 0.0

    return {
        f"{prefix}_M_clear_m": float(np.min(arr_v)),
        f"{prefix}_M_ctr": float(np.mean(unsafe)),
        f"{prefix}_M_cc": int(compute_collision_count(unsafe)),
    }


def compute_self_collision_clearances(
    kin: G1Kinematics,
    q_full: np.ndarray,
    pairs: List[Tuple[str, str]],
) -> Tuple[Dict[str, float], float, str]:
    kin.update(q_full)

    pair_values: Dict[str, float] = {}
    global_min = math.inf
    global_pair = ""

    for a_name, b_name in pairs:
        a1, a2, _, _ = kin.get_endpoint_jacobians(a_name)
        b1, b2, _, _ = kin.get_endpoint_jacobians(b_name)

        r_a = float(kin.collision_bodies[a_name]["radius"])
        r_b = float(kin.collision_bodies[b_name]["radius"])

        d_seg, _, _ = segment_segment_distance(a1, a2, b1, b2)
        clearance = d_seg - (r_a + r_b)

        tag = f"{a_name}__{b_name}"
        pair_values[tag] = float(clearance)

        if clearance < global_min:
            global_min = float(clearance)
            global_pair = tag

    return pair_values, global_min, global_pair


def parse_human_capsules(flat: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    flat = np.asarray(flat, dtype=np.float64).reshape(-1)
    n = min(len(HUMAN_CAPSULE_NAMES), flat.size // 7)

    caps = {}
    for i in range(n):
        name = HUMAN_CAPSULE_NAMES[i]
        block = flat[7 * i: 7 * i + 7]
        if block.size != 7:
            continue
        if not np.all(np.isfinite(block)):
            continue
        caps[name] = {
            "a": block[0:3].copy(),
            "b": block[3:6].copy(),
            "radius": float(block[6]),
        }
    return caps


def robot_capsule_from_fk(kin: G1Kinematics, body_name: str) -> Dict[str, np.ndarray]:
    a, b, _, _ = kin.get_endpoint_jacobians(body_name)
    body = kin.collision_bodies[body_name]
    return {
        "a": a,
        "b": b,
        "radius": float(body["radius"]),
    }


def compute_human_robot_clearances(
    kin: G1Kinematics,
    q_full: np.ndarray,
    human_caps_flat: np.ndarray,
    pairs: List[Tuple[str, str]],
) -> Tuple[Dict[str, float], float, str]:
    kin.update(q_full)
    human_caps = parse_human_capsules(human_caps_flat)

    pair_values: Dict[str, float] = {}
    global_min = math.inf
    global_pair = ""

    for robot_name, human_name in pairs:
        tag = f"{robot_name}__{human_name}"

        if robot_name not in kin.collision_bodies or human_name not in human_caps:
            pair_values[tag] = math.nan
            continue

        rc = robot_capsule_from_fk(kin, robot_name)
        hc = human_caps[human_name]

        d_seg, _, _ = segment_segment_distance(rc["a"], rc["b"], hc["a"], hc["b"])
        clearance = d_seg - (float(rc["radius"]) + float(hc["radius"]))

        pair_values[tag] = float(clearance)

        if np.isfinite(clearance) and clearance < global_min:
            global_min = float(clearance)
            global_pair = tag

    if not np.isfinite(global_min):
        global_min = math.nan
        global_pair = ""

    return pair_values, global_min, global_pair


def skeleton_flat_to_points(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float64).reshape(-1)
    if frame.size % 3 != 0:
        raise ValueError(f"Skeleton frame length must be divisible by 3, got {frame.size}")
    return frame.reshape((-1, 3))


def get_point(points: np.ndarray, idx: int) -> Optional[np.ndarray]:
    if idx < 0 or idx >= points.shape[0]:
        return None
    p = points[idx]
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        return None
    return p.copy()


def build_torso_frame_from_points(
    pelvis: np.ndarray,
    neck: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Return R with columns [x, y, z] expressed in the original skeleton frame.
    Local vector = R.T @ global vector.

    z: pelvis -> neck
    y: right_shoulder -> left_shoulder, orthogonalized against z
    x: y x z
    """
    z = unit(neck - pelvis)
    if z is None:
        return None

    y_raw = left_shoulder - right_shoulder
    y_proj = y_raw - float(np.dot(y_raw, z)) * z
    y = unit(y_proj)
    if y is None:
        return None

    x = unit(np.cross(y, z))
    if x is None:
        return None

    # Recompute y for orthonormality.
    y = unit(np.cross(z, x))
    if y is None:
        return None

    R = np.column_stack([x, y, z])
    return R


def human_local_link_dirs_from_skeleton(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns 4x3 local unit vectors in human torso frame:
      [left upper arm, left forearm, right upper arm, right forearm]

    Link directions:
      upper arm: shoulder -> elbow
      forearm: elbow -> middle_tip if valid else wrist
    """
    pts = skeleton_flat_to_points(frame)

    pelvis = get_point(pts, ZED["pelvis"])
    neck = get_point(pts, ZED["neck"])
    l_sh = get_point(pts, ZED["left_shoulder"])
    r_sh = get_point(pts, ZED["right_shoulder"])
    l_el = get_point(pts, ZED["left_elbow"])
    r_el = get_point(pts, ZED["right_elbow"])
    l_wr = get_point(pts, ZED["left_wrist"])
    r_wr = get_point(pts, ZED["right_wrist"])
    l_tip = get_point(pts, ZED["left_middle_tip"])
    r_tip = get_point(pts, ZED["right_middle_tip"])

    if any(p is None for p in [pelvis, neck, l_sh, r_sh, l_el, r_el]):
        return None

    l_hand = l_tip if l_tip is not None else l_wr
    r_hand = r_tip if r_tip is not None else r_wr
    if l_hand is None or r_hand is None:
        return None

    R = build_torso_frame_from_points(pelvis, neck, l_sh, r_sh)
    if R is None:
        return None

    global_dirs = [
        l_el - l_sh,
        l_hand - l_el,
        r_el - r_sh,
        r_hand - r_el,
    ]

    local_dirs = []
    for v in global_dirs:
        vv = unit(R.T @ v)
        if vv is None:
            return None
        local_dirs.append(vv)

    return np.stack(local_dirs, axis=0)


def robot_local_link_dirs_from_fk(
    kin: G1Kinematics,
    q_full: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Returns 4x3 local unit vectors in robot pelvis/body local frame:
      [left upper arm, left forearm, right upper arm, right forearm]

    Current implementation uses collision capsule axes because these are exactly
    the geometry used by the CBF layer. The axes are flipped automatically per
    frame to best match the expected human-link directions later in the cost
    function, so endpoint ordering will not dominate nDTW.
    """
    kin.update(q_full)

    names = ["left_upper_arm", "left_arm", "right_upper_arm", "right_arm"]
    dirs = []

    for name in names:
        a, b, _, _ = kin.get_endpoint_jacobians(name)
        d = unit(a - b)
        if d is None:
            return None
        dirs.append(d)

    return np.stack(dirs, axis=0)


def link_direction_cost_allow_flip(h_dirs: np.ndarray, r_dirs: np.ndarray) -> float:
    """
    c(i,j)=sum acos(h_l dot r_l), but for robot capsule endpoint ambiguity we
    choose min(angle(h,r), angle(h,-r)) per link.

    This is still the same local-link angular mismatch, but avoids arbitrary
    capsule endpoint ordering causing pi-radian penalties.
    """
    if h_dirs is None or r_dirs is None:
        return math.nan
    if h_dirs.shape != (4, 3) or r_dirs.shape != (4, 3):
        return math.nan

    total = 0.0
    for k in range(4):
        c1 = clip_acos_dot(h_dirs[k], r_dirs[k])
        c2 = clip_acos_dot(h_dirs[k], -r_dirs[k])
        c = min(c1, c2)
        if not np.isfinite(c):
            return math.nan
        total += c
    return float(total)


def compute_dtw(cost: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    C = np.asarray(cost, dtype=np.float64)
    n, m = C.shape

    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0

    back = np.zeros((n, m), dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
            arg = int(np.argmin(choices))
            D[i, j] = C[i - 1, j - 1] + choices[arg]
            back[i - 1, j - 1] = arg

    path = []
    i = n - 1
    j = m - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        move = int(back[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
        i = max(i, 0)
        j = max(j, 0)

    path.reverse()
    return float(D[n, m]), path


def choose_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n).round().astype(int)


def compute_global_best_lag(
    h_times: np.ndarray,
    h_dirs: np.ndarray,
    r_times: np.ndarray,
    r_dirs: np.ndarray,
    max_lag_sec: float,
    lag_step_sec: float,
) -> Dict[str, Any]:
    """
    Search a single fixed lag Delta:
      compare h(t) with r(t + Delta)

    Positive Delta means robot sequence is later than human sequence.
    """
    if len(h_times) < 2 or len(r_times) < 2:
        return {
            "global_best_lag_sec": math.nan,
            "global_best_lag_cost_rad": math.nan,
        }

    lags = np.arange(-max_lag_sec, max_lag_sec + 0.5 * lag_step_sec, lag_step_sec)
    best_lag = math.nan
    best_cost = math.inf

    for lag in lags:
        costs = []
        for i, th in enumerate(h_times):
            tr = th + lag
            if tr < r_times[0] or tr > r_times[-1]:
                continue

            j = int(np.searchsorted(r_times, tr, side="left"))
            if j <= 0:
                jj = 0
            elif j >= len(r_times):
                jj = len(r_times) - 1
            else:
                jj = j if abs(r_times[j] - tr) < abs(r_times[j - 1] - tr) else j - 1

            c = link_direction_cost_allow_flip(h_dirs[i], r_dirs[jj])
            if np.isfinite(c):
                costs.append(c)

        if len(costs) < max(5, int(0.2 * len(h_times))):
            continue

        mean_cost = float(np.mean(costs))
        if mean_cost < best_cost:
            best_cost = mean_cost
            best_lag = float(lag)

    return {
        "global_best_lag_sec": best_lag,
        "global_best_lag_cost_rad": best_cost if np.isfinite(best_cost) else math.nan,
    }


def compute_link_ndtw(
    kin: G1Kinematics,
    skeleton_t: np.ndarray,
    skeleton_data: np.ndarray,
    q_t: np.ndarray,
    q_ctrl: np.ndarray,
    q_template: np.ndarray,
    max_dtw_frames: int,
    max_lag_sec: float,
    lag_step_sec: float,
) -> Dict[str, Any]:
    skeleton_t = np.asarray(skeleton_t, dtype=np.float64)
    q_t = np.asarray(q_t, dtype=np.float64)
    skeleton_data = as_2d_float_array(skeleton_data, "skeleton_data")
    q_ctrl = as_2d_float_array(q_ctrl, "q_ctrl")

    h_times = []
    h_dirs_list = []
    for i in range(len(skeleton_t)):
        dirs = human_local_link_dirs_from_skeleton(skeleton_data[i])
        if dirs is not None:
            h_times.append(float(skeleton_t[i]))
            h_dirs_list.append(dirs)

    r_times = []
    r_dirs_list = []
    for j in range(len(q_t)):
        q_full = build_q_full_from_controlled(kin, q_ctrl[j], q_template)
        dirs = robot_local_link_dirs_from_fk(kin, q_full)
        if dirs is not None:
            r_times.append(float(q_t[j]))
            r_dirs_list.append(dirs)

    if len(h_times) < 2 or len(r_times) < 2:
        return {
            "ndtw_link_rad": math.nan,
            "dtw_total_cost": math.nan,
            "dtw_path_length": 0,
            "dtw_lag_mean_sec": math.nan,
            "dtw_lag_median_sec": math.nan,
            "dtw_lag_min_sec": math.nan,
            "dtw_lag_max_sec": math.nan,
            "dtw_lag_abs_mean_sec": math.nan,
            "dtw_best_lag_sec": math.nan,
            "global_best_lag_sec": math.nan,
            "global_best_lag_cost_rad": math.nan,
            "dtw_num_human_frames": len(h_times),
            "dtw_num_robot_frames": len(r_times),
        }

    h_times = np.asarray(h_times, dtype=np.float64)
    r_times = np.asarray(r_times, dtype=np.float64)
    h_dirs = np.stack(h_dirs_list, axis=0)
    r_dirs = np.stack(r_dirs_list, axis=0)

    hi = choose_indices(len(h_times), max_dtw_frames)
    ri = choose_indices(len(r_times), max_dtw_frames)

    h_times_ds = h_times[hi]
    r_times_ds = r_times[ri]
    h_dirs_ds = h_dirs[hi]
    r_dirs_ds = r_dirs[ri]

    n = len(h_times_ds)
    m = len(r_times_ds)

    cost = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            c = link_direction_cost_allow_flip(h_dirs_ds[i], r_dirs_ds[j])
            if not np.isfinite(c):
                c = 1e6
            cost[i, j] = c

    total, path = compute_dtw(cost)
    path_costs = np.asarray([cost[i, j] for i, j in path], dtype=np.float64)
    ndtw = float(np.mean(path_costs)) if len(path_costs) > 0 else math.nan

    lags = np.asarray([r_times_ds[j] - h_times_ds[i] for i, j in path], dtype=np.float64)

    global_lag = compute_global_best_lag(
        h_times=h_times_ds,
        h_dirs=h_dirs_ds,
        r_times=r_times_ds,
        r_dirs=r_dirs_ds,
        max_lag_sec=max_lag_sec,
        lag_step_sec=lag_step_sec,
    )

    return {
        "ndtw_link_rad": ndtw,
        "dtw_total_cost": float(total),
        "dtw_path_length": int(len(path)),
        "dtw_lag_mean_sec": float(np.mean(lags)) if len(lags) > 0 else math.nan,
        "dtw_lag_median_sec": float(np.median(lags)) if len(lags) > 0 else math.nan,
        "dtw_lag_min_sec": float(np.min(lags)) if len(lags) > 0 else math.nan,
        "dtw_lag_max_sec": float(np.max(lags)) if len(lags) > 0 else math.nan,
        "dtw_lag_abs_mean_sec": float(np.mean(np.abs(lags))) if len(lags) > 0 else math.nan,
        "dtw_best_lag_sec": float(np.median(lags)) if len(lags) > 0 else math.nan,
        "global_best_lag_sec": global_lag["global_best_lag_sec"],
        "global_best_lag_cost_rad": global_lag["global_best_lag_cost_rad"],
        "dtw_num_human_frames": int(n),
        "dtw_num_robot_frames": int(m),
    }


def load_topics(run_dir: Path):
    topics_path = run_dir / "topics.npz"
    if not topics_path.exists():
        raise FileNotFoundError(f"Missing topics.npz: {topics_path}")
    return np.load(str(topics_path), allow_pickle=True)


def write_summary_csv(path: Path, summary: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])


def write_timeseries_csv(
    path: Path,
    t: np.ndarray,
    q_nom: np.ndarray,
    q_cbf: np.ndarray,
    correction_norm: np.ndarray,
    series: Dict[str, Any],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["t_sec"]

    for j in CONTROLLED_JOINTS:
        header.append(f"q_nom_{j}")
    for j in CONTROLLED_JOINTS:
        header.append(f"q_cbf_{j}")

    header.append("correction_norm")

    scalar_keys = []
    string_keys = []
    for k, v in series.items():
        if isinstance(v, list) and len(v) == len(t):
            if len(v) > 0 and isinstance(v[0], str):
                string_keys.append(k)
            else:
                scalar_keys.append(k)
        elif isinstance(v, np.ndarray) and v.shape[0] == len(t):
            scalar_keys.append(k)

    header += scalar_keys
    header += string_keys

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(len(t)):
            row = [float(t[i])]
            row += [float(x) for x in q_nom[i]]
            row += [float(x) for x in q_cbf[i]]
            row.append(float(correction_norm[i]))

            for k in scalar_keys:
                val = series[k][i]
                row.append(float(val) if np.isfinite(val) else math.nan)

            for k in string_keys:
                row.append(series[k][i])

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Offline BDCC metrics: self/human clearance + RMSE + local-frame nDTW."
    )

    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--urdf-path", required=True)
    parser.add_argument("--outdir", default=None)

    parser.add_argument(
        "--mode",
        choices=["self_collision", "human_robot", "both"],
        default="both",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--max-dtw-frames", type=int, default=600)

    parser.add_argument("--max-lag-sec", type=float, default=1.0)
    parser.add_argument("--lag-step-sec", type=float, default=0.02)

    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else run_dir
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_topics(run_dir)

    required = ["q_nom_t", "q_nom_data", "q_cbf_t", "q_cbf_data"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"topics.npz missing required keys: {missing}")

    q_nom_t_raw = np.asarray(data["q_nom_t"], dtype=np.float64)
    q_cbf_t_raw = np.asarray(data["q_cbf_t"], dtype=np.float64)
    q_nom_raw = as_2d_float_array(data["q_nom_data"], "q_nom_data")
    q_cbf_raw = as_2d_float_array(data["q_cbf_data"], "q_cbf_data")

    time_arrays = [q_nom_t_raw, q_cbf_t_raw]

    need_hr = args.mode in ["human_robot", "both"]

    if need_hr:
        if "human_capsules_robot_t" not in data.files or "human_capsules_robot_data" not in data.files:
            raise KeyError("human_robot mode requires human_capsules_robot_t and human_capsules_robot_data")
        human_caps_t_raw = np.asarray(data["human_capsules_robot_t"], dtype=np.float64)
        human_caps_raw = as_2d_float_array(data["human_capsules_robot_data"], "human_capsules_robot_data")
        time_arrays.append(human_caps_t_raw)
    else:
        human_caps_t_raw = None
        human_caps_raw = None

    if "skeleton_filtered_t" not in data.files or "skeleton_filtered_data" not in data.files:
        raise KeyError("nDTW requires skeleton_filtered_t and skeleton_filtered_data")

    skeleton_t_raw = np.asarray(data["skeleton_filtered_t"], dtype=np.float64)
    skeleton_raw = as_2d_float_array(data["skeleton_filtered_data"], "skeleton_filtered_data")
    time_arrays.append(skeleton_t_raw)

    t = make_common_time(time_arrays, args.sample_rate_hz)

    q_nom = interpolate_matrix(q_nom_t_raw, q_nom_raw, t)
    q_cbf = interpolate_matrix(q_cbf_t_raw, q_cbf_raw, t)
    skeleton = interpolate_matrix(skeleton_t_raw, skeleton_raw, t)

    if need_hr:
        human_caps = interpolate_matrix(human_caps_t_raw, human_caps_raw, t)
    else:
        human_caps = None

    correction = q_cbf - q_nom
    correction_norm = np.linalg.norm(correction, axis=1)
    rmse_q0 = float(np.sqrt(np.mean(correction ** 2)))

    kin = G1Kinematics(args.urdf_path)
    q_template = kin.q_full.copy()

    series: Dict[str, Any] = {}
    summary: Dict[str, Any] = {
        "num_samples": int(len(t)),
        "sample_rate_hz": float(args.sample_rate_hz),
        "duration_sec": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "rmse_q0_rad": rmse_q0,
        "mean_correction_norm_rad": float(np.mean(correction_norm)),
        "max_correction_norm_rad": float(np.max(correction_norm)),
    }

    if args.mode in ["self_collision", "both"]:
        pairs = list(COLLISION_PAIRS)
        pair_tags = [f"{a}__{b}" for a, b in pairs]

        self_unsafe_global_min = []
        self_safe_global_min = []
        self_unsafe_global_pair = []
        self_safe_global_pair = []

        for tag in pair_tags:
            series[f"self_unsafe_clearance__{tag}"] = []
            series[f"self_safe_clearance__{tag}"] = []

        for i in range(len(t)):
            q_nom_full = build_q_full_from_controlled(kin, q_nom[i], q_template)
            q_cbf_full = build_q_full_from_controlled(kin, q_cbf[i], q_template)

            u_vals, u_min, u_pair = compute_self_collision_clearances(kin, q_nom_full, pairs)
            s_vals, s_min, s_pair = compute_self_collision_clearances(kin, q_cbf_full, pairs)

            self_unsafe_global_min.append(float(u_min))
            self_safe_global_min.append(float(s_min))
            self_unsafe_global_pair.append(u_pair)
            self_safe_global_pair.append(s_pair)

            for tag in pair_tags:
                series[f"self_unsafe_clearance__{tag}"].append(float(u_vals[tag]))
                series[f"self_safe_clearance__{tag}"].append(float(s_vals[tag]))

        self_unsafe_global_min = np.asarray(self_unsafe_global_min, dtype=np.float64)
        self_safe_global_min = np.asarray(self_safe_global_min, dtype=np.float64)

        series["self_unsafe_global_min_clearance"] = self_unsafe_global_min
        series["self_safe_global_min_clearance"] = self_safe_global_min
        series["self_unsafe_global_min_pair"] = self_unsafe_global_pair
        series["self_safe_global_min_pair"] = self_safe_global_pair

        summary.update(safety_summary("self_unsafe", self_unsafe_global_min))
        summary.update(safety_summary("self_safe", self_safe_global_min))
        summary["self_delta_M_clear_m"] = summary["self_safe_M_clear_m"] - summary["self_unsafe_M_clear_m"]
        summary["self_delta_M_ctr"] = summary["self_safe_M_ctr"] - summary["self_unsafe_M_ctr"]
        summary["self_delta_M_cc"] = summary["self_safe_M_cc"] - summary["self_unsafe_M_cc"]

    if args.mode in ["human_robot", "both"]:
        pairs = list(ROBOT_HUMAN_COLLISION_PAIRS)
        pair_tags = [f"{a}__{b}" for a, b in pairs]

        hr_unsafe_global_min = []
        hr_safe_global_min = []
        hr_unsafe_global_pair = []
        hr_safe_global_pair = []

        for tag in pair_tags:
            series[f"hr_unsafe_clearance__{tag}"] = []
            series[f"hr_safe_clearance__{tag}"] = []

        for i in range(len(t)):
            q_nom_full = build_q_full_from_controlled(kin, q_nom[i], q_template)
            q_cbf_full = build_q_full_from_controlled(kin, q_cbf[i], q_template)

            u_vals, u_min, u_pair = compute_human_robot_clearances(
                kin, q_nom_full, human_caps[i], pairs
            )
            s_vals, s_min, s_pair = compute_human_robot_clearances(
                kin, q_cbf_full, human_caps[i], pairs
            )

            hr_unsafe_global_min.append(float(u_min))
            hr_safe_global_min.append(float(s_min))
            hr_unsafe_global_pair.append(u_pair)
            hr_safe_global_pair.append(s_pair)

            for tag in pair_tags:
                series[f"hr_unsafe_clearance__{tag}"].append(float(u_vals.get(tag, math.nan)))
                series[f"hr_safe_clearance__{tag}"].append(float(s_vals.get(tag, math.nan)))

        hr_unsafe_global_min = np.asarray(hr_unsafe_global_min, dtype=np.float64)
        hr_safe_global_min = np.asarray(hr_safe_global_min, dtype=np.float64)

        series["hr_unsafe_global_min_clearance"] = hr_unsafe_global_min
        series["hr_safe_global_min_clearance"] = hr_safe_global_min
        series["hr_unsafe_global_min_pair"] = hr_unsafe_global_pair
        series["hr_safe_global_min_pair"] = hr_safe_global_pair

        summary.update(safety_summary("hr_unsafe", hr_unsafe_global_min))
        summary.update(safety_summary("hr_safe", hr_safe_global_min))
        summary["hr_delta_M_clear_m"] = summary["hr_safe_M_clear_m"] - summary["hr_unsafe_M_clear_m"]
        summary["hr_delta_M_ctr"] = summary["hr_safe_M_ctr"] - summary["hr_unsafe_M_ctr"]
        summary["hr_delta_M_cc"] = summary["hr_safe_M_cc"] - summary["hr_unsafe_M_cc"]

    unsafe_dtw = compute_link_ndtw(
        kin=kin,
        skeleton_t=t,
        skeleton_data=skeleton,
        q_t=t,
        q_ctrl=q_nom,
        q_template=q_template,
        max_dtw_frames=args.max_dtw_frames,
        max_lag_sec=args.max_lag_sec,
        lag_step_sec=args.lag_step_sec,
    )

    safe_dtw = compute_link_ndtw(
        kin=kin,
        skeleton_t=t,
        skeleton_data=skeleton,
        q_t=t,
        q_ctrl=q_cbf,
        q_template=q_template,
        max_dtw_frames=args.max_dtw_frames,
        max_lag_sec=args.max_lag_sec,
        lag_step_sec=args.lag_step_sec,
    )

    for k, v in unsafe_dtw.items():
        summary[f"unsafe_{k}"] = v
    for k, v in safe_dtw.items():
        summary[f"safe_{k}"] = v

    summary["zed_indices_used"] = ZED
    summary["ndtw_link_definition"] = (
        "human local torso-frame link directions vs robot local pelvis-frame/capsule-axis "
        "directions for Lu,Lf,Ru,Rf; cost=sum acos(dot); capsule axes use sign-invariant "
        "min(angle(v), angle(-v)) to remove endpoint-order ambiguity."
    )

    timeseries_path = outdir / "metrics_timeseries.csv"
    summary_json_path = outdir / "metrics_summary.json"
    summary_csv_path = outdir / "metrics_summary.csv"

    write_timeseries_csv(
        path=timeseries_path,
        t=t,
        q_nom=q_nom,
        q_cbf=q_cbf,
        correction_norm=correction_norm,
        series=series,
    )

    summary_json_path.write_text(json.dumps(summary, indent=2))
    write_summary_csv(summary_csv_path, summary)

    print("========== Offline metrics complete ==========")
    print(f"Run dir:  {run_dir}")
    print(f"Out dir:  {outdir}")
    print(f"Mode:     {args.mode}")
    print(f"Samples:  {len(t)}")
    print(f"RMSE_q0:  {summary['rmse_q0_rad']:.6f} rad")
    print("")

    if args.mode in ["self_collision", "both"]:
        print("Self-collision:")
        print(f"  unsafe M_clear = {summary['self_unsafe_M_clear_m']:.6f} m")
        print(f"  safe   M_clear = {summary['self_safe_M_clear_m']:.6f} m")
        print(f"  unsafe M_ctr   = {summary['self_unsafe_M_ctr']:.6f}")
        print(f"  safe   M_ctr   = {summary['self_safe_M_ctr']:.6f}")
        print(f"  unsafe M_cc    = {summary['self_unsafe_M_cc']}")
        print(f"  safe   M_cc    = {summary['self_safe_M_cc']}")
        print("")

    if args.mode in ["human_robot", "both"]:
        print("Human-robot:")
        print(f"  unsafe M_clear = {summary['hr_unsafe_M_clear_m']:.6f} m")
        print(f"  safe   M_clear = {summary['hr_safe_M_clear_m']:.6f} m")
        print(f"  unsafe M_ctr   = {summary['hr_unsafe_M_ctr']:.6f}")
        print(f"  safe   M_ctr   = {summary['hr_safe_M_ctr']:.6f}")
        print(f"  unsafe M_cc    = {summary['hr_unsafe_M_cc']}")
        print(f"  safe   M_cc    = {summary['hr_safe_M_cc']}")
        print("")

    print("nDTW-link:")
    print(f"  unsafe_ndtw_link_rad       = {summary['unsafe_ndtw_link_rad']:.6f}")
    print(f"  safe_ndtw_link_rad         = {summary['safe_ndtw_link_rad']:.6f}")
    print(f"  unsafe_dtw_best_lag_sec    = {summary['unsafe_dtw_best_lag_sec']:.6f}")
    print(f"  safe_dtw_best_lag_sec      = {summary['safe_dtw_best_lag_sec']:.6f}")
    print(f"  unsafe_global_best_lag_sec = {summary['unsafe_global_best_lag_sec']:.6f}")
    print(f"  safe_global_best_lag_sec   = {summary['safe_global_best_lag_sec']:.6f}")
    print("")

    print("Files written:")
    print(f"  {timeseries_path}")
    print(f"  {summary_json_path}")
    print(f"  {summary_csv_path}")
    print("==============================================")


if __name__ == "__main__":
    main()