#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g1_cbf.kinematics import G1Kinematics


CONTROLLED_JOINTS = [
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
]

SELF_PAIRS = [
    ("left_arm", "right_arm"),
    ("left_arm", "torso"),
    ("right_arm", "torso"),
    ("left_arm", "left_thigh"),
    ("right_arm", "right_thigh"),
]


def segment_segment_distance(p1, q1, p2, q2):
    p1 = np.asarray(p1, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q2 = np.asarray(q2, dtype=float)

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

    sN, sD = D, D
    tN, tD = D, D

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
    dist = float(np.linalg.norm(c1 - c2))
    return dist, c1, c2


class RealtimeSelfCollisionRecorder(Node):
    def __init__(self):
        super().__init__("realtime_self_collision_recorder")

        self.declare_parameter("urdf_path", "/ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf")
        self.declare_parameter("csv_path", "/ws/ecc_sim/data/realtime_self_collision.csv")
        self.declare_parameter("joint_state_topic", "/sim/joint_states")
        self.declare_parameter("unsafe_cmd_topic", "/joint_commands_unsafe")
        self.declare_parameter("safe_cmd_topic", "/sim/joint_commands")
        self.declare_parameter("log_rate_hz", 60.0)
        self.declare_parameter("extra_margin", 0.0)

        urdf_path = str(self.get_parameter("urdf_path").value)
        if not urdf_path:
            raise RuntimeError("urdf_path is required")

        self.csv_path = Path(str(self.get_parameter("csv_path").value))
        self.extra_margin = float(self.get_parameter("extra_margin").value)

        self.kin = G1Kinematics(urdf_path)
        self.q_full_template: Optional[np.ndarray] = None
        self.latest_unsafe: Optional[Dict[str, float]] = None
        self.latest_safe: Optional[Dict[str, float]] = None
        self.t0 = self.get_clock().now()

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_states,
            20,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("unsafe_cmd_topic").value),
            self._on_unsafe,
            50,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("safe_cmd_topic").value),
            self._on_safe,
            50,
        )

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.csv_path.open("w", newline="")
        self.writer = csv.writer(self.f)
        self.writer.writerow(self._header())
        self.f.flush()

        rate = float(self.get_parameter("log_rate_hz").value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"Recording realtime self-collision data to {self.csv_path}")

    def destroy_node(self):
        try:
            self.f.close()
        except Exception:
            pass
        return super().destroy_node()

    def _header(self):
        cols = ["t_sec"]
        for j in CONTROLLED_JOINTS:
            cols.append(f"unsafe_{j}")
        for j in CONTROLLED_JOINTS:
            cols.append(f"safe_{j}")
        for a, b in SELF_PAIRS:
            cols.append(f"unsafe_pair__{a}__{b}")
        cols += ["unsafe_global_min_h", "unsafe_global_min_pair"]
        for a, b in SELF_PAIRS:
            cols.append(f"safe_pair__{a}__{b}")
        cols += ["safe_global_min_h", "safe_global_min_pair"]
        return cols

    def _on_joint_states(self, msg: JointState):
        self.q_full_template = self.kin.joint_names_to_q_full(list(msg.name), list(msg.position))

    def _on_unsafe(self, msg: JointState):
        self.latest_unsafe = dict(zip(msg.name, msg.position))

    def _on_safe(self, msg: JointState):
        self.latest_safe = dict(zip(msg.name, msg.position))

    def _cmd_to_qfull(self, cmd: Dict[str, float]):
        if self.q_full_template is None:
            return None
        for j in CONTROLLED_JOINTS:
            if j not in cmd:
                return None
        q_full = self.q_full_template.copy()
        q_ctrl = np.array([cmd[j] for j in CONTROLLED_JOINTS], dtype=float)
        q_full[self.kin.controlled_q_indices] = q_ctrl
        return q_full

    def _self_metrics(self, q_full):
        self.kin.update(q_full)
        vals = {}
        gmin = math.inf
        gpair = ""
        for a, b in SELF_PAIRS:
            a1, a2, _, _ = self.kin.get_endpoint_jacobians(a)
            b1, b2, _, _ = self.kin.get_endpoint_jacobians(b)
            ra = float(self.kin.collision_bodies[a]["radius"])
            rb = float(self.kin.collision_bodies[b]["radius"])
            d_seg, _, _ = segment_segment_distance(a1, a2, b1, b2)
            h = d_seg - (ra + rb + self.extra_margin)
            vals[(a, b)] = h
            if h < gmin:
                gmin = h
                gpair = f"{a}__{b}"
        return vals, gmin, gpair

    def _tick(self):
        if self.q_full_template is None or self.latest_unsafe is None or self.latest_safe is None:
            return

        q_unsafe = self._cmd_to_qfull(self.latest_unsafe)
        q_safe = self._cmd_to_qfull(self.latest_safe)
        if q_unsafe is None or q_safe is None:
            return

        unsafe_vals, unsafe_min, unsafe_pair = self._self_metrics(q_unsafe)
        safe_vals, safe_min, safe_pair = self._self_metrics(q_safe)

        t_sec = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        row = [t_sec]
        row += [self.latest_unsafe[j] for j in CONTROLLED_JOINTS]
        row += [self.latest_safe[j] for j in CONTROLLED_JOINTS]
        for p in SELF_PAIRS:
            row.append(unsafe_vals[p])
        row += [unsafe_min, unsafe_pair]
        for p in SELF_PAIRS:
            row.append(safe_vals[p])
        row += [safe_min, safe_pair]
        self.writer.writerow(row)
        self.f.flush()


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeSelfCollisionRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
