#!/usr/bin/env python3
import csv
import math
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

try:
    from g1_cbf.kinematics import G1Kinematics, CONTROLLED_JOINTS
except Exception as e:
    raise RuntimeError(
        'Failed to import g1_cbf.kinematics. Run this inside your ROS2 workspace environment.'
    ) from e


HUMAN_CAPSULE_NAMES = [
    'torso',
    'left_upper_arm',
    'left_forearm_hand',
    'right_upper_arm',
    'right_forearm_hand',
    'left_thigh',
    'right_thigh',
    'left_shin',
    'right_shin',
    'head_sphere',
]

ALL_ROBOT_HUMAN_PAIRS = [
    ('left_arm', 'right_upper_arm'),
    ('left_arm', 'right_forearm_hand'),
    ('right_arm', 'left_upper_arm'),
    ('right_arm', 'left_forearm_hand'),
    ('left_upper_arm', 'right_upper_arm'),
    ('left_upper_arm', 'right_forearm_hand'),
    ('right_upper_arm', 'left_upper_arm'),
    ('right_upper_arm', 'left_forearm_hand'),
]


@dataclass
class Capsule:
    a: np.ndarray
    b: np.ndarray
    radius: float


def segment_segment_distance_with_points(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return distance and closest points between two 3D segments."""
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b
    SMALL = 1e-12

    sN = 0.0
    sD = D
    tN = 0.0
    tD = D

    if D < SMALL:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
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
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < SMALL else sN / sD
    tc = 0.0 if abs(tN) < SMALL else tN / tD

    c1 = p1 + sc * u
    c2 = p2 + tc * v
    dP = c1 - c2
    return float(np.linalg.norm(dP)), c1, c2


class CommandLevelHumanRobotMetricLogger(Node):
    def __init__(self):
        super().__init__('command_level_human_robot_metric_logger')

        self.declare_parameter('urdf_path', '/ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf')
        self.declare_parameter('csv_path', '/ws/ecc_sim/result2/command_level_human_robot_metrics.csv')
        self.declare_parameter('sample_rate_hz', 60.0)
        self.declare_parameter('extra_margin', 0.0)
        self.declare_parameter('capsule_length', 70)
        self.declare_parameter('human_capsule_topic', '/sim/human_capsules_robot')
        self.declare_parameter('unsafe_cmd_topic', '/joint_commands_unsafe')
        self.declare_parameter('safe_cmd_topic', '/sim/joint_commands')
        self.declare_parameter('joint_state_topic', '/sim/joint_states')

        urdf_path = str(self.get_parameter('urdf_path').value)
        if not urdf_path:
            raise RuntimeError('urdf_path parameter is required')

        self.csv_path = str(self.get_parameter('csv_path').value)
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        self.sample_rate_hz = float(self.get_parameter('sample_rate_hz').value)
        self.extra_margin = float(self.get_parameter('extra_margin').value)
        self.capsule_length = int(self.get_parameter('capsule_length').value)

        self.rep_pairs = [
            ('left_arm', 'right_forearm_hand'),
            ('left_upper_arm', 'right_forearm_hand'),
        ]

        self.kin = G1Kinematics(urdf_path)
        self.latest_unsafe: Optional[JointState] = None
        self.latest_safe: Optional[JointState] = None
        self.latest_human: Optional[Dict[str, Capsule]] = None
        self.latest_template_q: Optional[np.ndarray] = None
        self.t0: Optional[float] = None
        self.row_count = 0

        self.create_subscription(
            JointState,
            str(self.get_parameter('unsafe_cmd_topic').value),
            self._unsafe_cb,
            20,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('safe_cmd_topic').value),
            self._safe_cb,
            20,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self._joint_state_cb,
            20,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter('human_capsule_topic').value),
            self._human_cb,
            20,
        )

        self._open_csv()
        self.create_timer(1.0 / self.sample_rate_hz, self._tick)
        self.get_logger().info(f'Writing metrics to {self.csv_path}')

    def _open_csv(self):
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        header = ['t_sec']
        for prefix in ['unsafe', 'safe']:
            for j in CONTROLLED_JOINTS:
                header.append(f'{prefix}_{j}')
        for prefix in ['unsafe', 'safe']:
            header += [
                f'{prefix}_rep1_h',
                f'{prefix}_rep2_h',
                f'{prefix}_global_min_h',
                f'{prefix}_global_min_pair',
            ]
        self.writer.writerow(header)
        self.csv_file.flush()

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()

    def _unsafe_cb(self, msg: JointState):
        self.latest_unsafe = msg

    def _safe_cb(self, msg: JointState):
        self.latest_safe = msg

    def _joint_state_cb(self, msg: JointState):
        try:
            self.latest_template_q = self.kin.joint_names_to_q_full(list(msg.name), list(msg.position))
        except Exception:
            pass

    def _human_cb(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size != self.capsule_length:
            self.get_logger().warn(
                f'Expected human capsule length {self.capsule_length}, got {data.size}',
                throttle_duration_sec=2.0,
            )
            return
        n_caps = data.size // 7
        names = HUMAN_CAPSULE_NAMES[:n_caps]
        caps: Dict[str, Capsule] = {}
        for i, name in enumerate(names):
            block = data[7 * i: 7 * i + 7]
            if not np.all(np.isfinite(block)):
                continue
            caps[name] = Capsule(
                a=block[0:3].copy(),
                b=block[3:6].copy(),
                radius=float(block[6]),
            )
        self.latest_human = caps

    def _tick(self):
        if self.latest_template_q is None:
            return
        if self.latest_unsafe is None or self.latest_safe is None or self.latest_human is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t_sec = now - self.t0

        unsafe_q_full, unsafe_ctrl = self._command_to_qfull_and_ctrl(self.latest_unsafe)
        safe_q_full, safe_ctrl = self._command_to_qfull_and_ctrl(self.latest_safe)
        if unsafe_q_full is None or safe_q_full is None:
            return

        unsafe_rep1, unsafe_rep2, unsafe_gmin, unsafe_gpair = self._compute_metrics(unsafe_q_full)
        safe_rep1, safe_rep2, safe_gmin, safe_gpair = self._compute_metrics(safe_q_full)

        row = [t_sec]
        row += unsafe_ctrl.tolist()
        row += safe_ctrl.tolist()
        row += [unsafe_rep1, unsafe_rep2, unsafe_gmin, unsafe_gpair]
        row += [safe_rep1, safe_rep2, safe_gmin, safe_gpair]
        self.writer.writerow(row)
        self.row_count += 1
        if self.row_count % 30 == 0:
            self.csv_file.flush()

    def _command_to_qfull_and_ctrl(self, msg: JointState) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        template = self.latest_template_q.copy()
        name_to_pos = dict(zip(msg.name, msg.position))
        ctrl = np.zeros(len(CONTROLLED_JOINTS), dtype=np.float64)
        for i, jn in enumerate(CONTROLLED_JOINTS):
            if jn not in name_to_pos:
                self.get_logger().warn(f'Missing controlled joint {jn}', throttle_duration_sec=2.0)
                return None, None
            ctrl[i] = float(name_to_pos[jn])
        try:
            ctrl_q = self.kin.joint_names_to_q_full(list(msg.name), list(msg.position))
        except Exception:
            # Fallback: if helper expects full list, overwrite controlled entries manually.
            ctrl_q = None

        if ctrl_q is not None and len(ctrl_q) == len(template):
            # Use template for non-controlled joints; overwrite controlled entries wherever nonzero helper filled them.
            q_full = template
            nz = np.isfinite(ctrl_q)
            q_full[nz] = ctrl_q[nz]
            return q_full, ctrl

        # Last-resort fallback: try common 8-joint name replacement on template via dict conversion.
        try:
            base_from_template = self.kin.joint_names_to_q_full(list(CONTROLLED_JOINTS), list(ctrl))
            q_full = template
            nz = np.isfinite(base_from_template)
            q_full[nz] = base_from_template[nz]
            return q_full, ctrl
        except Exception:
            return None, None

    def _compute_metrics(self, q_full: np.ndarray) -> Tuple[float, float, float, str]:
        self.kin.update(q_full)

        robot_caps: Dict[str, Capsule] = {}
        for name, body in self.kin.collision_bodies.items():
            a, b, _, _ = self.kin.get_endpoint_jacobians(name)
            robot_caps[name] = Capsule(a=a.copy(), b=b.copy(), radius=float(body['radius']))

        rep_values = []
        global_min = math.inf
        global_pair = ''
        for pair in ALL_ROBOT_HUMAN_PAIRS:
            rname, hname = pair
            if rname not in robot_caps or hname not in self.latest_human:
                val = math.nan
            else:
                val = self._signed_distance(robot_caps[rname], self.latest_human[hname])
                if val < global_min:
                    global_min = val
                    global_pair = f'{rname}__{hname}'
            if pair in self.rep_pairs:
                rep_values.append(val)

        while len(rep_values) < 2:
            rep_values.append(math.nan)
        if not math.isfinite(global_min):
            global_min = math.nan
        return rep_values[0], rep_values[1], global_min, global_pair

    def _signed_distance(self, rc: Capsule, hc: Capsule) -> float:
        d_seg, _, _ = segment_segment_distance_with_points(rc.a, rc.b, hc.a, hc.b)
        return d_seg - (rc.radius + hc.radius + self.extra_margin)


def main(args=None):
    rclpy.init(args=args)
    node = CommandLevelHumanRobotMetricLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
