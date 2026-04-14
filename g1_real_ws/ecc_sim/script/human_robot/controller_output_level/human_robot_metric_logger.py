#!/usr/bin/env python3
import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from g1_cbf.kinematics import G1Kinematics


CMD_JOINTS = [
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_elbow_joint',
]

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

DEFAULT_HUMAN_PAIRS = [
    ('left_arm', 'right_upper_arm'),
    ('left_arm', 'right_forearm_hand'),
    ('right_arm', 'left_upper_arm'),
    ('right_arm', 'left_forearm_hand'),
    ('left_upper_arm', 'right_upper_arm'),
    ('left_upper_arm', 'right_forearm_hand'),
    ('right_upper_arm', 'left_upper_arm'),
    ('right_upper_arm', 'left_forearm_hand'),
]

REPRESENTATIVE_PAIRS = [
    ('left_arm', 'right_forearm_hand'),
    ('left_upper_arm', 'right_forearm_hand'),
]


def closest_points_segment_segment(p1, q1, p2, q2):
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

    sN, sD = D, D
    tN, tD = D, D

    if D < SMALL:
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

    sc = 0.0 if abs(sN) < SMALL else sN / sD
    tc = 0.0 if abs(tN) < SMALL else tN / tD

    cp1 = p1 + sc * u
    cp2 = p2 + tc * v
    dist = np.linalg.norm(cp1 - cp2)
    return cp1, cp2, float(dist)


class HumanRobotMetricLogger(Node):
    def __init__(self):
        super().__init__('human_robot_metric_logger')

        self.declare_parameter('urdf_path', '/ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf')
        self.declare_parameter('joint_state_topic', '/sim/joint_states')
        self.declare_parameter('human_capsule_topic', '/sim/human_capsules_robot')
        self.declare_parameter('unsafe_cmd_topic', '/joint_commands_unsafe')
        self.declare_parameter('safe_cmd_topic', '/sim/joint_commands')
        self.declare_parameter('csv_path', '/ws/ecc_sim/result1/human_robot_metrics_no_cbf.csv')
        self.declare_parameter('timer_period_sec', 0.02)
        self.declare_parameter('extra_margin', 0.0)
        self.declare_parameter('record_global_min', True)

        urdf_path = str(self.get_parameter('urdf_path').value)
        if not urdf_path:
            raise RuntimeError('urdf_path parameter is required')

        self.kin = G1Kinematics(urdf_path)
        self.csv_path = str(self.get_parameter('csv_path').value)
        self.extra_margin = float(self.get_parameter('extra_margin').value)
        self.record_global_min = bool(self.get_parameter('record_global_min').value)
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)

        self.q_full: Optional[np.ndarray] = None
        self.human_capsules: Dict[str, Dict[str, np.ndarray]] = {}
        self.unsafe_cmd = {jn: math.nan for jn in CMD_JOINTS}
        self.safe_cmd = {jn: math.nan for jn in CMD_JOINTS}
        self.t0 = None
        self.row_count = 0

        self.fp = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.fp)
        hdr = ['t_sec']
        hdr += [f'unsafe_{jn}' for jn in CMD_JOINTS]
        hdr += [f'safe_{jn}' for jn in CMD_JOINTS]
        if self.record_global_min:
            hdr += ['human_global_min', 'human_global_min_pair']
        for robot_name, human_name in REPRESENTATIVE_PAIRS:
            tag = f'{robot_name}__{human_name}'
            hdr += [
                f'{tag}_signed_distance',
                f'{tag}_robot_px', f'{tag}_robot_py', f'{tag}_robot_pz',
                f'{tag}_human_px', f'{tag}_human_py', f'{tag}_human_pz',
            ]
        self.writer.writerow(hdr)

        self.create_subscription(JointState, str(self.get_parameter('joint_state_topic').value), self._joint_state_cb, 50)
        self.create_subscription(Float32MultiArray, str(self.get_parameter('human_capsule_topic').value), self._human_capsule_cb, 50)
        self.create_subscription(JointState, str(self.get_parameter('unsafe_cmd_topic').value), self._unsafe_cb, 50)
        self.create_subscription(JointState, str(self.get_parameter('safe_cmd_topic').value), self._safe_cb, 50)
        self.create_timer(float(self.get_parameter('timer_period_sec').value), self._tick)

        self.get_logger().info(f'Logging human-robot metrics to {self.csv_path}')

    def _set_t0_now(self):
        now = self.get_clock().now()
        if self.t0 is None:
            self.t0 = now
        return (now - self.t0).nanoseconds * 1e-9

    def _joint_state_cb(self, msg: JointState):
        self.q_full = self.kin.joint_names_to_q_full(list(msg.name), list(msg.position))

    def _unsafe_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        for jn in CMD_JOINTS:
            if jn in name_to_pos:
                self.unsafe_cmd[jn] = float(name_to_pos[jn])

    def _safe_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        for jn in CMD_JOINTS:
            if jn in name_to_pos:
                self.safe_cmd[jn] = float(name_to_pos[jn])

    def _human_capsule_cb(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float64)
        expected = 7 * len(HUMAN_CAPSULE_NAMES)
        if data.size != expected:
            self.get_logger().warn(
                f'Expected {expected} human capsule values, got {data.size}',
                throttle_duration_sec=2.0,
            )
            return
        caps = {}
        for i, name in enumerate(HUMAN_CAPSULE_NAMES):
            block = data[7 * i: 7 * i + 7]
            caps[name] = {
                'a': block[0:3].copy(),
                'b': block[3:6].copy(),
                'radius': float(block[6]),
            }
        self.human_capsules = caps

    def _robot_capsule(self, body_name: str):
        a, b, _, _ = self.kin.get_endpoint_jacobians(body_name)
        body = self.kin.collision_bodies[body_name]
        return {'a': a, 'b': b, 'radius': float(body['radius'])}

    def _signed_distance(self, robot_name: str, human_name: str):
        if human_name not in self.human_capsules:
            return None
        rc = self._robot_capsule(robot_name)
        hc = self.human_capsules[human_name]
        p_r, p_h, d_seg = closest_points_segment_segment(rc['a'], rc['b'], hc['a'], hc['b'])
        h = d_seg - (rc['radius'] + hc['radius'] + self.extra_margin)
        return float(h), p_r, p_h

    def _tick(self):
        if self.q_full is None or not self.human_capsules:
            return
        t_sec = self._set_t0_now()
        self.kin.update(self.q_full)

        row = [f'{t_sec:.9f}']
        row += [f'{self.unsafe_cmd[jn]:.12g}' for jn in CMD_JOINTS]
        row += [f'{self.safe_cmd[jn]:.12g}' for jn in CMD_JOINTS]

        if self.record_global_min:
            best_h = math.inf
            best_pair = ''
            for robot_name, human_name in DEFAULT_HUMAN_PAIRS:
                if human_name not in self.human_capsules or robot_name not in self.kin.collision_bodies:
                    continue
                out = self._signed_distance(robot_name, human_name)
                if out is None:
                    continue
                h, _, _ = out
                if h < best_h:
                    best_h = h
                    best_pair = f'{robot_name}__{human_name}'
            row += [f'{best_h:.12g}' if math.isfinite(best_h) else 'nan', best_pair]

        for robot_name, human_name in REPRESENTATIVE_PAIRS:
            out = self._signed_distance(robot_name, human_name)
            if out is None:
                row += ['nan'] * 7
            else:
                h, p_r, p_h = out
                row += [
                    f'{h:.12g}',
                    f'{p_r[0]:.12g}', f'{p_r[1]:.12g}', f'{p_r[2]:.12g}',
                    f'{p_h[0]:.12g}', f'{p_h[1]:.12g}', f'{p_h[2]:.12g}',
                ]

        self.writer.writerow(row)
        self.row_count += 1
        if self.row_count % 20 == 0:
            self.fp.flush()

    def destroy_node(self):
        try:
            self.fp.flush()
            self.fp.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HumanRobotMetricLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
