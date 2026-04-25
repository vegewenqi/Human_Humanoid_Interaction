#!/usr/bin/env python3
import csv
import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray
from unitree_hg.msg import LowState
from scipy.spatial.transform import Rotation

from g1_cbf.calibration_kinematics import CalibrationKinematics


G1_FULL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class AutoCalibrationSampler(Node):
    def __init__(self):
        super().__init__('auto_calibration_sampler')

        self.declare_parameter('urdf_path', '/ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf')
        self.declare_parameter('tag_topic', '/tag_center_zed_world')
        self.declare_parameter('lowstate_topic', '/lowstate')
        self.declare_parameter('qdes_topic', '/calib_upperbody_q_des')

        self.declare_parameter('csv_path', '/ws/calibration/g1_tag_calibration_samples.csv')
        self.declare_parameter('result_path', '/ws/calibration/g1_tag_extrinsic_result.json')

        self.declare_parameter('tag_frame', 'torso_link')
        self.declare_parameter('tag_offset_x', 0.08)
        self.declare_parameter('tag_offset_y', 0.0)
        self.declare_parameter('tag_offset_z', 0.125)

        self.declare_parameter('control_dt', 0.02)
        self.declare_parameter('settle_sec', 2.5)
        self.declare_parameter('sample_sec', 0.8)
        self.declare_parameter('roll_amp', 0.08)
        self.declare_parameter('pitch_amp', 0.08)

        urdf_path = self.get_parameter('urdf_path').value
        if not urdf_path:
            raise RuntimeError('urdf_path parameter is required')

        tag_offset = np.array([
            float(self.get_parameter('tag_offset_x').value),
            float(self.get_parameter('tag_offset_y').value),
            float(self.get_parameter('tag_offset_z').value),
        ])

        self.kin = CalibrationKinematics(
            urdf_path=urdf_path,
            tag_frame=self.get_parameter('tag_frame').value,
            tag_offset_xyz=tag_offset,
        )

        self.csv_path = self.get_parameter('csv_path').value
        self.result_path = self.get_parameter('result_path').value
        self.control_dt = float(self.get_parameter('control_dt').value)
        self.settle_sec = float(self.get_parameter('settle_sec').value)
        self.sample_sec = float(self.get_parameter('sample_sec').value)

        roll_amp = float(self.get_parameter('roll_amp').value)
        pitch_amp = float(self.get_parameter('pitch_amp').value)

        self.pose_offsets = [
            (0.0, 0.0),
            (+roll_amp, 0.0),
            (-roll_amp, 0.0),
            (0.0, +pitch_amp),
            (0.0, -pitch_amp),
            (+roll_amp, +0.75 * pitch_amp),
            (+roll_amp, -0.75 * pitch_amp),
            (-roll_amp, +0.75 * pitch_amp),
            (-roll_amp, -0.75 * pitch_amp),
        ]

        self.latest_tag_zed = None
        self.latest_q_full = None
        self.latest_q8 = None
        self.base_q8 = None

        self.pose_idx = -1
        self.phase = 'WAIT'
        self.phase_start = None
        self.sample_tag_pts = []
        self.sample_robot_pts = []
        self.samples = []

        self.q_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter('qdes_topic').value,
            10,
        )

        self.create_subscription(
            PointStamped,
            self.get_parameter('tag_topic').value,
            self._tag_cb,
            10,
        )

        self.create_subscription(
            LowState,
            self.get_parameter('lowstate_topic').value,
            self._lowstate_cb,
            10,
        )

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id',
                'zed_x', 'zed_y', 'zed_z',
                'pelvis_x', 'pelvis_y', 'pelvis_z'
            ])

        self.create_timer(self.control_dt, self._tick)

        self.get_logger().info('Auto calibration sampler started.')
        self.get_logger().info(f'CSV: {self.csv_path}')
        self.get_logger().info(f'Result: {self.result_path}')
        self.get_logger().info(f'Number of poses: {len(self.pose_offsets)}')

    def _tag_cb(self, msg):
        self.latest_tag_zed = np.array([
            msg.point.x,
            msg.point.y,
            msg.point.z,
        ], dtype=np.float64)

    def _lowstate_cb(self, msg):
        if len(msg.motor_state) < 29:
            self.get_logger().warn(
                f'/lowstate motor_state length={len(msg.motor_state)} < 29',
                throttle_duration_sec=2.0,
            )
            return

        positions = [float(msg.motor_state[i].q) for i in range(29)]
        self.latest_q_full = self.kin.joint_names_to_q_full(
            G1_FULL_JOINT_NAMES,
            positions,
        )

        # q8 layout expected:
        # [wr, wp, lsp, lsr, le, rsp, rsr, re]
        self.latest_q8 = np.array([
            positions[13],  # waist_roll_joint
            positions[14],  # waist_pitch_joint
            positions[15],  # left_shoulder_pitch_joint
            positions[16],  # left_shoulder_roll_joint
            positions[18],  # left_elbow_joint
            positions[22],  # right_shoulder_pitch_joint
            positions[23],  # right_shoulder_roll_joint
            positions[25],  # right_elbow_joint
        ], dtype=np.float64)

    def _current_target_q8(self):
        q = self.base_q8.copy()
        if 0 <= self.pose_idx < len(self.pose_offsets):
            d_roll, d_pitch = self.pose_offsets[self.pose_idx]
            q[0] = self.base_q8[0] + d_roll
            q[1] = self.base_q8[1] + d_pitch
        return q

    def _publish_target(self):
        if self.base_q8 is None:
            return
        msg = Float32MultiArray()
        msg.data = self._current_target_q8().astype(np.float32).tolist()
        self.q_pub.publish(msg)

    def _tick(self):
        now = time.perf_counter()

        if self.latest_tag_zed is None or self.latest_q_full is None or self.latest_q8 is None:
            self.get_logger().info(
                'Waiting for /tag_center_zed_world and /lowstate...',
                throttle_duration_sec=2.0,
            )
            return

        if self.base_q8 is None:
            self.base_q8 = self.latest_q8.copy()
            self.get_logger().info(
                f'Base q8 captured: {np.round(self.base_q8, 4)}'
            )
            self.pose_idx = 0
            self.phase = 'SETTLE'
            self.phase_start = now
            self.get_logger().info(
                f'Pose {self.pose_idx}: offset={self.pose_offsets[self.pose_idx]}, settling...'
            )

        self._publish_target()

        if self.phase == 'SETTLE':
            if now - self.phase_start >= self.settle_sec:
                self.sample_tag_pts = []
                self.sample_robot_pts = []
                self.phase = 'SAMPLE'
                self.phase_start = now
                self.get_logger().info(f'Pose {self.pose_idx}: sampling...')

        elif self.phase == 'SAMPLE':
            self.kin.update(self.latest_q_full)
            p_robot, _ = self.kin.get_tag_pose()

            self.sample_tag_pts.append(self.latest_tag_zed.copy())
            self.sample_robot_pts.append(p_robot.copy())

            if now - self.phase_start >= self.sample_sec:
                self._save_current_pose_sample()
                self.pose_idx += 1

                if self.pose_idx >= len(self.pose_offsets):
                    self.get_logger().info('All poses sampled. Solving extrinsic...')
                    self._solve_and_save()
                    self.get_logger().info('Auto calibration finished.')
                    rclpy.shutdown()
                    return

                self.phase = 'SETTLE'
                self.phase_start = now
                self.get_logger().info(
                    f'Pose {self.pose_idx}: offset={self.pose_offsets[self.pose_idx]}, settling...'
                )

    def _save_current_pose_sample(self):
        if len(self.sample_tag_pts) < 5:
            self.get_logger().warn(f'Pose {self.pose_idx}: not enough samples, skipped.')
            return

        zed_pts = np.asarray(self.sample_tag_pts)
        robot_pts = np.asarray(self.sample_robot_pts)

        zed_mean = zed_pts.mean(axis=0)
        robot_mean = robot_pts.mean(axis=0)

        self.samples.append((zed_mean, robot_mean))

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                len(self.samples) - 1,
                zed_mean[0], zed_mean[1], zed_mean[2],
                robot_mean[0], robot_mean[1], robot_mean[2],
            ])

        self.get_logger().info(
            f'Saved sample {len(self.samples)-1}: '
            f'zed={np.round(zed_mean, 4)}, pelvis={np.round(robot_mean, 4)}, '
            f'zed_std={np.round(zed_pts.std(axis=0), 5)}, '
            f'pelvis_std={np.round(robot_pts.std(axis=0), 5)}'
        )

    def _solve_and_save(self):
        if len(self.samples) < 3:
            self.get_logger().error('Need at least 3 samples to solve extrinsic.')
            return

        P = np.asarray([s[0] for s in self.samples], dtype=np.float64)  # zed
        Q = np.asarray([s[1] for s in self.samples], dtype=np.float64)  # pelvis

        p_mean = P.mean(axis=0)
        q_mean = Q.mean(axis=0)

        X = P - p_mean
        Y = Q - q_mean

        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = q_mean - R @ p_mean

        Q_pred = (R @ P.T).T + t
        residuals = np.linalg.norm(Q_pred - Q, axis=1)

        quat_xyzw = Rotation.from_matrix(R).as_quat()

        result = {
            'mapping': 'p_pelvis = R @ p_zed_world + t',
            'num_samples': int(len(self.samples)),
            'R': R.tolist(),
            't': t.tolist(),
            'quat_xyzw': quat_xyzw.tolist(),
            'extrinsic_tx': float(t[0]),
            'extrinsic_ty': float(t[1]),
            'extrinsic_tz': float(t[2]),
            'extrinsic_qx': float(quat_xyzw[0]),
            'extrinsic_qy': float(quat_xyzw[1]),
            'extrinsic_qz': float(quat_xyzw[2]),
            'extrinsic_qw': float(quat_xyzw[3]),
            'residuals_m': residuals.tolist(),
            'mean_residual_m': float(residuals.mean()),
            'max_residual_m': float(residuals.max()),
        }

        with open(self.result_path, 'w') as f:
            json.dump(result, f, indent=2)

        self.get_logger().info('======== Calibration Result ========')
        self.get_logger().info(f't = {np.round(t, 6)}')
        self.get_logger().info(f'quat_xyzw = {np.round(quat_xyzw, 6)}')
        self.get_logger().info(
            f'mean residual = {residuals.mean():.6f} m, '
            f'max residual = {residuals.max():.6f} m'
        )
        self.get_logger().info(f'Result written to {self.result_path}')


def main(args=None):
    rclpy.init(args=args)
    node = AutoCalibrationSampler()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()