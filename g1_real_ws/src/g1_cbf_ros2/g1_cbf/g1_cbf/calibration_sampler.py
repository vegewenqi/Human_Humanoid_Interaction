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
from scipy.optimize import least_squares

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

        # ---------------- Parameters ----------------
        self.declare_parameter(
            'urdf_path',
            '/ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf'
        )
        self.declare_parameter('tag_topic', '/tag_center_zed_world')
        self.declare_parameter('lowstate_topic', '/lowstate')
        self.declare_parameter('qdes_topic', '/calib_upperbody_q_des')

        self.declare_parameter(
            'csv_path',
            '/ws/calibration/g1_tag_calibration_samples.csv'
        )
        self.declare_parameter(
            'result_path',
            '/ws/calibration/g1_tag_extrinsic_result.json'
        )

        # This is now used only as an initial guess for optimization,
        # not as the final assumed truth.
        self.declare_parameter('tag_frame', 'torso_link')
        self.declare_parameter('tag_offset_x', 0.11613403306399124)
        self.declare_parameter('tag_offset_y', 0.008857834098559534)
        self.declare_parameter('tag_offset_z', 0.13122103529705606)
        self.declare_parameter('vision_frame_name', 'zed_world')

        self.declare_parameter('control_dt', 0.02)
        self.declare_parameter('start_delay_sec', 4.0)
        self.declare_parameter('settle_sec', 4.0)
        self.declare_parameter('sample_sec', 1.0)

        self.declare_parameter('roll_amp', 0.08)
        self.declare_parameter('pitch_amp', 0.08)

        # Safety / quality gates
        self.declare_parameter('tag_timeout_sec', 0.30)
        self.declare_parameter('max_zed_std_m', 0.015)
        self.declare_parameter('max_robot_std_m', 0.005)
        self.declare_parameter('max_retries_per_pose', 2)

        urdf_path = self.get_parameter('urdf_path').value
        if not urdf_path:
            raise RuntimeError('urdf_path parameter is required')

        self.tag_frame = self.get_parameter('tag_frame').value
        self.initial_tag_offset = np.array([
            float(self.get_parameter('tag_offset_x').value),
            float(self.get_parameter('tag_offset_y').value),
            float(self.get_parameter('tag_offset_z').value),
        ], dtype=np.float64)
        self.vision_frame_name = self.get_parameter('vision_frame_name').value

        self.kin = CalibrationKinematics(
            urdf_path=urdf_path,
            tag_frame=self.tag_frame,
            tag_offset_xyz=self.initial_tag_offset,
        )

        # Resolve torso/tag frame id directly from Pinocchio model.
        self.tag_frame_id = self.kin.model.getFrameId(self.tag_frame)
        if self.tag_frame_id >= self.kin.model.nframes:
            raise RuntimeError(f'Frame {self.tag_frame} not found in URDF.')

        self.csv_path = self.get_parameter('csv_path').value
        self.result_path = self.get_parameter('result_path').value

        self.control_dt = float(self.get_parameter('control_dt').value)
        self.start_delay_sec = float(self.get_parameter('start_delay_sec').value)
        self.settle_sec = float(self.get_parameter('settle_sec').value)
        self.sample_sec = float(self.get_parameter('sample_sec').value)
        self.tag_timeout_sec = float(self.get_parameter('tag_timeout_sec').value)
        self.max_zed_std_m = float(self.get_parameter('max_zed_std_m').value)
        self.max_robot_std_m = float(self.get_parameter('max_robot_std_m').value)
        self.max_retries_per_pose = int(
            self.get_parameter('max_retries_per_pose').value
        )

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

        # ---------------- State ----------------
        self.latest_tag_zed = None
        self.latest_tag_time = None
        self.latest_q_full = None
        self.latest_q8 = None
        self.base_q8 = None

        self.inputs_ready_time = None
        self.pose_idx = -1
        self.phase = 'WAIT'
        self.phase_start = None

        self.sample_tag_pts = []
        self.sample_torso_ts = []
        self.sample_torso_Rs = []

        # Each sample:
        # {
        #   'zed': (3,),
        #   'torso_t': (3,),
        #   'torso_R': (3,3),
        # }
        self.samples = []

        self.retries_for_current_pose = 0
        self.finished = False

        # ---------------- ROS interfaces ----------------
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
                'torso_tx', 'torso_ty', 'torso_tz',
                'torso_qx', 'torso_qy', 'torso_qz', 'torso_qw',
                'rough_tag_pelvis_x', 'rough_tag_pelvis_y', 'rough_tag_pelvis_z',
            ])

        self.create_timer(self.control_dt, self._tick)

        self.get_logger().info('Auto calibration sampler started.')
        self.get_logger().info(f'CSV: {self.csv_path}')
        self.get_logger().info(f'Result: {self.result_path}')
        self.get_logger().info(f'Number of poses: {len(self.pose_offsets)}')
        self.get_logger().info(
            f'tag_frame={self.tag_frame}, '
            f'initial_tag_offset={self.initial_tag_offset.tolist()}'
        )
        self.get_logger().info(
            f'start_delay_sec={self.start_delay_sec:.2f}, '
            f'settle_sec={self.settle_sec:.2f}, '
            f'sample_sec={self.sample_sec:.2f}'
        )
        self.get_logger().info(
            f'quality gates: tag_timeout_sec={self.tag_timeout_sec:.2f}, '
            f'max_zed_std_m={self.max_zed_std_m:.4f}, '
            f'max_robot_std_m={self.max_robot_std_m:.4f}'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _tag_cb(self, msg):
        self.latest_tag_zed = np.array([
            msg.point.x,
            msg.point.y,
            msg.point.z,
        ], dtype=np.float64)
        self.latest_tag_time = time.perf_counter()

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

        # q8 layout expected by g1_arm_sdk_bridge:
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

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

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

    def _tag_is_fresh(self, now):
        if self.latest_tag_time is None:
            return False
        return (now - self.latest_tag_time) <= self.tag_timeout_sec

    def _get_tag_frame_pose_in_pelvis(self):
        """Return tag_frame pose in pelvis/URDF world after kin.update()."""
        oMf = self.kin.data.oMf[self.tag_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    # ------------------------------------------------------------------
    # Main state machine
    # ------------------------------------------------------------------

    def _tick(self):
        if self.finished:
            return

        now = time.perf_counter()

        if (
            self.latest_tag_zed is None
            or self.latest_q_full is None
            or self.latest_q8 is None
        ):
            self.get_logger().info(
                'Waiting for /tag_center_zed_world and /lowstate...',
                throttle_duration_sec=2.0,
            )
            return

        if self.inputs_ready_time is None:
            self.inputs_ready_time = now
            self.get_logger().info(
                f'Inputs ready. Waiting start_delay_sec='
                f'{self.start_delay_sec:.2f}s before starting calibration...'
            )
            return

        if now - self.inputs_ready_time < self.start_delay_sec:
            return

        if self.base_q8 is None:
            self.base_q8 = self.latest_q8.copy()
            self.get_logger().info(
                f'Base q8 captured: {np.round(self.base_q8, 4)}'
            )
            self.pose_idx = 0
            self.phase = 'SETTLE'
            self.phase_start = now
            self.retries_for_current_pose = 0
            self.get_logger().info(
                f'Pose {self.pose_idx}: '
                f'offset={self.pose_offsets[self.pose_idx]}, settling...'
            )

        self._publish_target()

        if self.phase == 'SETTLE':
            if now - self.phase_start >= self.settle_sec:
                self.sample_tag_pts = []
                self.sample_torso_ts = []
                self.sample_torso_Rs = []
                self.phase = 'SAMPLE'
                self.phase_start = now
                self.get_logger().info(
                    f'Pose {self.pose_idx}: sampling for '
                    f'{self.sample_sec:.2f}s...'
                )

        elif self.phase == 'SAMPLE':
            if not self._tag_is_fresh(now):
                self.get_logger().warn(
                    'Tag data stale; not collecting this frame.',
                    throttle_duration_sec=1.0,
                )
                return

            self.kin.update(self.latest_q_full)
            torso_t, torso_R = self._get_tag_frame_pose_in_pelvis()

            self.sample_tag_pts.append(self.latest_tag_zed.copy())
            self.sample_torso_ts.append(torso_t.copy())
            self.sample_torso_Rs.append(torso_R.copy())

            if now - self.phase_start >= self.sample_sec:
                accepted = self._save_current_pose_sample()

                if accepted:
                    self.pose_idx += 1
                    self.retries_for_current_pose = 0
                else:
                    self.retries_for_current_pose += 1
                    if self.retries_for_current_pose <= self.max_retries_per_pose:
                        self.get_logger().warn(
                            f'Pose {self.pose_idx}: retry '
                            f'{self.retries_for_current_pose}/'
                            f'{self.max_retries_per_pose}'
                        )
                        self.phase = 'SETTLE'
                        self.phase_start = now
                        return

                    self.get_logger().error(
                        f'Pose {self.pose_idx}: failed after '
                        f'{self.max_retries_per_pose} retries. Skipping pose.'
                    )
                    self.pose_idx += 1
                    self.retries_for_current_pose = 0

                if self.pose_idx >= len(self.pose_offsets):
                    self.get_logger().info(
                        'All poses processed. Solving extrinsic + tag offset...'
                    )
                    self._solve_and_save()
                    self.get_logger().info('Auto calibration finished.')
                    self.finished = True
                    return

                self.phase = 'SETTLE'
                self.phase_start = now
                self.get_logger().info(
                    f'Pose {self.pose_idx}: '
                    f'offset={self.pose_offsets[self.pose_idx]}, settling...'
                )

    # ------------------------------------------------------------------
    # Sample / solve
    # ------------------------------------------------------------------

    def _mean_rotation_matrix(self, rot_mats):
        rots = Rotation.from_matrix(np.asarray(rot_mats, dtype=np.float64))
        return rots.mean().as_matrix()

    def _save_current_pose_sample(self):
        if len(self.sample_tag_pts) < 5:
            self.get_logger().warn(
                f'Pose {self.pose_idx}: not enough samples, skipped.'
            )
            return False

        zed_pts = np.asarray(self.sample_tag_pts, dtype=np.float64)
        torso_ts = np.asarray(self.sample_torso_ts, dtype=np.float64)
        torso_Rs = np.asarray(self.sample_torso_Rs, dtype=np.float64)

        zed_mean = zed_pts.mean(axis=0)
        torso_t_mean = torso_ts.mean(axis=0)
        torso_R_mean = self._mean_rotation_matrix(torso_Rs)

        # Quality gates
        zed_std = zed_pts.std(axis=0)
        torso_t_std = torso_ts.std(axis=0)

        if np.max(zed_std) > self.max_zed_std_m:
            self.get_logger().warn(
                f'Pose {self.pose_idx}: rejected due to high ZED std '
                f'{np.round(zed_std, 5)} > {self.max_zed_std_m:.5f}'
            )
            return False

        if np.max(torso_t_std) > self.max_robot_std_m:
            self.get_logger().warn(
                f'Pose {self.pose_idx}: rejected due to high torso translation std '
                f'{np.round(torso_t_std, 5)} > {self.max_robot_std_m:.5f}'
            )
            return False

        rough_tag_pelvis = torso_R_mean @ self.initial_tag_offset + torso_t_mean
        torso_quat_xyzw = Rotation.from_matrix(torso_R_mean).as_quat()

        self.samples.append({
            'zed': zed_mean,
            'torso_t': torso_t_mean,
            'torso_R': torso_R_mean,
        })

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                len(self.samples) - 1,
                zed_mean[0], zed_mean[1], zed_mean[2],
                torso_t_mean[0], torso_t_mean[1], torso_t_mean[2],
                torso_quat_xyzw[0], torso_quat_xyzw[1],
                torso_quat_xyzw[2], torso_quat_xyzw[3],
                rough_tag_pelvis[0], rough_tag_pelvis[1], rough_tag_pelvis[2],
            ])

        self.get_logger().info(
            f'Saved sample {len(self.samples)-1}: '
            f'zed={np.round(zed_mean, 4)}, '
            f'torso_t={np.round(torso_t_mean, 4)}, '
            f'rough_tag_pelvis={np.round(rough_tag_pelvis, 4)}, '
            f'zed_std={np.round(zed_std, 5)}, '
            f'torso_t_std={np.round(torso_t_std, 5)}'
        )
        return True

    def _initial_extrinsic_from_guess_offset(self):
        """Use initial tag offset to get a reasonable initial R,t."""
        P = np.asarray([s['zed'] for s in self.samples], dtype=np.float64)

        Q_guess = np.asarray([
            s['torso_R'] @ self.initial_tag_offset + s['torso_t']
            for s in self.samples
        ], dtype=np.float64)

        p_mean = P.mean(axis=0)
        q_mean = Q_guess.mean(axis=0)

        X = P - p_mean
        Y = Q_guess - q_mean

        H = X.T @ Y
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = q_mean - R @ p_mean
        return R, t

    def _solve_joint_extrinsic_and_offset(self):
        P = np.asarray([s['zed'] for s in self.samples], dtype=np.float64)
        torso_ts = np.asarray([s['torso_t'] for s in self.samples], dtype=np.float64)
        torso_Rs = np.asarray([s['torso_R'] for s in self.samples], dtype=np.float64)

        R0, t0 = self._initial_extrinsic_from_guess_offset()
        rvec0 = Rotation.from_matrix(R0).as_rotvec()

        x0 = np.zeros(9, dtype=np.float64)
        x0[0:3] = rvec0
        x0[3:6] = t0
        x0[6:9] = self.initial_tag_offset

        def residual_fn(x):
            R_ext = Rotation.from_rotvec(x[0:3]).as_matrix()
            t_ext = x[3:6]
            tag_offset = x[6:9]

            residuals = []
            for i in range(P.shape[0]):
                p_from_vision = R_ext @ P[i] + t_ext
                p_from_fk = torso_Rs[i] @ tag_offset + torso_ts[i]
                residuals.append(p_from_vision - p_from_fk)

            return np.concatenate(residuals)

        result = least_squares(
            residual_fn,
            x0,
            loss='soft_l1',
            f_scale=0.02,
            max_nfev=2000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        x = result.x
        R_ext = Rotation.from_rotvec(x[0:3]).as_matrix()
        t_ext = x[3:6]
        tag_offset = x[6:9]

        raw_res = residual_fn(x).reshape(-1, 3)
        residual_norms = np.linalg.norm(raw_res, axis=1)

        return R_ext, t_ext, tag_offset, residual_norms, result

    def _solve_and_save(self):
        if len(self.samples) < 4:
            self.get_logger().error(
                f'Need at least 4 accepted samples to solve joint extrinsic + offset, '
                f'got {len(self.samples)}.'
            )
            return

        R, t, tag_offset, residuals, opt_result = (
            self._solve_joint_extrinsic_and_offset()
        )

        quat_xyzw = Rotation.from_matrix(R).as_quat()

        result = {
            'mapping': f'p_pelvis = R @ p_{self.vision_frame_name} + t',
            'method': 'joint_optimization_extrinsic_and_tag_offset',
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

            'optimized_tag_frame': self.tag_frame,
            'initial_tag_offset_xyz': self.initial_tag_offset.tolist(),
            'optimized_tag_offset_xyz': tag_offset.tolist(),

            'residuals_m': residuals.tolist(),
            'mean_residual_m': float(residuals.mean()),
            'max_residual_m': float(residuals.max()),
            'optimizer_success': bool(opt_result.success),
            'optimizer_status': int(opt_result.status),
            'optimizer_message': str(opt_result.message),
            'optimizer_cost': float(opt_result.cost),
        }

        with open(self.result_path, 'w') as f:
            json.dump(result, f, indent=2)

        self.get_logger().info('======== Calibration Result ========')
        self.get_logger().info(f't = {np.round(t, 6)}')
        self.get_logger().info(f'quat_xyzw = {np.round(quat_xyzw, 6)}')
        self.get_logger().info(
            f'optimized tag offset in {self.tag_frame}: '
            f'{np.round(tag_offset, 6)}'
        )
        self.get_logger().info(
            f'mean residual = {residuals.mean():.6f} m, '
            f'max residual = {residuals.max():.6f} m'
        )
        self.get_logger().info(f'optimizer success = {opt_result.success}')
        self.get_logger().info(f'Result written to {self.result_path}')


def main(args=None):
    rclpy.init(args=args)
    node = AutoCalibrationSampler()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()