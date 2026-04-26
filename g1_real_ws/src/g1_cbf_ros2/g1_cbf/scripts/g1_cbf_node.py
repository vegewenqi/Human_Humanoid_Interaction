#!/usr/bin/env python3
"""CBF safety filter node for G1 humanoid.

Supports:
- robot self-collision CBF
- robot vs human-capsule CBF
- optional robot vs box-obstacle CBF

Safe commands are published on /joint_commands at a fixed rate (1/dt Hz).
"""

import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as Rot
import tf2_ros

try:
    from vision_msgs.msg import Detection3DArray
    _HAS_VISION_MSGS = True
except Exception:
    Detection3DArray = None
    _HAS_VISION_MSGS = False

from g1_cbf.kinematics import G1Kinematics, CONTROLLED_JOINTS, COLLISION_PAIRS
from g1_cbf.qp_solver import CBFQPSolver
from g1_cbf.collider_viz import ColliderVisualizer


# Must match the exact order published by human_skeleton_capsule.py /
# human_capsule_frame_transform.py
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

ROBOT_HUMAN_COLLISION_PAIRS = [
    ('left_arm', 'right_upper_arm'),
    ('left_arm', 'right_forearm_hand'),
    ('right_arm', 'left_upper_arm'),
    ('right_arm', 'left_forearm_hand'),
    ('left_upper_arm', 'right_upper_arm'),
    ('left_upper_arm', 'right_forearm_hand'),
    ('right_upper_arm', 'left_upper_arm'),
    ('right_upper_arm', 'left_forearm_hand'),
    ('torso', 'right_upper_arm'),
    ('torso', 'right_forearm_hand'),
    ('torso', 'left_upper_arm'),
    ('torso', 'left_forearm_hand'),
    ('left_arm', 'torso'),
    ('right_arm', 'torso'),
    ('left_upper_arm', 'torso'),
    ('right_upper_arm', 'torso'),
    ('left_arm', 'right_thigh'),
    ('right_arm', 'left_thigh'),

]


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('rr_margin_phi', 0.0063)
        self.declare_parameter('hr_margin_phi', 0.028)
        self.declare_parameter('rr_safety_distance', 0.02)
        self.declare_parameter('hr_safety_distance', 0.08)
        self.declare_parameter('beta', 1.05)
        self.declare_parameter('rr_gamma', 2.0)
        self.declare_parameter('hr_gamma', 2.0)
        self.declare_parameter('K', 5.0)
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('lpf_gain', 0.1)
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('collision_geometry', 'capsules')
        self.declare_parameter('use_gpu', False)

        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('unsafe_cmd_topic', '/joint_commands_unsafe')
        self.declare_parameter('safe_cmd_topic', '/joint_commands')
        self.declare_parameter('obstacle_topic', '/bbox_3d')
        self.declare_parameter('human_capsule_topic', '/human_capsules_robot')

        self.declare_parameter('enable_self_collision', True)
        self.declare_parameter('enable_human_collision', True)
        self.declare_parameter('enable_box_obstacles', False)
        self.declare_parameter('enable_coarse_gating', True)
        self.declare_parameter('coarse_distance_activate', 0.60)

        self.declare_parameter('enable_dynamic_human_cbf', True)
        self.declare_parameter('human_velocity_lpf_alpha', 0.35)
        self.declare_parameter('human_velocity_max', 2.0)
        self.declare_parameter('human_velocity_dt_min', 0.01)
        self.declare_parameter('human_velocity_dt_max', 0.20)  

        # Runtime / debug switches
        self.declare_parameter('enable_robot_caps_viz', True)
        self.declare_parameter('enable_distance_viz', True)
        self.declare_parameter('log_summary', False)
        self.declare_parameter('summary_period_sec', 1.0)

        # Set JAX config before importing cbf module
        use_gpu = self.get_parameter('use_gpu').value
        os.environ['JAX_PLATFORMS'] = 'cuda' if use_gpu else 'cpu'
        os.environ['JAX_ENABLE_X64'] = '1'
        try:
            import jax
            self.get_logger().info(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")
            self.get_logger().info(f"JAX backend={jax.default_backend()}")
            self.get_logger().info(f"JAX devices={jax.devices()}")
        except Exception as e:
            self.get_logger().error(f"JAX precheck failed: {e}")
            raise
        from g1_cbf.cbf import DpaxCapsuleCBF, DpaxBoxCBF  # noqa: E402

        dt = float(self.get_parameter('dt').value)
        rr_gamma = float(self.get_parameter('rr_gamma').value)
        hr_gamma = float(self.get_parameter('hr_gamma').value)
        rr_margin_phi = float(self.get_parameter('rr_margin_phi').value)
        hr_margin_phi = float(self.get_parameter('hr_margin_phi').value)
        rr_safety_distance = float(self.get_parameter('rr_safety_distance').value)  
        hr_safety_distance = float(self.get_parameter('hr_safety_distance').value)
        urdf_path = self.get_parameter('urdf_path').value
        self.geom_type = self.get_parameter('collision_geometry').value

        joint_state_topic = self.get_parameter('joint_state_topic').value
        unsafe_cmd_topic = self.get_parameter('unsafe_cmd_topic').value
        safe_cmd_topic = self.get_parameter('safe_cmd_topic').value
        obstacle_topic = self.get_parameter('obstacle_topic').value
        human_capsule_topic = self.get_parameter('human_capsule_topic').value

        self.enable_self_collision = bool(
            self.get_parameter('enable_self_collision').value
        )
        self.enable_human_collision = bool(
            self.get_parameter('enable_human_collision').value
        )
        self.enable_box_obstacles = bool(
            self.get_parameter('enable_box_obstacles').value
        )
        self.enable_coarse_gating = bool(
            self.get_parameter('enable_coarse_gating').value
        )
        self.coarse_distance_activate = float(
            self.get_parameter('coarse_distance_activate').value
        )
        self.enable_dynamic_human_cbf = bool(
            self.get_parameter('enable_dynamic_human_cbf').value
        )
        self.human_velocity_lpf_alpha = float(
            self.get_parameter('human_velocity_lpf_alpha').value
        )
        self.human_velocity_max = float(
            self.get_parameter('human_velocity_max').value
        )
        self.human_velocity_dt_min = float(
            self.get_parameter('human_velocity_dt_min').value
        )
        self.human_velocity_dt_max = float(
            self.get_parameter('human_velocity_dt_max').value
        )
        self.enable_robot_caps_viz = bool(self.get_parameter('enable_robot_caps_viz').value)
        self.enable_distance_viz = bool(
            self.get_parameter('enable_distance_viz').value
        )
        self.log_summary = bool(self.get_parameter('log_summary').value)
        self.summary_period_sec = float(
            self.get_parameter('summary_period_sec').value
        )

        if not urdf_path:
            self.get_logger().fatal('urdf_path parameter is required')
            raise RuntimeError('urdf_path not set')

        self.get_logger().info(f'Loading URDF: {urdf_path}')
        self.get_logger().info(
            f'CBF params: dt={dt}, rr_gamma={rr_gamma}, rr_margin_phi={rr_margin_phi}, '
            f'hr_gamma={hr_gamma}, hr_margin_phi={hr_margin_phi}, self_geom={self.geom_type}'
        )
        self.get_logger().info(
            f'enable_self_collision={self.enable_self_collision}, '
            f'enable_human_collision={self.enable_human_collision}, '
            f'enable_box_obstacles={self.enable_box_obstacles}, '
            f'coarse_gating: enable={self.enable_coarse_gating}, '
            f'd_activate={self.coarse_distance_activate:.3f} m, '
            f'dynamic_human_cbf={self.enable_dynamic_human_cbf}, '
            f'human_vel_lpf={self.human_velocity_lpf_alpha:.2f}, '
            f'human_vel_max={self.human_velocity_max:.2f} m/s, '
            f'viz: enable_robot_caps_viz={self.enable_robot_caps_viz}, '
            f'enable_distance_viz={self.enable_distance_viz}, '
            f'log_summary={self.log_summary}, '
            f'summary_period_sec={self.summary_period_sec:.2f}'
        )

        # ---------------- Subsystems ----------------
        self.kin = G1Kinematics(urdf_path)
        self.get_logger().info('Initializing dpax CBFs (JAX JIT warmup)...')

        # Self-collision CBF
        if self.geom_type == 'boxes':
            beta = float(self.get_parameter('beta').value)
            self.self_cbf = DpaxBoxCBF(gamma=rr_gamma, beta=beta)
        else:
            self.self_cbf = DpaxCapsuleCBF(
                gamma=rr_gamma, margin_phi=rr_margin_phi, safety_distance=rr_safety_distance
            )

        # Human-collision CBF always uses capsules
        self.human_cbf = DpaxCapsuleCBF(
            gamma=hr_gamma, margin_phi=hr_margin_phi, safety_distance=hr_safety_distance
        )

        # Optional box obstacle CBF
        beta = float(self.get_parameter('beta').value)
        self.box_cbf = DpaxBoxCBF(gamma=hr_gamma, beta=beta)

        self.get_logger().info('dpax CBFs ready')

        # Max possible constraints:
        #   self collision + robot-human all-pairs + some obstacle pairs
        max_self = len(COLLISION_PAIRS)
        max_human = len(ROBOT_HUMAN_COLLISION_PAIRS)
        max_box = len(self.kin.collision_bodies) * 2  # assume at most 2 box obstacles at a time
        self.qp = CBFQPSolver(
            n_joints=self.kin.n_q,
            n_cbf=max_self + max_human + max_box,
        )

        self.viz = None
        if self.enable_robot_caps_viz or self.enable_distance_viz:
            # TODO:
            # If sim and real CBF nodes run together, split collider/distance
            # marker topics for sim vs real to avoid mixed visualization.
            self.viz = ColliderVisualizer(
                self, self.kin, geometry_type=self.geom_type,
            )

        # ---------------- State ----------------
        self.q_full = None
        self.q_des_latest = None
        self.q_des_filtered = None
        self.q_cbf_target = None

        # human capsules in robot frame
        # dict[name] = {
        #   'a':(3,), 'b':(3,), 'radius':float,
        #   'v_a':(3,), 'v_b':(3,),
        # }
        self.human_capsules = {}
        self._prev_human_capsules_raw = {}
        self._prev_human_capsules_time = None

        # box obstacles
        self._obstacles = []  # list of {center, rot, half_extents}
        self._zero_J6 = np.zeros((6, self.kin.n_q))

        # TF for obstacle frame conversion
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Velocity limits (rad/s)
        self.dq_max = np.array([
            30.0, 30.0,               # waist roll/pitch
            37.0, 37.0, 37.0,         # left arm
            37.0, 37.0, 37.0,         # right arm
        ])
        self.dq_min = -self.dq_max

        # Summary counters
        self._summary_last_time = time.perf_counter()
        self._summary_tick_count = 0
        self._summary_last_constraints = 0
        self._summary_last_total_ms = 0.0
        self._summary_last_qp_ms = 0.0

        # ---------------- Subscribers ----------------
        self.create_subscription(
            JointState, joint_state_topic,
            self._joint_states_cb, 10,
        )
        self.create_subscription(
            JointState, unsafe_cmd_topic,
            self._unsafe_cmd_cb, 10,
        )
        self.create_subscription(
            Float32MultiArray, human_capsule_topic,
            self._human_capsules_cb, 10,
        )

        if self.enable_box_obstacles and _HAS_VISION_MSGS:
            self.create_subscription(
                Detection3DArray, obstacle_topic,
                self._bbox_cb, 10,
            )
        elif self.enable_box_obstacles and not _HAS_VISION_MSGS:
            self.get_logger().warn(
                'enable_box_obstacles=True but vision_msgs is unavailable; '
                'box obstacle constraints disabled.'
            )

        # ---------------- Publisher ----------------
        self.cmd_pub = self.create_publisher(
            JointState, safe_cmd_topic, 10,
        )

        # ---------------- Timer ----------------
        self.create_timer(dt, self._tick)

        self.get_logger().info(
            f'topics: state={joint_state_topic}, unsafe={unsafe_cmd_topic}, '
            f'safe={safe_cmd_topic}, human={human_capsule_topic}, '
            f'obstacle={obstacle_topic}'
        )
        self.get_logger().info(
            f'enable_self_collision={self.enable_self_collision}, '
            f'enable_human_collision={self.enable_human_collision}, '
            f'enable_box_obstacles={self.enable_box_obstacles}'
        )
        self.get_logger().info(
            f'g1_cbf_node ready — timer period={dt:.3f}s ({1.0/dt:.0f} Hz target)'
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_states_cb(self, msg: JointState):
        self.q_full = self.kin.joint_names_to_q_full(
            list(msg.name), list(msg.position),
        )

    def _unsafe_cmd_cb(self, msg: JointState):
        q_des = self._extract_controlled(msg)
        if q_des is not None:
            self.q_des_latest = q_des

    def _human_capsules_cb(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float64)
        expected = 7 * len(HUMAN_CAPSULE_NAMES)
        if data.size != expected:
            self.get_logger().warn(
                f'/human_capsules_robot expected length {expected}, got {data.size}',
                throttle_duration_sec=2.0,
            )
            return

        now = time.perf_counter()

        raw_caps = {}
        for i, name in enumerate(HUMAN_CAPSULE_NAMES):
            s = 7 * i
            block = data[s:s + 7]
            if not np.all(np.isfinite(block[:6])) or not np.isfinite(block[6]):
                continue

            raw_caps[name] = {
                'a': block[0:3].copy(),
                'b': block[3:6].copy(),
                'radius': float(block[6]),
            }

        # Estimate dt from callback arrival time.
        dt_h = None
        if self._prev_human_capsules_time is not None:
            dt_h = now - self._prev_human_capsules_time

        alpha = float(np.clip(self.human_velocity_lpf_alpha, 0.0, 1.0))
        v_max = max(float(self.human_velocity_max), 0.0)

        use_velocity = (
            self.enable_dynamic_human_cbf
            and dt_h is not None
            and self.human_velocity_dt_min <= dt_h <= self.human_velocity_dt_max
        )

        caps = {}
        for name, cur in raw_caps.items():
            a = cur['a']
            b = cur['b']
            r = cur['radius']

            v_a = np.zeros(3, dtype=np.float64)
            v_b = np.zeros(3, dtype=np.float64)

            if use_velocity and name in self._prev_human_capsules_raw:
                prev = self._prev_human_capsules_raw[name]
                a_prev = prev['a']
                b_prev = prev['b']

                v_a_raw = (a - a_prev) / dt_h
                v_b_raw = (b - b_prev) / dt_h

                v_a_raw = self._clip_vec_norm(v_a_raw, v_max)
                v_b_raw = self._clip_vec_norm(v_b_raw, v_max)

                # Low-pass filter velocity using previous filtered velocity if available.
                prev_filt = self.human_capsules.get(name, None)
                if prev_filt is not None:
                    v_a_prev = prev_filt.get('v_a', np.zeros(3, dtype=np.float64))
                    v_b_prev = prev_filt.get('v_b', np.zeros(3, dtype=np.float64))
                    v_a = (1.0 - alpha) * v_a_prev + alpha * v_a_raw
                    v_b = (1.0 - alpha) * v_b_prev + alpha * v_b_raw
                else:
                    v_a = v_a_raw
                    v_b = v_b_raw

                v_a = self._clip_vec_norm(v_a, v_max)
                v_b = self._clip_vec_norm(v_b, v_max)

            caps[name] = {
                'a': a,
                'b': b,
                'radius': r,
                'v_a': v_a,
                'v_b': v_b,
            }

        self.human_capsules = caps
        self._prev_human_capsules_raw = raw_caps
        self._prev_human_capsules_time = now

    def _bbox_cb(self, msg: Detection3DArray):
        obstacles = []
        for det in msg.detections:
            frame_id = det.header.frame_id or msg.header.frame_id
            try:
                tf = self.tf_buffer.lookup_transform(
                    'pelvis', frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                continue

            t = tf.transform
            tf_pos = np.array([
                t.translation.x, t.translation.y, t.translation.z,
            ])
            tf_rot = Rot.from_quat([
                t.rotation.x, t.rotation.y,
                t.rotation.z, t.rotation.w,
            ]).as_matrix()

            det_pos = np.array([
                det.bbox.center.position.x,
                det.bbox.center.position.y,
                det.bbox.center.position.z,
            ])
            det_rot = Rot.from_quat([
                det.bbox.center.orientation.x,
                det.bbox.center.orientation.y,
                det.bbox.center.orientation.z,
                det.bbox.center.orientation.w,
            ]).as_matrix()

            center = tf_rot @ det_pos + tf_pos
            rot = tf_rot @ det_rot
            half_extents = np.array([
                det.bbox.size.x / 2.0,
                det.bbox.size.y / 2.0,
                det.bbox.size.z / 2.0,
            ])

            obstacles.append({
                'center': center,
                'rot': rot,
                'half_extents': half_extents,
            })

        self._obstacles = obstacles

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _tick(self):
        if self.q_full is None or self.q_des_latest is None:
            return

        t0 = time.perf_counter()

        dt = float(self.get_parameter('dt').value)
        K = float(self.get_parameter('K').value)
        max_vel = float(self.get_parameter('max_velocity').value)
        lpf = float(self.get_parameter('lpf_gain').value)

        q_ctrl = self.kin.extract_controlled(self.q_full)

        # Initialize targets
        if self.q_des_filtered is None:
            self.q_des_filtered = self.q_des_latest.copy()
        if self.q_cbf_target is None:
            self.q_cbf_target = q_ctrl.copy()

        # LPF
        if 0 < lpf < 1:
            self.q_des_filtered += lpf * (
                self.q_des_latest - self.q_des_filtered
            )
        else:
            self.q_des_filtered = self.q_des_latest.copy()

        # Nominal velocity
        dq_ref = K * (self.q_des_filtered - self.q_cbf_target)
        dq_ref = np.clip(dq_ref, -max_vel, max_vel)

        # FK
        self.kin.update(self.q_full)

        stamp = self.get_clock().now().to_msg()

        # Robot collider visualization
        if self.enable_robot_caps_viz and self.viz is not None:
            self.viz.publish(stamp)

        # Build robot endpoint cache once for capsule mode
        robot_endpoints = None
        if self.geom_type != 'boxes':
            robot_endpoints = self._build_robot_endpoint_cache()

        constraints = []
        closest_points = []

        # self collision
        if self.enable_self_collision:
            if self.geom_type == 'boxes':
                self._build_box_constraints(
                    constraints, closest_points,
                )
            else:
                self._build_capsule_constraints(
                    robot_endpoints, constraints, closest_points,
                )

        # human collision
        if self.enable_human_collision and self.human_capsules:
            self._build_human_capsule_constraints(
                robot_endpoints, constraints, closest_points,
            )

        # optional box obstacles
        if self.enable_box_obstacles and self._obstacles:
            self._build_obstacle_constraints(
                constraints, closest_points,
            )

        # Publish distance lines
        if self.enable_distance_viz and self.viz is not None:
            self.viz.publish_distances(stamp, closest_points)

        t_qp0 = time.perf_counter()
        dq_safe = self.qp.solve(
            dq_ref, constraints,
            self.dq_min, self.dq_max,
        )
        t_qp1 = time.perf_counter()

        self.q_cbf_target += dq_safe * dt

        # Prevent divergence
        max_lead = 0.5
        self.q_cbf_target = np.clip(
            self.q_cbf_target,
            q_ctrl - max_lead,
            q_ctrl + max_lead,
        )

        safe_msg = JointState()
        safe_msg.header.stamp = stamp
        safe_msg.name = list(CONTROLLED_JOINTS)
        safe_msg.position = self.q_cbf_target.tolist()
        safe_msg.velocity = dq_safe.tolist()
        self.cmd_pub.publish(safe_msg)

        t1 = time.perf_counter()
        self._update_summary(
            n_constraints=len(constraints),
            total_ms=(t1 - t0) * 1000.0,
            qp_ms=(t_qp1 - t_qp0) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Constraint builders
    # ------------------------------------------------------------------

    def _build_robot_endpoint_cache(self):
        endpoints = {}
        for name in self.kin.collision_bodies:
            a, b, J_a, J_b = self.kin.get_endpoint_jacobians(name)
            body = self.kin.collision_bodies[name]
            endpoints[name] = {
                'a': a,
                'b': b,
                'J_a': J_a,
                'J_b': J_b,
                'radius': body['radius'],
            }
        return endpoints

    def _build_capsule_constraints(self, robot_endpoints, constraints, closest_points):
        for nameA, nameB in COLLISION_PAIRS:
            eA, eB = robot_endpoints[nameA], robot_endpoints[nameB]
            if not self._pair_is_active_by_center_distance(
                eA['a'], eA['b'], eB['a'], eB['b']
            ):
                continue
            phi, A_row, b_val, p1, p2 = self.self_cbf.build_constraint(
                eA['radius'], eA['a'], eA['b'],
                eA['J_a'], eA['J_b'],
                eB['radius'], eB['a'], eB['b'],
                eB['J_a'], eB['J_b'],
                need_closest_points=self.enable_distance_viz,
            )
            constraints.append((A_row, b_val))
            if p1 is not None and p2 is not None:
                closest_points.append((p1, p2))

    def _build_human_capsule_constraints(
        self, robot_endpoints, constraints, closest_points
    ):
        zero_J_template = None
        for robot_name, human_name in ROBOT_HUMAN_COLLISION_PAIRS:
            if robot_name not in robot_endpoints:
                self.get_logger().warn(
                    f'Robot collision body "{robot_name}" not found in kinematics.',
                    throttle_duration_sec=2.0,
                )
                continue
            if human_name not in self.human_capsules:
                continue

            eR = robot_endpoints[robot_name]
            eH = self.human_capsules[human_name]

            if not self._pair_is_active_by_center_distance(
                eR['a'], eR['b'], eH['a'], eH['b']
            ):
                continue

            # Human is not a QP decision variable, so its Jacobian wrt robot dq is zero.
            # Its measured endpoint velocity enters the CBF RHS through hdot_obstacle.
            if zero_J_template is None:
                zero_J_template = np.zeros_like(eR['J_a'])
            J_zero_a = zero_J_template
            J_zero_b = zero_J_template

            vHa = eH.get('v_a', np.zeros(3, dtype=np.float64))
            vHb = eH.get('v_b', np.zeros(3, dtype=np.float64))
            if not self.enable_dynamic_human_cbf:
                vHa = np.zeros(3, dtype=np.float64)
                vHb = np.zeros(3, dtype=np.float64)

            phi, A_row, b_val, p1, p2 = self.human_cbf.build_constraint(
                eR['radius'], eR['a'], eR['b'],
                eR['J_a'], eR['J_b'],
                eH['radius'], eH['a'], eH['b'],
                J_zero_a, J_zero_b,
                need_closest_points=self.enable_distance_viz,
                v_a2=vHa,
                v_b2=vHb,
            )
            constraints.append((A_row, b_val))
            if p1 is not None and p2 is not None:
                closest_points.append((p1, p2))

    def _build_box_constraints(self, constraints, closest_points):
        for nameA, nameB in COLLISION_PAIRS:
            bodyA = self.kin.collision_bodies[nameA]
            bodyB = self.kin.collision_bodies[nameB]
            centerA, rotA = self.kin.get_collision_pose(nameA)
            centerB, rotB = self.kin.get_collision_pose(nameB)
            J6_A = self.kin.get_collision_jacobian(nameA)
            J6_B = self.kin.get_collision_jacobian(nameB)

            alpha, A_row, b_val, p1, p2 = self.self_cbf.build_constraint(
                bodyA, centerA, rotA, J6_A,
                bodyB, centerB, rotB, J6_B,
            )
            constraints.append((A_row, b_val))
            closest_points.append((p1, p2))

    def _build_obstacle_constraints(self, constraints, closest_points):
        from g1_cbf.cbf import _box_b_from_half_extents

        for obs in self._obstacles:
            obs_b = _box_b_from_half_extents(obs['half_extents'])
            for body_name in self.kin.collision_bodies:
                bodyA = self.kin.collision_bodies[body_name]
                centerA, rotA = self.kin.get_collision_pose(body_name)
                J6_A = self.kin.get_collision_jacobian(body_name)

                alpha, A_row, b_val, p1, p2 = self.box_cbf.build_constraint(
                    bodyA, centerA, rotA, J6_A,
                    None, obs['center'], obs['rot'], self._zero_J6,
                    b_override_B=obs_b,
                )
                constraints.append((A_row, b_val))
                closest_points.append((p1, p2))

    # ------------------------------------------------------------------
    # Logging / summary
    # ------------------------------------------------------------------

    def _update_summary(self, n_constraints, total_ms, qp_ms):
        if not self.log_summary:
            return

        self._summary_tick_count += 1
        self._summary_last_constraints = n_constraints
        self._summary_last_total_ms = total_ms
        self._summary_last_qp_ms = qp_ms

        now = time.perf_counter()
        elapsed = now - self._summary_last_time
        if elapsed < self.summary_period_sec:
            return

        hz = self._summary_tick_count / max(elapsed, 1e-9)
        self.get_logger().info(
            f'[summary] tick_hz={hz:.2f}, '
            f'constraints={self._summary_last_constraints}, '
            f'total_ms={self._summary_last_total_ms:.2f}, '
            f'qp_ms={self._summary_last_qp_ms:.2f}'
        )
        self._summary_last_time = now
        self._summary_tick_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_controlled(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(self.kin.n_q)
        for i, jname in enumerate(CONTROLLED_JOINTS):
            if jname not in name_to_pos:
                self.get_logger().warn(
                    f'Joint {jname} missing from /joint_commands_unsafe, dropping',
                    throttle_duration_sec=2.0,
                )
                return None
            q[i] = name_to_pos[jname]
        return q
    
    def _capsule_center(self, a, b):
        return 0.5 * (a + b)

    def _pair_is_active_by_center_distance(self, a1, b1, a2, b2):
        if not self.enable_coarse_gating:
            return True
        c1 = self._capsule_center(a1, b1)
        c2 = self._capsule_center(a2, b2)
        d_center = np.linalg.norm(c1 - c2)
        return d_center < self.coarse_distance_activate\
    
    @staticmethod
    def _clip_vec_norm(v, max_norm):
        v = np.asarray(v, dtype=np.float64)
        if max_norm <= 0.0:
            return np.zeros_like(v)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-12:
            return np.zeros_like(v)
        if n > max_norm:
            return v * (max_norm / n)
        return v


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()