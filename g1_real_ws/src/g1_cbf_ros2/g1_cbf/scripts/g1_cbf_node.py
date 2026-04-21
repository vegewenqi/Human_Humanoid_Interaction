#!/usr/bin/env python3
"""CBF safety filter node for G1 humanoid.

Supports:
- robot self-collision CBF
- robot vs human-capsule CBF
- optional robot vs box-obstacle CBF

Safe commands are published on /joint_commands at a fixed rate (1/dt Hz).
"""

import os
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
    # ('torso', 'right_upper_arm'),
    # ('torso', 'right_forearm_hand'),
    # ('torso', 'left_upper_arm'),
    # ('torso', 'left_forearm_hand'),
]


class G1CBFNode(Node):
    def __init__(self):
        super().__init__('g1_cbf_node')

        # ---------------- Parameters ----------------
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('rr_margin_phi', 0.0063)
        self.declare_parameter('hr_margin_phi', 0.028)
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

        # Set JAX config before importing cbf module
        use_gpu = self.get_parameter('use_gpu').value
        os.environ['JAX_PLATFORMS'] = 'cuda' if use_gpu else 'cpu'
        os.environ['JAX_ENABLE_X64'] = '1'
        from g1_cbf.cbf import DpaxCapsuleCBF, DpaxBoxCBF  # noqa: E402

        dt = self.get_parameter('dt').value
        rr_gamma = self.get_parameter('rr_gamma').value
        hr_gamma = self.get_parameter('hr_gamma').value
        rr_margin_phi = self.get_parameter('rr_margin_phi').value
        hr_margin_phi = self.get_parameter('hr_margin_phi').value
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

        if not urdf_path:
            self.get_logger().fatal('urdf_path parameter is required')
            raise RuntimeError('urdf_path not set')

        self.get_logger().info(f'Loading URDF: {urdf_path}')
        self.get_logger().info(
            f'CBF params: dt={dt}, rr_gamma={rr_gamma}, rr_margin_phi={rr_margin_phi}, hr_gamma={hr_gamma}, hr_margin_phi={hr_margin_phi}, self_geom={self.geom_type}'
        )

        # ---------------- Subsystems ----------------
        self.kin = G1Kinematics(urdf_path)
        self.get_logger().info('Initializing dpax CBFs (JAX JIT warmup)...')

        # Self-collision CBF
        if self.geom_type == 'boxes':
            beta = self.get_parameter('beta').value
            self.self_cbf = DpaxBoxCBF(gamma=rr_gamma, beta=beta)
        else:
            self.self_cbf = DpaxCapsuleCBF(
                gamma=rr_gamma, margin_phi=rr_margin_phi,
            )

        # Human-collision CBF always uses capsules
        self.human_cbf = DpaxCapsuleCBF(
            gamma=hr_gamma, margin_phi=hr_margin_phi,
        )

        # Optional box obstacle CBF
        beta = self.get_parameter('beta').value
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
        # dict[name] = {'a':(3,), 'b':(3,), 'radius':float}
        self.human_capsules = {}

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
            f'g1_cbf_node ready — publishing at {1.0/dt:.0f} Hz'
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

        caps = {}
        for i, name in enumerate(HUMAN_CAPSULE_NAMES):
            s = 7 * i
            block = data[s:s + 7]
            if not np.all(np.isfinite(block[:6])) or not np.isfinite(block[6]):
                continue

            caps[name] = {
                'a': block[0:3].copy(),
                'b': block[3:6].copy(),
                'radius': float(block[6]),
            }

        self.human_capsules = caps

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

        dt = self.get_parameter('dt').value
        K = self.get_parameter('K').value
        max_vel = self.get_parameter('max_velocity').value
        lpf = self.get_parameter('lpf_gain').value

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

        # Robot collider visualization
        self.viz.publish(self.get_clock().now().to_msg())

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
                    constraints, closest_points,
                )

        # human collision
        if self.enable_human_collision and self.human_capsules:
            self._build_human_capsule_constraints(
                constraints, closest_points,
            )

        # optional box obstacles
        if self.enable_box_obstacles and self._obstacles:
            self._build_obstacle_constraints(
                constraints, closest_points,
            )

        # Publish distance lines for all active constraints
        self.viz.publish_distances(
            self.get_clock().now().to_msg(), closest_points,
        )

        dq_safe = self.qp.solve(
            dq_ref, constraints,
            self.dq_min, self.dq_max,
        )

        self.q_cbf_target += dq_safe * dt

        # Prevent divergence
        max_lead = 0.5
        self.q_cbf_target = np.clip(
            self.q_cbf_target,
            q_ctrl - max_lead,
            q_ctrl + max_lead,
        )

        safe_msg = JointState()
        safe_msg.header.stamp = self.get_clock().now().to_msg()
        safe_msg.name = list(CONTROLLED_JOINTS)
        safe_msg.position = self.q_cbf_target.tolist()
        safe_msg.velocity = dq_safe.tolist()
        self.cmd_pub.publish(safe_msg)

    # ------------------------------------------------------------------
    # Constraint builders
    # ------------------------------------------------------------------

    def _build_capsule_constraints(self, constraints, closest_points):
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

        for nameA, nameB in COLLISION_PAIRS:
            eA, eB = endpoints[nameA], endpoints[nameB]

            phi, A_row, b_val, p1, p2 = self.self_cbf.build_constraint(
                eA['radius'], eA['a'], eA['b'],
                eA['J_a'], eA['J_b'],
                eB['radius'], eB['a'], eB['b'],
                eB['J_a'], eB['J_b'],
            )
            constraints.append((A_row, b_val))
            closest_points.append((p1, p2))

    def _build_human_capsule_constraints(self, constraints, closest_points):
        robot_endpoints = {}
        for name in self.kin.collision_bodies:
            a, b, J_a, J_b = self.kin.get_endpoint_jacobians(name)
            body = self.kin.collision_bodies[name]
            robot_endpoints[name] = {
                'a': a,
                'b': b,
                'J_a': J_a,
                'J_b': J_b,
                'radius': body['radius'],
            }

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

            # human capsule treated as static wrt robot command
            J_zero_a = np.zeros_like(eR['J_a'])
            J_zero_b = np.zeros_like(eR['J_b'])

            phi, A_row, b_val, p1, p2 = self.human_cbf.build_constraint(
                eR['radius'], eR['a'], eR['b'],
                eR['J_a'], eR['J_b'],
                eH['radius'], eH['a'], eH['b'],
                J_zero_a, J_zero_b,
            )
            constraints.append((A_row, b_val))
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


def main(args=None):
    rclpy.init(args=args)
    node = G1CBFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()