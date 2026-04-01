#!/usr/bin/env python3
"""Ghost robot publisher for visualizing unsafe commands.

Behavior:
- Subscribe to full /joint_states as the current full-body base state
- Subscribe to 8-DoF /joint_commands_unsafe as nominal upper-body command
- Publish full 29-DoF /ghost/joint_states for a second robot_state_publisher

This makes the ghost robot complete in RViz:
- uncontrolled joints come from current /joint_states
- controlled 8 joints are overridden by unsafe nominal commands
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

FULL_JOINT_MAP = {
    'left_hip_pitch_joint': 0,
    'left_hip_roll_joint': 1,
    'left_hip_yaw_joint': 2,
    'left_knee_joint': 3,
    'left_ankle_pitch_joint': 4,
    'left_ankle_roll_joint': 5,
    'right_hip_pitch_joint': 6,
    'right_hip_roll_joint': 7,
    'right_hip_yaw_joint': 8,
    'right_knee_joint': 9,
    'right_ankle_pitch_joint': 10,
    'right_ankle_roll_joint': 11,
    'waist_yaw_joint': 12,
    'waist_roll_joint': 13,
    'waist_pitch_joint': 14,
    'left_shoulder_pitch_joint': 15,
    'left_shoulder_roll_joint': 16,
    'left_shoulder_yaw_joint': 17,
    'left_elbow_joint': 18,
    'left_wrist_roll_joint': 19,
    'left_wrist_pitch_joint': 20,
    'left_wrist_yaw_joint': 21,
    'right_shoulder_pitch_joint': 22,
    'right_shoulder_roll_joint': 23,
    'right_shoulder_yaw_joint': 24,
    'right_elbow_joint': 25,
    'right_wrist_roll_joint': 26,
    'right_wrist_pitch_joint': 27,
    'right_wrist_yaw_joint': 28,
}
FULL_JOINT_NAMES = [name for name, _ in sorted(FULL_JOINT_MAP.items(), key=lambda kv: kv[1])]


class GhostPublisherNode(Node):
    def __init__(self):
        super().__init__('ghost_publisher_node')

        self.declare_parameter('offset_y', -1.0)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('unsafe_topic', '/joint_commands_unsafe')
        self.declare_parameter('ghost_topic', '/ghost/joint_states')

        offset_y = float(self.get_parameter('offset_y').value)
        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.unsafe_topic = str(self.get_parameter('unsafe_topic').value)
        self.ghost_topic = str(self.get_parameter('ghost_topic').value)

        self.js_pub = self.create_publisher(JointState, self.ghost_topic, 10)

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_cb,
            10,
        )

        self.create_subscription(
            JointState,
            self.unsafe_topic,
            self._unsafe_cb,
            10,
        )

        # Keep latest full current state as base
        self.latest_full_pos = None
        self.latest_full_vel = None
        self.has_full_state = False

        # Static TF: pelvis -> ghost/pelvis
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'pelvis'
        t.child_frame_id = 'ghost/pelvis'
        t.transform.translation.x = 0.0
        t.transform.translation.y = float(offset_y)
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f'Ghost publisher ready (offset_y={offset_y})'
        )

    def _joint_state_cb(self, msg: JointState):
        # Build full 29-DoF state from /joint_states by name
        name_to_idx = {name: i for i, name in enumerate(msg.name)}

        pos = [0.0] * len(FULL_JOINT_NAMES)
        vel = [0.0] * len(FULL_JOINT_NAMES)

        for j, name in enumerate(FULL_JOINT_NAMES):
            if name in name_to_idx:
                i = name_to_idx[name]
                if i < len(msg.position):
                    pos[j] = float(msg.position[i])
                if i < len(msg.velocity):
                    vel[j] = float(msg.velocity[i])

        self.latest_full_pos = pos
        self.latest_full_vel = vel
        self.has_full_state = True

    def _unsafe_cb(self, msg: JointState):
        # Need a full current state as base
        if not self.has_full_state or self.latest_full_pos is None:
            self.get_logger().warn(
                'No /joint_states received yet; cannot publish full ghost joint state.'
            )
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(FULL_JOINT_NAMES)

        # Start from current full state
        pos = list(self.latest_full_pos)
        vel = list(self.latest_full_vel) if self.latest_full_vel is not None else [0.0] * len(FULL_JOINT_NAMES)

        # Override only the commanded joints from /joint_commands_unsafe
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for name, full_idx in FULL_JOINT_MAP.items():
            if name in name_to_idx:
                i = name_to_idx[name]
                if i < len(msg.position):
                    pos[full_idx] = float(msg.position[i])
                if i < len(msg.velocity):
                    vel[full_idx] = float(msg.velocity[i])

        out.position = pos
        out.velocity = vel
        out.effort = []

        self.js_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GhostPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()