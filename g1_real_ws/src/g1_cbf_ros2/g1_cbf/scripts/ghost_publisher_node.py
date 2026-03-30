#!/usr/bin/env python3
"""Ghost robot publisher for visualizing unsafe commands.

Subscribes to /joint_commands_unsafe, republishes joint states
on /ghost/joint_states for a second robot_state_publisher to
visualize as a transparent "ghost" robot offset from the real one.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class GhostPublisherNode(Node):
    def __init__(self):
        super().__init__('ghost_publisher_node')

        self.declare_parameter('offset_y', -1.0)
        offset_y = self.get_parameter('offset_y').value

        self.js_pub = self.create_publisher(
            JointState, '/ghost/joint_states', 10,
        )

        self.create_subscription(
            JointState, '/joint_commands_unsafe',
            self._unsafe_cb, 10,
        )

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

    def _unsafe_cb(self, msg: JointState):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(msg.name)
        out.position = list(msg.position)
        self.js_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GhostPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
