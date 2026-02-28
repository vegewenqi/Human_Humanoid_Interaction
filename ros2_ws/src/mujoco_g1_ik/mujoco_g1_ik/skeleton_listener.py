import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8
from sensor_msgs_py import point_cloud2


class SkeletonListener(Node):
    def __init__(self):
        super().__init__('skeleton_listener')
        self.conf = None

        self.sub_points = self.create_subscription(
            PointCloud2, '/skeleton/points', self.on_points, 10
        )
        self.sub_conf = self.create_subscription(
            UInt8, '/skeleton/confidence', self.on_conf, 10
        )

        self.get_logger().info("Listening to /skeleton/points and /skeleton/confidence")

    def on_conf(self, msg: UInt8):
        self.conf = int(msg.data)

    def on_points(self, msg: PointCloud2):
        # Read xyz points
        pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False))
        n = len(pts)
        c = self.conf if self.conf is not None else -1

        # Print only occasionally to avoid spamming terminal
        if n > 0:
            p0 = pts[0]
            plast = pts[-1]
            self.get_logger().info(
                f"Got {n} joints, conf={c}, p0={p0}, plast={plast}, frame={msg.header.frame_id}"
            )


def main():
    rclpy.init()
    node = SkeletonListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()