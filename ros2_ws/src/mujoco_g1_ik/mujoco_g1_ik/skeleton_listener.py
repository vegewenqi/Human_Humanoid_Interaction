import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import UInt8, Float32MultiArray


class SkeletonListener(Node):
    # Indices of the skeleton joints to watch and print in logs
    WATCH_INDICES = [0, 17, 13, 15]

    def __init__(self):
        super().__init__('skeleton_listener')

        self.latest_conf = None
        self.latest_points = None
        self.latest_orientations = None

        self.create_subscription(
            PointCloud2,
            '/skeleton/points',
            self.on_points,
            10
        )
        self.create_subscription(
            UInt8,
            '/skeleton/confidence',
            self.on_confidence,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/skeleton/local_orientations',
            self.on_orientations,
            10
        )

        self.get_logger().info(
            f'Listening to /skeleton/points, /skeleton/confidence, /skeleton/local_orientations | '
            f'watch_indices={self.WATCH_INDICES}'
        )

    def on_confidence(self, msg: UInt8):
        self.latest_conf = int(msg.data)

    def on_points(self, msg: PointCloud2):
        pts = list(
            point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=False
            )
        )
        self.latest_points = pts
        self.try_print_selected()

    def on_orientations(self, msg: Float32MultiArray):
        data = list(msg.data)

        if len(data) % 4 != 0:
            self.get_logger().warn(
                f'/skeleton/local_orientations length={len(data)} is not divisible by 4'
            )
            return

        quats = []
        for i in range(0, len(data), 4):
            quats.append((data[i], data[i + 1], data[i + 2], data[i + 3]))

        self.latest_orientations = quats
        self.try_print_selected()

    def _fmt_xyz(self, p):
        if p is None:
            return 'None'
        x, y, z = p
        return f'({x:.4f}, {y:.4f}, {z:.4f})'

    def _fmt_quat(self, q):
        if q is None:
            return 'None'
        x, y, z, w = q
        return f'({x:.5f}, {y:.5f}, {z:.5f}, {w:.5f})'

    def _get_point(self, idx):
        if self.latest_points is None:
            return None
        if idx < 0 or idx >= len(self.latest_points):
            return None
        p = self.latest_points[idx]
        if p is None or len(p) != 3:
            return None
        if not all(math.isfinite(v) for v in p):
            return p
        return p

    def _get_quat(self, idx):
        if self.latest_orientations is None:
            return None
        if idx < 0 or idx >= len(self.latest_orientations):
            return None
        q = self.latest_orientations[idx]
        if q is None or len(q) != 4:
            return None
        if not all(math.isfinite(v) for v in q):
            return q
        return q

    def try_print_selected(self):
        conf_str = str(self.latest_conf) if self.latest_conf is not None else 'None'

        parts = [f'[conf={conf_str}]']

        for idx in self.WATCH_INDICES:
            xyz = self._get_point(idx)
            quat = self._get_quat(idx)
            parts.append(
                f'joint[{idx}]: xyz={self._fmt_xyz(xyz)}, quat={self._fmt_quat(quat)}'
            )

        self.get_logger().info(' | '.join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = SkeletonListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()