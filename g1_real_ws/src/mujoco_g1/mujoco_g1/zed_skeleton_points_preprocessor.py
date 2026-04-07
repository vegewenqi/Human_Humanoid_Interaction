#!/usr/bin/env python3
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8, Float32MultiArray

from .components.utils import pc2_to_xyz_array
from .components.human_pose_preprocessor import HumanPosePreprocessor


class ZedSkeletonPointsPreprocessorNode(Node):
    """
    Read raw ZED skeleton PointCloud2, apply the same point-level preprocessing
    used by human angle estimation, and publish filtered skeleton points as
    Float32MultiArray:

        [x0, y0, z0, x1, y1, z1, ..., xN, yN, zN]

    Unit of published points: meters
    """

    def __init__(self):
        super().__init__("zed_skeleton_points_preprocessor")

        self.declare_parameter("input_points_topic", "/skeleton/points")
        self.declare_parameter("input_conf_topic", "/skeleton/confidence")
        self.declare_parameter("output_points_topic", "/skeleton/points_filtered")

        self.declare_parameter("min_confidence", 40)
        self.declare_parameter("point_ema_alpha", 1.0)
        self.declare_parameter("point_max_jump", 1.0)  # meters
        self.declare_parameter("point_max_reject_count", 5)

        self.input_points_topic = str(self.get_parameter("input_points_topic").value)
        self.input_conf_topic = str(self.get_parameter("input_conf_topic").value)
        self.output_points_topic = str(self.get_parameter("output_points_topic").value)

        self.min_confidence = int(self.get_parameter("min_confidence").value)
        self.point_ema_alpha = float(self.get_parameter("point_ema_alpha").value)
        self.point_max_jump = float(self.get_parameter("point_max_jump").value)
        self.point_max_reject_count = int(self.get_parameter("point_max_reject_count").value)


        self.pre = HumanPosePreprocessor(
            alpha=self.point_ema_alpha,
            max_jump=self.point_max_jump,
            max_reject_count=self.point_max_reject_count,
        )

        self.latest_conf: Optional[int] = None

        self.sub_points = self.create_subscription(
            PointCloud2, self.input_points_topic, self.on_points, 10
        )
        self.sub_conf = self.create_subscription(
            UInt8, self.input_conf_topic, self.on_conf, 10
        )
        self.pub_points = self.create_publisher(
            Float32MultiArray, self.output_points_topic, 10
        )

        self.get_logger().info("ZedSkeletonPointsPreprocessorNode started.")
        self.get_logger().info(f"input_points_topic  = {self.input_points_topic}")
        self.get_logger().info(f"input_conf_topic    = {self.input_conf_topic}")
        self.get_logger().info(f"output_points_topic = {self.output_points_topic}")
        self.get_logger().info(f"min_confidence      = {self.min_confidence}")
        self.get_logger().info(f"point_ema_alpha     = {self.point_ema_alpha}")
        self.get_logger().info(f"point_max_jump      = {self.point_max_jump}")

    def on_conf(self, msg: UInt8):
        self.latest_conf = int(msg.data)

    def on_points(self, msg: PointCloud2):
        conf = self.latest_conf if self.latest_conf is not None else -1
        if conf >= 0 and conf < self.min_confidence:
            return

        pts_xyz = pc2_to_xyz_array(msg)
        if pts_xyz is None or pts_xyz.size == 0:
            return

        pts_xyz = pts_xyz.astype(np.float64)
        pts_xyz *= 0.001  # mm -> m

        pts_filtered = np.full_like(pts_xyz, np.nan, dtype=np.float64)

        for idx in range(pts_xyz.shape[0]):
            p_f = self.pre.filter_point(pts_xyz, idx)
            if p_f is not None:
                pts_filtered[idx, :] = p_f

        out_msg = Float32MultiArray()
        out_msg.data = pts_filtered.astype(np.float32).reshape(-1).tolist()
        self.pub_points.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ZedSkeletonPointsPreprocessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()