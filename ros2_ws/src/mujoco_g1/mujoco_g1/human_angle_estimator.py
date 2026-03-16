import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8, Float32MultiArray

from .components.utils import pc2_to_xyz_array
from .components.human_pose_preprocessor import HumanPosePreprocessor
from .components.human_angle_estimator_core import HumanAngleEstimatorCore
from .components.angle_filter import AngleFilter
from .components import zed_indices as zi


class HumanAngleEstimatorNode(Node):
    def __init__(self):
        super().__init__("human_angle_estimator")

        self.declare_parameter("min_confidence", 40)
        self.declare_parameter("point_ema_alpha", 0.25)
        self.declare_parameter("point_max_jump", 0.12)
        self.declare_parameter("angle_ema_alpha", 0.25)
        self.declare_parameter("angle_max_rate_deg", 180.0)
        self.declare_parameter("publish_deg", True)

        self.min_confidence = int(self.get_parameter("min_confidence").value)
        self.publish_deg = bool(self.get_parameter("publish_deg").value)

        self.pre = HumanPosePreprocessor(
            alpha=float(self.get_parameter("point_ema_alpha").value),
            max_jump=float(self.get_parameter("point_max_jump").value),
        )
        self.core = HumanAngleEstimatorCore()
        self.af = AngleFilter(
            alpha=float(self.get_parameter("angle_ema_alpha").value),
            max_rate_deg=float(self.get_parameter("angle_max_rate_deg").value),
            dt=1.0 / 30.0,
        )

        self.latest_conf: Optional[int] = None

        self.index_map = {
            "pelvis": zi.PELVIS,
            "l_shoulder": zi.LEFT_SHOULDER,
            "r_shoulder": zi.RIGHT_SHOULDER,
            "l_elbow": zi.LEFT_ELBOW,
            "r_elbow": zi.RIGHT_ELBOW,
            "l_wrist": zi.LEFT_WRIST,
            "r_wrist": zi.RIGHT_WRIST,
        }

        self.sub_points = self.create_subscription(PointCloud2, "/skeleton/points", self.on_points, 10)
        self.sub_conf = self.create_subscription(UInt8, "/skeleton/confidence", self.on_conf, 10)

        self.pub = self.create_publisher(Float32MultiArray, "/human_joint_angles", 10)

        self.last_log_t = time.time()
        self.get_logger().info("HumanAngleEstimatorNode started.")

    def on_conf(self, msg: UInt8):
        self.latest_conf = int(msg.data)

    def on_points(self, msg: PointCloud2):
        conf = self.latest_conf if self.latest_conf is not None else -1
        if conf >= 0 and conf < self.min_confidence:
            return

        pts_xyz = pc2_to_xyz_array(msg).astype(np.float64)
        pts_xyz *= 0.001  # mm -> m
        if pts_xyz is None or pts_xyz.size == 0:
            return

        pts = self.pre.extract_points(pts_xyz, self.index_map)
        angles = self.core.estimate(pts)
        if angles is None:
            return

        angle_dict = {
            "torso_roll": angles.torso_roll,
            "torso_pitch": angles.torso_pitch,
            "l_sh_roll": angles.l_sh_roll,
            "l_el_pitch": angles.l_el_pitch,
            "r_sh_roll": angles.r_sh_roll,
            "r_el_pitch": angles.r_el_pitch,
        }

        angle_dict = self.af.update_dict(angle_dict)

        data = [
            angle_dict["torso_roll"],
            angle_dict["torso_pitch"],
            angle_dict["l_sh_roll"],
            angle_dict["l_el_pitch"],
            angle_dict["r_sh_roll"],
            angle_dict["r_el_pitch"],
        ]

        if self.publish_deg:
            data = [float(np.rad2deg(x)) for x in data]

        msg_out = Float32MultiArray()
        msg_out.data = data
        self.pub.publish(msg_out)

        now = time.time()
        if now - self.last_log_t > 1.0:
            unit = "deg" if self.publish_deg else "rad"
            self.get_logger().info(
                f"[{unit}] "
                f"torso_roll={data[0]:.2f}, torso_pitch={data[1]:.2f}, "
                f"l_sh_roll={data[2]:.2f}, l_el_pitch={data[3]:.2f}, "
                f"r_sh_roll={data[4]:.2f}, r_el_pitch={data[5]:.2f}"
            )
            self.last_log_t = now


def main(args=None):
    rclpy.init(args=args)
    node = HumanAngleEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()