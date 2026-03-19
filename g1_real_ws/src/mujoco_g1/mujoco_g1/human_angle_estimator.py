import time
from typing import Optional, List

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
        self.declare_parameter("point_ema_alpha", 1.0)
        self.declare_parameter("point_max_jump", 1.0)  # in meters
        self.declare_parameter("angle_ema_alpha", 1.0)
        self.declare_parameter("angle_max_rate_deg", 360.0)
        self.declare_parameter("publish_deg", True)

        # neutral calibration related
        self.declare_parameter("enable_neutral_calibration", True)
        self.declare_parameter("neutral_calibration_duration", 10.0)   # seconds
        self.declare_parameter("log_output", "raw")  # "raw" | "delta" | "both"

        self.min_confidence = int(self.get_parameter("min_confidence").value)
        self.publish_deg = bool(self.get_parameter("publish_deg").value)

        self.enable_neutral_calibration = bool(
            self.get_parameter("enable_neutral_calibration").value
        )
        self.neutral_calibration_duration = float(
            self.get_parameter("neutral_calibration_duration").value
        )
        self.log_output = str(self.get_parameter("log_output").value).strip().lower()
        if self.log_output not in ["raw", "delta", "both"]:
            self.get_logger().warn(
                f"Invalid log_output='{self.log_output}', fallback to 'both'."
            )
            self.log_output = "both"

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

        self.sub_points = self.create_subscription(
            PointCloud2, "/skeleton/points", self.on_points, 10
        )
        self.sub_conf = self.create_subscription(
            UInt8, "/skeleton/confidence", self.on_conf, 10
        )

        self.pub = self.create_publisher(Float32MultiArray, "/human_joint_angles", 10)
        self.pub_delta = self.create_publisher(
            Float32MultiArray, "/human_joint_angles_delta", 10
        )

        self.last_log_t = time.time()

        # neutral calibration state
        self.calib_start_t: Optional[float] = None
        self.calib_samples: List[np.ndarray] = []
        self.neutral_offset: Optional[np.ndarray] = None
        self.calib_done = not self.enable_neutral_calibration

        if self.calib_done:
            self.get_logger().info(
                "HumanAngleEstimatorNode started. Neutral calibration disabled."
            )
        else:
            self.get_logger().info(
                f"HumanAngleEstimatorNode started. Neutral calibration enabled "
                f"({self.neutral_calibration_duration:.2f}s). Please stand in neutral pose."
            )

    def on_conf(self, msg: UInt8):
        self.latest_conf = int(msg.data)

    def _dict_to_array(self, angle_dict) -> np.ndarray:
        return np.array(
            [
                angle_dict["torso_roll"],
                angle_dict["torso_pitch"],
                angle_dict["l_sh_roll"],
                angle_dict["l_el_pitch"],
                angle_dict["r_sh_roll"],
                angle_dict["r_el_pitch"],
            ],
            dtype=np.float64,
        )

    def _convert_unit_for_publish(self, arr: np.ndarray) -> list:
        if self.publish_deg:
            return [float(np.rad2deg(x)) for x in arr]
        return [float(x) for x in arr]

    def _format_angle_line(self, data: list, prefix: str, unit: str) -> str:
        return (
            f"{prefix}[{unit}] "
            f"torso_roll={data[0]:.2f}, torso_pitch={data[1]:.2f}, "
            f"l_sh_roll={data[2]:.2f}, l_el_pitch={data[3]:.2f}, "
            f"r_sh_roll={data[4]:.2f}, r_el_pitch={data[5]:.2f}"
        )

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

        # filtered raw angles (internal unit: rad)
        angle_dict = self.af.update_dict(angle_dict)
        raw_arr = self._dict_to_array(angle_dict)

        now = time.time()

        # neutral calibration
        if not self.calib_done:
            if self.calib_start_t is None:
                self.calib_start_t = now

            elapsed = now - self.calib_start_t
            self.calib_samples.append(raw_arr.copy())

            if elapsed >= self.neutral_calibration_duration:
                if len(self.calib_samples) > 0:
                    self.neutral_offset = np.mean(np.stack(self.calib_samples, axis=0), axis=0)
                else:
                    self.neutral_offset = np.zeros_like(raw_arr)

                self.calib_done = True

                neutral_disp = self._convert_unit_for_publish(self.neutral_offset)
                unit = "deg" if self.publish_deg else "rad"
                self.get_logger().info("Neutral calibration completed.")
                self.get_logger().info(
                    self._format_angle_line(neutral_disp, prefix="neutral_", unit=unit)
                )

        # delta angles
        if self.neutral_offset is not None:
            delta_arr = raw_arr - self.neutral_offset
        else:
            delta_arr = np.zeros_like(raw_arr)

        # publish raw
        raw_data = self._convert_unit_for_publish(raw_arr)
        msg_out = Float32MultiArray()
        msg_out.data = raw_data
        self.pub.publish(msg_out)

        # publish delta
        delta_data = self._convert_unit_for_publish(delta_arr)
        msg_delta = Float32MultiArray()
        msg_delta.data = delta_data
        self.pub_delta.publish(msg_delta)

        # logging
        if now - self.last_log_t > 1.0:
            unit = "deg" if self.publish_deg else "rad"

            if not self.calib_done:
                elapsed = 0.0 if self.calib_start_t is None else (now - self.calib_start_t)
                self.get_logger().info(
                    f"Neutral calibrating... {elapsed:.2f}/{self.neutral_calibration_duration:.2f}s"
                )

            if self.log_output in ["raw", "both"]:
                self.get_logger().info(
                    self._format_angle_line(raw_data, prefix="raw_", unit=unit)
                )

            if self.log_output in ["delta", "both"]:
                self.get_logger().info(
                    self._format_angle_line(delta_data, prefix="delta_", unit=unit)
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