import math
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import UInt8, Float32MultiArray

from .components.utils import pc2_to_xyz_array
from .components import zed_indices as zi


def is_valid_point(p: np.ndarray) -> bool:
    return p.shape == (3,) and np.all(np.isfinite(p))


class HumanSkeletonCapsuleNode(Node):
    """
    Build human capsules directly from ZED BODY_38 skeleton points.

    Publish TWO versions:
      1) /human_capsules_zed
         absolute capsule endpoints in the incoming ZED frame
         (currently the publisher uses frame_id = "zed_world")

      2) /human_capsules_local
         human-local capsule endpoints, i.e. endpoints shifted by current human pelvis:
             p_local = p_zed - p_human_pelvis_zed

    Fixed order (7 capsules):
      0 torso
      1 left_upper_arm
      2 left_forearm_hand
      3 right_upper_arm
      4 right_forearm_hand
      5 left_thigh
      6 right_thigh

    Capsule flat layout:
      each capsule = [ax, ay, az, bx, by, bz, r]
      total length = 7 * 7 = 49
    """

    CAPSULE_ORDER = [
        "torso",
        "left_upper_arm",
        "left_forearm_hand",
        "right_upper_arm",
        "right_forearm_hand",
        "left_thigh",
        "right_thigh",
    ]

    def __init__(self):
        super().__init__("human_skeleton_capsule")

        self.declare_parameter("min_confidence", 40)

        self.declare_parameter("torso_radius", 0.1)
        self.declare_parameter("upper_arm_radius", 0.05)
        self.declare_parameter("forearm_radius", 0.04)
        self.declare_parameter("thigh_radius", 0.065)

        self.declare_parameter("capsule_zed_topic", "/human_capsules_zed")
        self.declare_parameter("capsule_local_topic", "/human_capsules_local")
        self.declare_parameter("pelvis_topic", "/human_pelvis_point_zed")

        self.min_confidence = int(self.get_parameter("min_confidence").value)

        self.torso_radius = float(self.get_parameter("torso_radius").value)
        self.upper_arm_radius = float(self.get_parameter("upper_arm_radius").value)
        self.forearm_radius = float(self.get_parameter("forearm_radius").value)
        self.thigh_radius = float(self.get_parameter("thigh_radius").value)

        self.capsule_zed_topic = str(self.get_parameter("capsule_zed_topic").value)
        self.capsule_local_topic = str(self.get_parameter("capsule_local_topic").value)
        self.pelvis_topic = str(self.get_parameter("pelvis_topic").value)

        self.latest_conf: Optional[int] = None

        self.sub_points = self.create_subscription(
            PointCloud2, "/skeleton/points", self.on_points, 10
        )
        self.sub_conf = self.create_subscription(
            UInt8, "/skeleton/confidence", self.on_conf, 10
        )

        self.pub_caps_zed = self.create_publisher(
            Float32MultiArray, self.capsule_zed_topic, 10
        )
        self.pub_caps_local = self.create_publisher(
            Float32MultiArray, self.capsule_local_topic, 10
        )
        self.pub_pelvis = self.create_publisher(
            Float32MultiArray, self.pelvis_topic, 10
        )

        self.index_map = {
            "pelvis": zi.PELVIS,
            "neck": zi.NECK,
            "left_shoulder": zi.LEFT_SHOULDER,
            "right_shoulder": zi.RIGHT_SHOULDER,
            "left_elbow": zi.LEFT_ELBOW,
            "right_elbow": zi.RIGHT_ELBOW,
            "left_wrist": zi.LEFT_WRIST,
            "right_wrist": zi.RIGHT_WRIST,
            "left_middle_tip": zi.LEFT_MIDDLE_TIP,
            "right_middle_tip": zi.RIGHT_MIDDLE_TIP,
            "left_hip": zi.LEFT_HIP,
            "right_hip": zi.RIGHT_HIP,
            "left_knee": zi.LEFT_KNEE,
            "right_knee": zi.RIGHT_KNEE,
        }

        self.get_logger().info("HumanSkeletonCapsuleNode started.")
        self.get_logger().info(f"capsule_zed_topic   = {self.capsule_zed_topic}")
        self.get_logger().info(f"capsule_local_topic = {self.capsule_local_topic}")
        self.get_logger().info(f"pelvis_topic        = {self.pelvis_topic}")

    def on_conf(self, msg: UInt8):
        self.latest_conf = int(msg.data)

    def _arm_distal_point(self, wrist: np.ndarray, tip: np.ndarray) -> Optional[np.ndarray]:
        if is_valid_point(tip):
            return tip
        if is_valid_point(wrist):
            return wrist
        return None

    def _build_capsules_zed(
        self, pts_xyz: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]]:
        def getp(name: str) -> np.ndarray:
            return pts_xyz[self.index_map[name], :]

        pelvis = getp("pelvis")
        neck = getp("neck")

        l_sh = getp("left_shoulder")
        r_sh = getp("right_shoulder")
        l_el = getp("left_elbow")
        r_el = getp("right_elbow")
        l_wr = getp("left_wrist")
        r_wr = getp("right_wrist")
        l_tip = getp("left_middle_tip")
        r_tip = getp("right_middle_tip")

        l_hip = getp("left_hip")
        r_hip = getp("right_hip")
        l_knee = getp("left_knee")
        r_knee = getp("right_knee")

        # ToDo: maybe forearm defines from elbow to wrist instead of middle fingertip?
        l_hand = self._arm_distal_point(l_wr, l_tip)
        r_hand = self._arm_distal_point(r_wr, r_tip)

        caps: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]] = {
            "torso": None,
            "left_upper_arm": None,
            "left_forearm_hand": None,
            "right_upper_arm": None,
            "right_forearm_hand": None,
            "left_thigh": None,
            "right_thigh": None,
        }

        if is_valid_point(pelvis) and is_valid_point(neck):
            caps["torso"] = (pelvis, neck, self.torso_radius)

        if is_valid_point(l_sh) and is_valid_point(l_el):
            caps["left_upper_arm"] = (l_sh, l_el, self.upper_arm_radius)

        if is_valid_point(l_el) and l_hand is not None:
            caps["left_forearm_hand"] = (l_el, l_hand, self.forearm_radius)

        if is_valid_point(r_sh) and is_valid_point(r_el):
            caps["right_upper_arm"] = (r_sh, r_el, self.upper_arm_radius)

        if is_valid_point(r_el) and r_hand is not None:
            caps["right_forearm_hand"] = (r_el, r_hand, self.forearm_radius)

        if is_valid_point(l_hip) and is_valid_point(l_knee):
            caps["left_thigh"] = (l_hip, l_knee, self.thigh_radius)

        if is_valid_point(r_hip) and is_valid_point(r_knee):
            caps["right_thigh"] = (r_hip, r_knee, self.thigh_radius)

        return pelvis, caps

    def _caps_to_flat_zed(
        self, caps_zed: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]
    ) -> list:
        arr = []
        for name in self.CAPSULE_ORDER:
            item = caps_zed[name]
            if item is None:
                arr.extend([math.nan] * 7)
            else:
                a, b, r = item
                arr.extend([
                    float(a[0]), float(a[1]), float(a[2]),
                    float(b[0]), float(b[1]), float(b[2]),
                    float(r),
                ])
        return arr

    def _caps_to_flat_local(
        self,
        pelvis_zed: np.ndarray,
        caps_zed: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]
    ) -> list:
        arr = []
        for name in self.CAPSULE_ORDER:
            item = caps_zed[name]
            if item is None:
                arr.extend([math.nan] * 7)
            else:
                a, b, r = item
                a_local = a - pelvis_zed
                b_local = b - pelvis_zed
                arr.extend([
                    float(a_local[0]), float(a_local[1]), float(a_local[2]),
                    float(b_local[0]), float(b_local[1]), float(b_local[2]),
                    float(r),
                ])
        return arr

    def on_points(self, msg: PointCloud2):
        conf = self.latest_conf if self.latest_conf is not None else -1
        if conf >= 0 and conf < self.min_confidence:
            return

        pts_xyz = pc2_to_xyz_array(msg)
        if pts_xyz is None or pts_xyz.size == 0:
            return

        pts_xyz = pts_xyz.astype(np.float64)
        pts_xyz *= 0.001  # mm -> m

        required_max_idx = max(self.index_map.values())
        if pts_xyz.shape[0] <= required_max_idx:
            self.get_logger().warn(
                f"Received {pts_xyz.shape[0]} points, but need index up to {required_max_idx}."
            )
            return

        pelvis_zed, caps_zed = self._build_capsules_zed(pts_xyz)
        if not is_valid_point(pelvis_zed):
            return

        msg_zed = Float32MultiArray()
        msg_zed.data = self._caps_to_flat_zed(caps_zed)
        self.pub_caps_zed.publish(msg_zed)

        msg_local = Float32MultiArray()
        msg_local.data = self._caps_to_flat_local(pelvis_zed, caps_zed)
        self.pub_caps_local.publish(msg_local)

        msg_pelvis = Float32MultiArray()
        msg_pelvis.data = [float(pelvis_zed[0]), float(pelvis_zed[1]), float(pelvis_zed[2])]
        self.pub_pelvis.publish(msg_pelvis)


def main(args=None):
    rclpy.init(args=args)
    node = HumanSkeletonCapsuleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()