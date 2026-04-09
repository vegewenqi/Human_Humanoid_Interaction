import math
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, Float32MultiArray

from .components import zed_indices as zi


def is_valid_point(p: np.ndarray) -> bool:
    return p is not None and p.shape == (3,) and np.all(np.isfinite(p))


class HumanSkeletonCapsuleNode(Node):
    """
    Build human capsules from filtered skeleton points published as Float32MultiArray:
        [x0, y0, z0, x1, y1, z1, ..., xN, yN, zN]
    Unit of incoming points: meters

    Publish TWO versions:
      1) /human_capsules_zed
         absolute capsule endpoints in the incoming ZED frame
         (currently the publisher uses frame_id = "zed_world")

      2) /human_capsules_local
         human-local capsule endpoints, i.e. endpoints shifted by current human pelvis:
             p_local = p_zed - p_human_pelvis_zed

    Fixed order (10 primitives):
      0 torso
      1 left_upper_arm
      2 left_forearm_hand
      3 right_upper_arm
      4 right_forearm_hand
      5 left_thigh
      6 right_thigh
      7 left_shin
      8 right_shin
      9 head_sphere

    Flat layout:
      each primitive = [ax, ay, az, bx, by, bz, r]
      sphere is encoded as a == b == center
      total length = 10 * 7 = 70
    """

    CAPSULE_ORDER = [
        "torso",
        "left_upper_arm",
        "left_forearm_hand",
        "right_upper_arm",
        "right_forearm_hand",
        "left_thigh",
        "right_thigh",
        "left_shin",
        "right_shin",
        "head_sphere",
    ]

    def __init__(self):
        super().__init__("human_skeleton_capsule")

        self.declare_parameter("input_points_topic", "/skeleton/points_filtered")
        self.declare_parameter("input_conf_topic", "/skeleton/confidence")

        self.declare_parameter("min_confidence", 40)

        self.declare_parameter("torso_radius", 0.1)
        self.declare_parameter("upper_arm_radius", 0.05)
        self.declare_parameter("forearm_radius", 0.04)
        self.declare_parameter("thigh_radius", 0.065)
        self.declare_parameter("shin_radius", 0.055)
        self.declare_parameter("head_radius", 0.08)

        self.declare_parameter("capsule_zed_topic", "/human_capsules_zed")
        self.declare_parameter("capsule_local_topic", "/human_capsules_local")
        self.declare_parameter("pelvis_topic", "/human_pelvis_point_zed")

        self.input_points_topic = str(self.get_parameter("input_points_topic").value)
        self.input_conf_topic = str(self.get_parameter("input_conf_topic").value)

        self.min_confidence = int(self.get_parameter("min_confidence").value)

        self.torso_radius = float(self.get_parameter("torso_radius").value)
        self.upper_arm_radius = float(self.get_parameter("upper_arm_radius").value)
        self.forearm_radius = float(self.get_parameter("forearm_radius").value)
        self.thigh_radius = float(self.get_parameter("thigh_radius").value)
        self.shin_radius = float(self.get_parameter("shin_radius").value)
        self.head_radius = float(self.get_parameter("head_radius").value)

        self.capsule_zed_topic = str(self.get_parameter("capsule_zed_topic").value)
        self.capsule_local_topic = str(self.get_parameter("capsule_local_topic").value)
        self.pelvis_topic = str(self.get_parameter("pelvis_topic").value)

        self.latest_conf: Optional[int] = None

        self.sub_points = self.create_subscription(
            Float32MultiArray, self.input_points_topic, self.on_points, 10
        )
        self.sub_conf = self.create_subscription(
            UInt8, self.input_conf_topic, self.on_conf, 10
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
            "nose": zi.NOSE,
            "left_eye": zi.LEFT_EYE,
            "right_eye": zi.RIGHT_EYE,
            "left_ear": zi.LEFT_EAR,
            "right_ear": zi.RIGHT_EAR,
            "left_shoulder": zi.LEFT_SHOULDER,
            "right_shoulder": zi.RIGHT_SHOULDER,
            "left_elbow": zi.LEFT_ELBOW,
            "right_elbow": zi.RIGHT_ELBOW,
            "left_wrist": zi.LEFT_WRIST,
            "right_wrist": zi.RIGHT_WRIST,
            "left_hip": zi.LEFT_HIP,
            "right_hip": zi.RIGHT_HIP,
            "left_knee": zi.LEFT_KNEE,
            "right_knee": zi.RIGHT_KNEE,
            "left_ankle": zi.LEFT_ANKLE,
            "right_ankle": zi.RIGHT_ANKLE,
        }

        self.get_logger().info("HumanSkeletonCapsuleNode started.")
        self.get_logger().info(f"input_points_topic   = {self.input_points_topic}")
        self.get_logger().info(f"capsule_zed_topic    = {self.capsule_zed_topic}")
        self.get_logger().info(f"capsule_local_topic  = {self.capsule_local_topic}")
        self.get_logger().info(f"pelvis_topic         = {self.pelvis_topic}")

    def on_conf(self, msg: UInt8):
        self.latest_conf = int(msg.data)

    def _estimate_head_sphere(
        self, pts: Dict[str, Optional[np.ndarray]]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        neck = pts.get("neck")
        nose = pts.get("nose")
        left_eye = pts.get("left_eye")
        right_eye = pts.get("right_eye")
        left_ear = pts.get("left_ear")
        right_ear = pts.get("right_ear")

        candidates = []
        for p in [nose, left_eye, right_eye, left_ear, right_ear]:
            if is_valid_point(p):
                candidates.append(p)

        if len(candidates) == 0:
            if is_valid_point(neck):
                center = neck + np.array([0.0, 0.0, 0.18], dtype=np.float64)
                return (center, center, self.head_radius)
            return None

        center = np.mean(np.stack(candidates, axis=0), axis=0)

        if is_valid_point(neck):
            d = float(np.linalg.norm(center - neck))
            radius = max(self.head_radius, 0.4 * d)
        else:
            radius = self.head_radius

        return (center, center, radius)

    def _build_capsules_from_points(
        self, pts: Dict[str, Optional[np.ndarray]]
    ) -> Tuple[np.ndarray, Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]]:
        pelvis = pts.get("pelvis")
        neck = pts.get("neck")

        l_sh = pts.get("left_shoulder")
        r_sh = pts.get("right_shoulder")
        l_el = pts.get("left_elbow")
        r_el = pts.get("right_elbow")
        l_wr = pts.get("left_wrist")
        r_wr = pts.get("right_wrist")

        l_hip = pts.get("left_hip")
        r_hip = pts.get("right_hip")
        l_knee = pts.get("left_knee")
        r_knee = pts.get("right_knee")
        l_ankle = pts.get("left_ankle")
        r_ankle = pts.get("right_ankle")

        caps: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]] = {
            "torso": None,
            "left_upper_arm": None,
            "left_forearm_hand": None,
            "right_upper_arm": None,
            "right_forearm_hand": None,
            "left_thigh": None,
            "right_thigh": None,
            "left_shin": None,
            "right_shin": None,
            "head_sphere": None,
        }

        if is_valid_point(pelvis) and is_valid_point(neck):
            caps["torso"] = (pelvis, neck, self.torso_radius)

        if is_valid_point(l_sh) and is_valid_point(l_el):
            caps["left_upper_arm"] = (l_sh, l_el, self.upper_arm_radius)

        if is_valid_point(l_el) and is_valid_point(l_wr):
            caps["left_forearm_hand"] = (l_el, l_wr, self.forearm_radius)

        if is_valid_point(r_sh) and is_valid_point(r_el):
            caps["right_upper_arm"] = (r_sh, r_el, self.upper_arm_radius)

        if is_valid_point(r_el) and is_valid_point(r_wr):
            caps["right_forearm_hand"] = (r_el, r_wr, self.forearm_radius)

        if is_valid_point(l_hip) and is_valid_point(l_knee):
            caps["left_thigh"] = (l_hip, l_knee, self.thigh_radius)

        if is_valid_point(r_hip) and is_valid_point(r_knee):
            caps["right_thigh"] = (r_hip, r_knee, self.thigh_radius)

        if is_valid_point(l_knee) and is_valid_point(l_ankle):
            caps["left_shin"] = (l_knee, l_ankle, self.shin_radius)

        if is_valid_point(r_knee) and is_valid_point(r_ankle):
            caps["right_shin"] = (r_knee, r_ankle, self.shin_radius)

        caps["head_sphere"] = self._estimate_head_sphere(pts)

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

    def on_points(self, msg: Float32MultiArray):
        conf = self.latest_conf if self.latest_conf is not None else -1
        if conf >= 0 and conf < self.min_confidence:
            return

        data = np.asarray(msg.data, dtype=np.float64)
        if data.size == 0 or data.size % 3 != 0:
            self.get_logger().warn(
                f"Expected flat xyz array length multiple of 3, got {data.size}",
                throttle_duration_sec=2.0,
            )
            return

        pts_xyz = data.reshape(-1, 3)

        required_max_idx = max(self.index_map.values())
        if pts_xyz.shape[0] <= required_max_idx:
            self.get_logger().warn(
                f"Received {pts_xyz.shape[0]} filtered points, but need index up to {required_max_idx}.",
                throttle_duration_sec=2.0,
            )
            return

        pts = {}
        for name, idx in self.index_map.items():
            p = np.asarray(pts_xyz[idx], dtype=np.float64)
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                pts[name] = None
            else:
                pts[name] = p

        pelvis_zed, caps_zed = self._build_capsules_from_points(pts)
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