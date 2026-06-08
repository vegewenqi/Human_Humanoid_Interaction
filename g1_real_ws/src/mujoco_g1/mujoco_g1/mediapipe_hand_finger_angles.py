#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray


WRIST_IDX = 0
THUMB_CMC_IDX = 1
THUMB_MCP_IDX = 2
THUMB_TIP_IDX = 4
INDEX_MCP_IDX = 5
INDEX_PIP_IDX = 6
MIDDLE_MCP_IDX = 9
MIDDLE_PIP_IDX = 10
RING_MCP_IDX = 13
RING_PIP_IDX = 14
PINKY_MCP_IDX = 17
PINKY_PIP_IDX = 18

FINGER_DEFS = [
    ("index", INDEX_MCP_IDX, INDEX_PIP_IDX),
    ("middle", MIDDLE_MCP_IDX, MIDDLE_PIP_IDX),
    ("ring", RING_MCP_IDX, RING_PIP_IDX),
    ("pinky", PINKY_MCP_IDX, PINKY_PIP_IDX),
]

HANDEDNESS_FLIP = {"Left": "Right", "Right": "Left"}
_MEDIAPIPE_IMPORT_ERROR = (
    "MediaPipe/OpenCV hand perception is not compatible with NumPy 2.x in this ROS image. "
    "Install a NumPy 1.x runtime, for example: "
    "python3 -m pip install --force-reinstall 'numpy==1.26.4' 'mediapipe==0.10.14'"
)


def _check_numpy_runtime():
    try:
        numpy_major = int(str(np.__version__).split(".", 1)[0])
    except (TypeError, ValueError):
        numpy_major = 0

    if numpy_major >= 2:
        raise RuntimeError(f"{_MEDIAPIPE_IMPORT_ERROR}; current numpy={np.__version__}")


def _import_cv2():
    _check_numpy_runtime()
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "cv2 is required for mediapipe_hand_finger_angles. "
            "Install python3-opencv or rebuild the Docker image with docker/requirements.txt."
        ) from exc
    return cv2


def _import_mediapipe():
    _check_numpy_runtime()

    try:
        import mediapipe as mp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required for mediapipe_hand_finger_angles. "
            "Install it with: python3 -m pip install 'mediapipe==0.10.14'"
        ) from exc
    return mp


@dataclass
class FingerConfig:
    name: str
    min_deg: float
    max_deg: float
    mcp_idx: int = -1
    pip_idx: int = -1


@dataclass
class FingerAngles:
    angle_deg: float
    norm: float


@dataclass
class HandResult:
    handedness: str
    score: float
    landmarks: list


def _lm_to_vec(lm: Any) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return None
    return v / norm


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = _unit(a)
    bu = _unit(b)
    if au is None or bu is None:
        return 0.0
    return float(np.degrees(np.arccos(float(np.clip(np.dot(au, bu), -1.0, 1.0)))))


def compute_finger_bend_deg(lms: Any, mcp_idx: int, pip_idx: int) -> float:
    wrist = _lm_to_vec(lms[WRIST_IDX])
    mcp = _lm_to_vec(lms[mcp_idx])
    pip = _lm_to_vec(lms[pip_idx])
    return _angle_between_deg(mcp - wrist, pip - mcp)


def compute_thumb_bend_deg(lms: Any) -> float:
    cmc = _lm_to_vec(lms[THUMB_CMC_IDX])
    mcp = _lm_to_vec(lms[THUMB_MCP_IDX])
    tip = _lm_to_vec(lms[THUMB_TIP_IDX])
    return _angle_between_deg(mcp - cmc, tip - mcp)


def compute_thumb_rotation_deg(lms: Any) -> float:
    wrist = _lm_to_vec(lms[WRIST_IDX])
    index_mcp = _lm_to_vec(lms[INDEX_MCP_IDX])
    pinky_mcp = _lm_to_vec(lms[PINKY_MCP_IDX])
    cmc = _lm_to_vec(lms[THUMB_CMC_IDX])
    tip = _lm_to_vec(lms[THUMB_TIP_IDX])

    palm_normal = _unit(np.cross(index_mcp - wrist, pinky_mcp - wrist))
    index_ref = _unit(index_mcp - wrist)
    if palm_normal is None or index_ref is None:
        return 0.0

    thumb_dir = tip - cmc
    thumb_proj = _unit(thumb_dir - np.dot(thumb_dir, palm_normal) * palm_normal)
    if thumb_proj is None:
        return 0.0

    return float(np.degrees(np.arccos(float(np.clip(np.dot(thumb_proj, index_ref), -1.0, 1.0)))))


def normalize_angle(
    angle_deg: float,
    min_deg: float,
    max_deg: float,
    warn_cb: Optional[Callable[[str], None]] = None,
) -> float:
    span = max_deg - min_deg
    if span <= 0.0:
        if warn_cb is not None:
            warn_cb(
                f"normalize_angle: max_deg ({max_deg:.1f}) <= min_deg ({min_deg:.1f}); "
                "returning 0.0"
            )
        return 0.0
    return float(max(0.0, min(1.0, (angle_deg - min_deg) / span)))


def compute_all_finger_angles(
    lms: Any,
    configs: List[FingerConfig],
    warn_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, FingerAngles]:
    results: Dict[str, FingerAngles] = {}
    for cfg in configs:
        if cfg.name == "thumb_bend":
            angle = compute_thumb_bend_deg(lms)
        elif cfg.name == "thumb_rot":
            angle = compute_thumb_rotation_deg(lms)
        else:
            angle = compute_finger_bend_deg(lms, cfg.mcp_idx, cfg.pip_idx)
        norm = normalize_angle(angle, cfg.min_deg, cfg.max_deg, warn_cb)
        results[cfg.name] = FingerAngles(angle_deg=angle, norm=norm)
    return results


class LandmarkDetector:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        mp = _import_mediapipe()
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_mp_results = None

    def close(self):
        self._hands.close()

    def detect(self, rgb_image: np.ndarray) -> List[HandResult]:
        self._last_mp_results = self._hands.process(rgb_image)
        output: List[HandResult] = []
        if self._last_mp_results.multi_hand_landmarks and self._last_mp_results.multi_handedness:
            for hand_lm, hand_class in zip(
                self._last_mp_results.multi_hand_landmarks,
                self._last_mp_results.multi_handedness,
            ):
                classification = hand_class.classification[0]
                output.append(
                    HandResult(
                        handedness=classification.label,
                        score=float(classification.score),
                        landmarks=hand_lm.landmark,
                    )
                )
        return output

    def draw_on(self, bgr_image: np.ndarray) -> np.ndarray:
        annotated = bgr_image.copy()
        if self._last_mp_results is not None and self._last_mp_results.multi_hand_landmarks:
            for hand_lm in self._last_mp_results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )
        return annotated


class MediaPipeHandFingerAnglesNode(Node):
    """
    Convert ZED image frames through MediaPipe Hands into /hand_finger_angles.

    Output is std_msgs/Float32MultiArray with the same convention as the G1
    Inspire bridge:
        1.0 = open
        0.0 = fist
    """

    def __init__(self):
        super().__init__("mediapipe_hand_finger_angles")

        self.declare_parameter("image_topic", "/image/compressed")
        self.declare_parameter("compressed_image", True)
        self.declare_parameter("reliable_image_qos", True)
        self.declare_parameter("output_topic", "/hand_finger_angles")
        self.declare_parameter("output_layout", "finger12")  # finger10 | finger12 | hand2

        self.declare_parameter("max_num_hands", 2)
        self.declare_parameter("min_detection_confidence", 0.7)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("flip_handedness", True)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("debug_image_topic", "/hand_landmarks_debug_image")
        self.declare_parameter("debug_log", False)
        self.declare_parameter("debug_log_period_sec", 1.0)

        self.declare_parameter("startup_open_amount", 0.0)
        self.declare_parameter("hold_last_on_no_detection", True)
        self.declare_parameter("ring_follows_middle", False)
        self.declare_parameter("ema_alpha", 0.35)
        self.declare_parameter("max_delta_per_update", 0.08)

        defaults = {
            "index": (19.0, 120.0),
            "middle": (19.0, 120.0),
            "ring": (19.0, 120.0),
            "pinky": (19.0, 120.0),
            "thumb_bend": (10.0, 60.0),
            "thumb_rot": (25.0, 80.0),
        }
        for name, (lo, hi) in defaults.items():
            self.declare_parameter(f"{name}_min_angle_deg", lo)
            self.declare_parameter(f"{name}_max_angle_deg", hi)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.compressed_image = bool(self.get_parameter("compressed_image").value)
        self.reliable_image_qos = bool(self.get_parameter("reliable_image_qos").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.output_layout = str(self.get_parameter("output_layout").value)
        self.flip_handedness = bool(self.get_parameter("flip_handedness").value)
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.debug_log = bool(self.get_parameter("debug_log").value)
        self.debug_log_period_sec = float(self.get_parameter("debug_log_period_sec").value)
        self.startup_open_amount = float(self.get_parameter("startup_open_amount").value)
        self.hold_last_on_no_detection = bool(self.get_parameter("hold_last_on_no_detection").value)
        self.ring_follows_middle = bool(self.get_parameter("ring_follows_middle").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.max_delta_per_update = float(self.get_parameter("max_delta_per_update").value)

        self._validate_params()
        self._finger_configs = self._make_finger_configs()
        self._cv2 = _import_cv2()

        max_num_hands = int(self.get_parameter("max_num_hands").value)
        min_det_conf = float(self.get_parameter("min_detection_confidence").value)
        min_trk_conf = float(self.get_parameter("min_tracking_confidence").value)
        self._detector = LandmarkDetector(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_trk_conf,
        )
        self._bridge = CvBridge()

        initial6 = [self.startup_open_amount] * 6
        self.left6 = list(initial6)
        self.right6 = list(initial6)
        self.last_log_time = self.get_clock().now()

        image_qos = self._make_image_qos()
        if self.compressed_image:
            self.sub_image = self.create_subscription(
                CompressedImage, self.image_topic, self.on_image, image_qos
            )
        else:
            self.sub_image = self.create_subscription(Image, self.image_topic, self.on_image, image_qos)

        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, qos_profile_sensor_data)
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(
                Image, self.debug_image_topic, qos_profile_sensor_data
            )

        self.get_logger().info("MediaPipeHandFingerAnglesNode started.")
        self.get_logger().info(f"image_topic      = {self.image_topic}")
        self.get_logger().info(f"compressed_image = {self.compressed_image}")
        self.get_logger().info(f"output_topic     = {self.output_topic}")
        self.get_logger().info(f"output_layout    = {self.output_layout}")
        self.get_logger().info("value convention = 1.0 open, 0.0 fist")

    def _validate_params(self):
        if self.output_layout not in ("finger10", "finger12", "hand2"):
            raise ValueError("output_layout must be one of: finger10, finger12, hand2")
        if not (0.0 <= self.startup_open_amount <= 1.0):
            raise ValueError("startup_open_amount must be in [0, 1]")
        if self.debug_log_period_sec <= 0.0:
            raise ValueError("debug_log_period_sec must be > 0")
        if not (0.0 <= self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in [0, 1]")
        if self.max_delta_per_update < 0.0:
            raise ValueError("max_delta_per_update must be >= 0")

    def _make_image_qos(self) -> QoSProfile:
        if not self.reliable_image_qos:
            return qos_profile_sensor_data
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

    def _make_finger_configs(self) -> List[FingerConfig]:
        configs = [
            FingerConfig(
                name=name,
                min_deg=float(self.get_parameter(f"{name}_min_angle_deg").value),
                max_deg=float(self.get_parameter(f"{name}_max_angle_deg").value),
                mcp_idx=mcp,
                pip_idx=pip,
            )
            for name, mcp, pip in FINGER_DEFS
        ]
        for name in ("thumb_bend", "thumb_rot"):
            configs.append(
                FingerConfig(
                    name=name,
                    min_deg=float(self.get_parameter(f"{name}_min_angle_deg").value),
                    max_deg=float(self.get_parameter(f"{name}_max_angle_deg").value),
                )
            )
        return configs

    @staticmethod
    def _correct_handedness(label: str, flip: bool) -> str:
        return HANDEDNESS_FLIP.get(label, label) if flip else label

    def _decode_image(self, msg) -> Optional[np.ndarray]:
        try:
            if self.compressed_image:
                bgr = self._cv2.imdecode(np.frombuffer(msg.data, np.uint8), self._cv2.IMREAD_COLOR)
                if bgr is None:
                    self.get_logger().warn("Failed to decode compressed image", throttle_duration_sec=2.0)
                    return None
                return bgr
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"cv_bridge error: {exc}", throttle_duration_sec=2.0)
            return None

    def _open6_from_angles(self, angles: Dict[str, FingerAngles]) -> List[float]:
        thumb_bend = 1.0 - angles["thumb_bend"].norm
        thumb_rot = 1.0 - angles["thumb_rot"].norm
        index = 1.0 - angles["index"].norm
        middle = 1.0 - angles["middle"].norm
        ring = 1.0 - angles["ring"].norm
        pinky = 1.0 - angles["pinky"].norm
        if self.ring_follows_middle:
            ring = middle
        return [
            float(np.clip(thumb_bend, 0.0, 1.0)),
            float(np.clip(thumb_rot, 0.0, 1.0)),
            float(np.clip(index, 0.0, 1.0)),
            float(np.clip(middle, 0.0, 1.0)),
            float(np.clip(ring, 0.0, 1.0)),
            float(np.clip(pinky, 0.0, 1.0)),
        ]

    def _filter_open6(self, previous: List[float], raw: List[float]) -> List[float]:
        filtered: List[float] = []
        for prev, target in zip(previous, raw):
            smoothed = (1.0 - self.ema_alpha) * prev + self.ema_alpha * target
            if self.max_delta_per_update > 0.0:
                delta = float(np.clip(
                    smoothed - prev,
                    -self.max_delta_per_update,
                    self.max_delta_per_update,
                ))
                smoothed = prev + delta
            filtered.append(float(np.clip(smoothed, 0.0, 1.0)))
        return filtered

    @staticmethod
    def _finger5_from_open6(open6: List[float]) -> List[float]:
        thumb = min(open6[0], open6[1])
        return [thumb, open6[2], open6[3], open6[4], open6[5]]

    def _output_data(self) -> List[float]:
        if self.output_layout == "hand2":
            left = float(np.mean(self.left6))
            right = float(np.mean(self.right6))
            return [left, right]
        if self.output_layout == "finger10":
            return [float(x) for x in self._finger5_from_open6(self.left6) + self._finger5_from_open6(self.right6)]
        return [float(x) for x in self.left6 + self.right6]

    def on_image(self, msg):
        bgr = self._decode_image(msg)
        if bgr is None:
            return

        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        results = self._detector.detect(rgb)

        seen = {"Left": False, "Right": False}
        angle_debug: List[str] = []
        for result in results:
            handedness = self._correct_handedness(result.handedness, self.flip_handedness)
            if handedness not in seen or seen[handedness]:
                continue
            seen[handedness] = True

            angles = compute_all_finger_angles(
                result.landmarks,
                self._finger_configs,
                warn_cb=lambda text: self.get_logger().warn(text, throttle_duration_sec=2.0),
            )
            raw_open6 = self._open6_from_angles(angles)
            if handedness == "Left":
                self.left6 = self._filter_open6(self.left6, raw_open6)
                open6 = self.left6
            else:
                self.right6 = self._filter_open6(self.right6, raw_open6)
                open6 = self.right6
            angle_debug.append(
                f"{handedness}: "
                f"tb={open6[0]:.2f} tr={open6[1]:.2f} "
                f"i={open6[2]:.2f} m={open6[3]:.2f} r={open6[4]:.2f} p={open6[5]:.2f}"
            )

        if not self.hold_last_on_no_detection:
            fallback = [self.startup_open_amount] * 6
            if not seen["Left"]:
                self.left6 = list(fallback)
            if not seen["Right"]:
                self.right6 = list(fallback)

        out = Float32MultiArray()
        out.data = self._output_data()
        self.pub.publish(out)

        if self.debug_pub is not None:
            annotated = self._detector.draw_on(bgr)
            try:
                debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as exc:
                self.get_logger().warn(f"cv_bridge debug error: {exc}", throttle_duration_sec=2.0)

        if self.debug_log:
            now = self.get_clock().now()
            if (now - self.last_log_time).nanoseconds * 1e-9 >= self.debug_log_period_sec:
                detail = " | ".join(angle_debug) if angle_debug else "no hand detected; holding last output"
                self.get_logger().info(
                    f"[mediapipe_hand_finger_angles] out={np.round(out.data, 3).tolist()} | {detail}"
                )
                self.last_log_time = now

    def destroy_node(self):
        self._detector.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeHandFingerAnglesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
