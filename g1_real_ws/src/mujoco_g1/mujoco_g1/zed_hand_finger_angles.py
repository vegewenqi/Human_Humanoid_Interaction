#!/usr/bin/env python3
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8

from .components import zed_indices as zi


@dataclass
class HandSideConfig:
    name: str
    wrist_idx: int
    elbow_idx: int
    finger_indices: List[int]
    open_score_ref: float
    fist_score_ref: float


class SideGestureState:
    def __init__(self, initial_state: float, ema_alpha: float):
        self.state = float(initial_state)
        self.closure_ema: Optional[float] = None
        self.last_valid_time: Optional[float] = None
        self.ema_alpha = float(ema_alpha)

    def update_closure(self, closure: float) -> float:
        closure = float(np.clip(closure, 0.0, 1.0))
        if self.closure_ema is None:
            self.closure_ema = closure
        else:
            self.closure_ema = (
                self.ema_alpha * closure + (1.0 - self.ema_alpha) * self.closure_ema
            )
        return float(self.closure_ema)


class ZedHandFingerAnglesNode(Node):
    """
    Estimate simple hand closure from ZED BODY_38 hand points.

    Output topic:
        /hand_finger_angles std_msgs/Float32MultiArray

    Default output layout:
        finger10:
        [
            left_thumb, left_index, left_middle, left_ring, left_pinky,
            right_thumb, right_index, right_middle, right_ring, right_pinky,
        ]

    Value convention:
        1.0 = open
        0.0 = fist
    """

    def __init__(self):
        super().__init__("zed_hand_finger_angles")

        self.declare_parameter("input_points_topic", "/skeleton/points_filtered")
        self.declare_parameter("input_conf_topic", "/skeleton/confidence")
        self.declare_parameter("output_topic", "/hand_finger_angles")

        self.declare_parameter("min_confidence", 40)
        self.declare_parameter("min_valid_hand_points", 3)
        self.declare_parameter("min_scale_m", 0.05)
        self.declare_parameter("fallback_scale_m", 0.25)

        # These references should be tuned from real ZED observations.
        self.declare_parameter("open_score_ref", 0.45)
        self.declare_parameter("fist_score_ref", 0.20)
        self.declare_parameter("left_open_score_ref", -1.0)
        self.declare_parameter("left_fist_score_ref", -1.0)
        self.declare_parameter("right_open_score_ref", -1.0)
        self.declare_parameter("right_fist_score_ref", -1.0)

        self.declare_parameter("spread_weight", 0.60)
        self.declare_parameter("extension_weight", 0.40)
        self.declare_parameter("closure_ema_alpha", 0.30)

        # Hysteresis thresholds are applied after closure smoothing.
        self.declare_parameter("open_threshold", 0.35)
        self.declare_parameter("fist_threshold", 0.65)

        self.declare_parameter("publish_continuous_open_amount", False)
        self.declare_parameter("missing_timeout_sec", 0.50)
        # Internal closure state used when the hand estimate times out:
        # 1.0 = fist, 0.0 = open. This also sets the startup state before the
        # first valid hand estimate.
        self.declare_parameter("timeout_state", 1.0)
        self.declare_parameter("hold_last_on_timeout", True)

        # finger10 is the default bridge-friendly layout. hand2 is kept for
        # quick testing with one open amount per hand.
        self.declare_parameter("output_layout", "finger10")  # finger10 | finger12 | hand2
        self.declare_parameter("ring_follows_middle", True)

        self.declare_parameter("debug_log", False)
        self.declare_parameter("debug_log_period_sec", 1.0)

        self.input_points_topic = str(self.get_parameter("input_points_topic").value)
        self.input_conf_topic = str(self.get_parameter("input_conf_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.min_confidence = int(self.get_parameter("min_confidence").value)
        self.min_valid_hand_points = int(self.get_parameter("min_valid_hand_points").value)
        self.min_scale_m = float(self.get_parameter("min_scale_m").value)
        self.fallback_scale_m = float(self.get_parameter("fallback_scale_m").value)

        self.spread_weight = float(self.get_parameter("spread_weight").value)
        self.extension_weight = float(self.get_parameter("extension_weight").value)
        self.closure_ema_alpha = float(self.get_parameter("closure_ema_alpha").value)
        self.open_threshold = float(self.get_parameter("open_threshold").value)
        self.fist_threshold = float(self.get_parameter("fist_threshold").value)
        self.publish_continuous_open_amount = bool(
            self.get_parameter("publish_continuous_open_amount").value
        )
        self.missing_timeout_sec = float(self.get_parameter("missing_timeout_sec").value)
        self.timeout_state = float(self.get_parameter("timeout_state").value)
        self.hold_last_on_timeout = bool(self.get_parameter("hold_last_on_timeout").value)
        self.output_layout = str(self.get_parameter("output_layout").value)
        self.ring_follows_middle = bool(self.get_parameter("ring_follows_middle").value)
        self.debug_log = bool(self.get_parameter("debug_log").value)
        self.debug_log_period_sec = float(self.get_parameter("debug_log_period_sec").value)

        self._validate_params()

        self.left_cfg = self._make_side_config("left")
        self.right_cfg = self._make_side_config("right")

        self.left_state = SideGestureState(
            initial_state=self.timeout_state,
            ema_alpha=self.closure_ema_alpha,
        )
        self.right_state = SideGestureState(
            initial_state=self.timeout_state,
            ema_alpha=self.closure_ema_alpha,
        )

        self.latest_conf: Optional[int] = None
        self.last_log_time = self.get_clock().now()

        self.sub_points = self.create_subscription(
            Float32MultiArray, self.input_points_topic, self.on_points, 10
        )
        self.sub_conf = self.create_subscription(
            UInt8, self.input_conf_topic, self.on_conf, 10
        )
        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        self.get_logger().info("ZedHandFingerAnglesNode started.")
        self.get_logger().info(f"input_points_topic = {self.input_points_topic}")
        self.get_logger().info(f"output_topic       = {self.output_topic}")
        self.get_logger().info(f"output_layout      = {self.output_layout}")
        self.get_logger().info(
            "ZED hand indices: "
            f"L thumb={zi.LEFT_THUMB_TIP}, L index={zi.LEFT_INDEX_KNUCKLE}, "
            f"L middle={zi.LEFT_MIDDLE_TIP}, L pinky={zi.LEFT_PINKY_KNUCKLE}; "
            f"R thumb={zi.RIGHT_THUMB_TIP}, R index={zi.RIGHT_INDEX_KNUCKLE}, "
            f"R middle={zi.RIGHT_MIDDLE_TIP}, R pinky={zi.RIGHT_PINKY_KNUCKLE}"
        )

    def _validate_params(self):
        if self.min_valid_hand_points < 2:
            raise ValueError("min_valid_hand_points must be >= 2")
        if self.fallback_scale_m <= 0.0:
            raise ValueError("fallback_scale_m must be > 0")
        if not (0.0 <= self.closure_ema_alpha <= 1.0):
            raise ValueError("closure_ema_alpha must be in [0, 1]")
        if not (0.0 <= self.timeout_state <= 1.0):
            raise ValueError("timeout_state must be in [0, 1]")
        if self.open_threshold >= self.fist_threshold:
            raise ValueError("open_threshold must be smaller than fist_threshold")
        if self.output_layout not in ("finger10", "finger12", "hand2"):
            raise ValueError("output_layout must be one of: finger10, finger12, hand2")

    def _make_side_config(self, side: str) -> HandSideConfig:
        global_open = float(self.get_parameter("open_score_ref").value)
        global_fist = float(self.get_parameter("fist_score_ref").value)
        side_open = float(self.get_parameter(f"{side}_open_score_ref").value)
        side_fist = float(self.get_parameter(f"{side}_fist_score_ref").value)

        open_ref = side_open if side_open > 0.0 else global_open
        fist_ref = side_fist if side_fist > 0.0 else global_fist
        if open_ref <= fist_ref:
            raise ValueError(f"{side} open_score_ref must be greater than fist_score_ref")

        if side == "left":
            return HandSideConfig(
                name="left",
                wrist_idx=zi.LEFT_WRIST,
                elbow_idx=zi.LEFT_ELBOW,
                finger_indices=[
                    zi.LEFT_THUMB_TIP,
                    zi.LEFT_INDEX_KNUCKLE,
                    zi.LEFT_MIDDLE_TIP,
                    zi.LEFT_PINKY_KNUCKLE,
                ],
                open_score_ref=open_ref,
                fist_score_ref=fist_ref,
            )

        return HandSideConfig(
            name="right",
            wrist_idx=zi.RIGHT_WRIST,
            elbow_idx=zi.RIGHT_ELBOW,
            finger_indices=[
                zi.RIGHT_THUMB_TIP,
                zi.RIGHT_INDEX_KNUCKLE,
                zi.RIGHT_MIDDLE_TIP,
                zi.RIGHT_PINKY_KNUCKLE,
            ],
            open_score_ref=open_ref,
            fist_score_ref=fist_ref,
        )

    def on_conf(self, msg: UInt8):
        self.latest_conf = int(msg.data)

    @staticmethod
    def _valid_point(pts_xyz: np.ndarray, idx: int) -> Optional[np.ndarray]:
        if idx < 0 or idx >= pts_xyz.shape[0]:
            return None
        p = np.asarray(pts_xyz[idx], dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return None
        return p

    @staticmethod
    def _mean_pairwise_distance(points: List[np.ndarray]) -> float:
        distances = [np.linalg.norm(a - b) for a, b in combinations(points, 2)]
        if not distances:
            return 0.0
        return float(np.mean(distances))

    def _estimate_closure(
        self,
        pts_xyz: np.ndarray,
        cfg: HandSideConfig,
    ) -> Optional[Dict[str, float]]:
        wrist = self._valid_point(pts_xyz, cfg.wrist_idx)
        elbow = self._valid_point(pts_xyz, cfg.elbow_idx)
        if wrist is None:
            return None

        finger_points = []
        for idx in cfg.finger_indices:
            p = self._valid_point(pts_xyz, idx)
            if p is not None:
                finger_points.append(p)

        if len(finger_points) < self.min_valid_hand_points:
            return None

        scale = self.fallback_scale_m
        if elbow is not None:
            forearm_len = float(np.linalg.norm(wrist - elbow))
            if forearm_len >= self.min_scale_m:
                scale = forearm_len

        spread = self._mean_pairwise_distance(finger_points) / scale
        extension = float(np.mean([np.linalg.norm(p - wrist) for p in finger_points])) / scale
        open_score = self.spread_weight * spread + self.extension_weight * extension

        denom = cfg.open_score_ref - cfg.fist_score_ref
        open_amount = np.clip((open_score - cfg.fist_score_ref) / denom, 0.0, 1.0)
        closure = float(1.0 - open_amount)

        return {
            "closure": closure,
            "open_score": float(open_score),
            "spread": float(spread),
            "extension": float(extension),
            "valid_points": float(len(finger_points)),
        }

    def _update_side(
        self,
        side_state: SideGestureState,
        estimate: Optional[Dict[str, float]],
        now_sec: float,
    ) -> float:
        if estimate is None:
            timed_out = (
                side_state.last_valid_time is None
                or now_sec - side_state.last_valid_time > self.missing_timeout_sec
            )
            if timed_out and not self.hold_last_on_timeout:
                side_state.state = self.timeout_state
                side_state.closure_ema = self.timeout_state
            return side_state.state

        side_state.last_valid_time = now_sec
        closure_smoothed = side_state.update_closure(estimate["closure"])

        if closure_smoothed >= self.fist_threshold:
            side_state.state = 1.0
        elif closure_smoothed <= self.open_threshold:
            side_state.state = 0.0

        if self.publish_continuous_open_amount:
            return closure_smoothed
        return side_state.state

    def _finger5_from_closure(self, closure: float) -> List[float]:
        open_amount = float(1.0 - np.clip(closure, 0.0, 1.0))
        fingers = [open_amount, open_amount, open_amount, open_amount, open_amount]
        if self.ring_follows_middle:
            fingers[3] = fingers[2]
        return fingers

    def _output_data(self, left_closure: float, right_closure: float) -> List[float]:
        left5 = self._finger5_from_closure(left_closure)
        right5 = self._finger5_from_closure(right_closure)

        if self.output_layout == "hand2":
            return [
                float(1.0 - np.clip(left_closure, 0.0, 1.0)),
                float(1.0 - np.clip(right_closure, 0.0, 1.0)),
            ]

        if self.output_layout == "finger12":
            left6 = [left5[0], left5[0], left5[1], left5[2], left5[3], left5[4]]
            right6 = [right5[0], right5[0], right5[1], right5[2], right5[3], right5[4]]
            return [float(x) for x in left6 + right6]

        return [float(x) for x in left5 + right5]

    def on_points(self, msg: Float32MultiArray):
        conf = self.latest_conf if self.latest_conf is not None else -1
        if 0 <= conf < self.min_confidence:
            return

        data = np.asarray(msg.data, dtype=np.float64)
        if data.size == 0 or data.size % 3 != 0:
            self.get_logger().warn(
                f"Expected flat xyz array length multiple of 3, got {data.size}",
                throttle_duration_sec=2.0,
            )
            return

        pts_xyz = data.reshape(-1, 3)
        required_max_idx = max(
            self.left_cfg.finger_indices
            + self.right_cfg.finger_indices
            + [
                self.left_cfg.wrist_idx,
                self.left_cfg.elbow_idx,
                self.right_cfg.wrist_idx,
                self.right_cfg.elbow_idx,
            ]
        )
        if pts_xyz.shape[0] <= required_max_idx:
            self.get_logger().warn(
                f"Received {pts_xyz.shape[0]} points, need index up to {required_max_idx}.",
                throttle_duration_sec=2.0,
            )
            return

        now = self.get_clock().now()
        now_sec = float(now.nanoseconds) * 1e-9

        left_est = self._estimate_closure(pts_xyz, self.left_cfg)
        right_est = self._estimate_closure(pts_xyz, self.right_cfg)

        left_out = self._update_side(self.left_state, left_est, now_sec)
        right_out = self._update_side(self.right_state, right_est, now_sec)

        out = Float32MultiArray()
        out.data = self._output_data(left_out, right_out)
        self.pub.publish(out)

        if self.debug_log:
            if (now - self.last_log_time).nanoseconds * 1e-9 >= self.debug_log_period_sec:
                self.get_logger().info(
                    self._format_debug(left_est, right_est, left_out, right_out, out.data)
                )
                self.last_log_time = now

    @staticmethod
    def _format_estimate(label: str, est: Optional[Dict[str, float]]) -> str:
        if est is None:
            return f"{label}=missing"
        return (
            f"{label}: closure={est['closure']:.2f}, score={est['open_score']:.2f}, "
            f"spread={est['spread']:.2f}, extension={est['extension']:.2f}, "
            f"valid={int(est['valid_points'])}"
        )

    def _format_debug(
        self,
        left_est: Optional[Dict[str, float]],
        right_est: Optional[Dict[str, float]],
        left_out: float,
        right_out: float,
        out_data: List[float],
    ) -> str:
        return (
            "[zed_hand_finger_angles] "
            f"left={left_out:.2f}, right={right_out:.2f}, "
            f"out={np.round(out_data, 3).tolist()} | "
            f"{self._format_estimate('left', left_est)} | "
            f"{self._format_estimate('right', right_est)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ZedHandFingerAnglesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
