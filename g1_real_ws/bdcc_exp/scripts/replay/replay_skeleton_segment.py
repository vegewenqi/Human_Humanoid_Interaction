#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8


def load_npz_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(str(path), allow_pickle=True)


def sanitize_points_array(points: np.ndarray) -> np.ndarray:
    """
    Expected normal case:
      points shape = (N, D), e.g. (794, 114)

    If saved as object array due to variable lengths, convert each frame later.
    """
    return points


class SkeletonSegmentReplay(Node):
    def __init__(self, args):
        super().__init__("bdcc_replay_skeleton_segment")
        self.args = args

        self.segment_dir = Path(args.segment).expanduser().resolve()
        if not self.segment_dir.exists():
            raise FileNotFoundError(f"Segment directory does not exist: {self.segment_dir}")

        if args.publish_mode == "filtered":
            skel_path = self.segment_dir / "skeleton_filtered.npz"
            self.points_topic = args.filtered_points_topic
        elif args.publish_mode == "raw":
            skel_path = self.segment_dir / "skeleton_raw.npz"
            self.points_topic = args.raw_points_topic
        else:
            raise ValueError(f"Unknown publish_mode: {args.publish_mode}")

        conf_path = self.segment_dir / "confidence.npz"

        skel = load_npz_required(skel_path)
        self.t = np.asarray(skel["t"], dtype=np.float64)
        self.points = sanitize_points_array(skel["points"])

        if self.t.ndim != 1 or len(self.t) == 0:
            raise RuntimeError(f"Invalid skeleton time array in {skel_path}")

        if len(self.points) != len(self.t):
            raise RuntimeError(
                f"points length {len(self.points)} does not match t length {len(self.t)}"
            )

        self.conf_t = None
        self.confidence = None
        if conf_path.exists():
            conf = np.load(str(conf_path), allow_pickle=True)
            self.conf_t = np.asarray(conf["t"], dtype=np.float64)
            self.confidence = np.asarray(conf["confidence"], dtype=np.int32)
            if len(self.conf_t) == 0 or len(self.confidence) == 0:
                self.conf_t = None
                self.confidence = None

        self.t0_data = float(self.t[0])
        self.t_data = self.t - self.t0_data
        self.duration = float(self.t_data[-1])

        if self.duration <= 0:
            raise RuntimeError("Segment duration is zero or negative.")

        self.pub_points = self.create_publisher(
            Float32MultiArray,
            self.points_topic,
            args.qos,
        )
        self.pub_conf = self.create_publisher(
            UInt8,
            args.confidence_topic,
            args.qos,
        )

        self._wall_start: Optional[float] = None
        self._started = False
        self._done = False
        self._last_idx = -1
        self._last_conf_idx = -1
        self._loops_done = 0

        timer_period = 1.0 / max(args.replay_rate_hz, 1e-6)
        self.timer = self.create_timer(timer_period, self._tick)

        self.get_logger().info("BDCC skeleton segment replay initialized.")
        self.get_logger().info(f"Segment directory: {self.segment_dir}")
        self.get_logger().info(f"Publish mode: {args.publish_mode}")
        self.get_logger().info(f"Skeleton frames: {len(self.t)}")
        self.get_logger().info(f"Duration: {self.duration:.3f} s")
        self.get_logger().info(f"Points topic: {self.points_topic}")
        self.get_logger().info(f"Confidence topic: {args.confidence_topic}")
        self.get_logger().info(f"Replay rate: {args.replay_rate_hz:.1f} Hz")
        self.get_logger().info(f"Time scale: {args.time_scale:.3f}")
        self.get_logger().info(f"Loop: {args.loop}")
        self.get_logger().info(f"Interpolate points: {args.interpolate_points}")

    def start(self):
        self._wall_start = time.perf_counter() + max(0.0, self.args.start_delay)
        self._started = True
        self.get_logger().info(
            f"Replay will start after {self.args.start_delay:.3f} s delay."
        )

    def _elapsed_data_time(self) -> Optional[float]:
        if not self._started or self._wall_start is None:
            return None

        now = time.perf_counter()
        if now < self._wall_start:
            return None

        elapsed_wall = now - self._wall_start
        t = elapsed_wall * max(self.args.time_scale, 1e-9)
        return t

    def _find_index(self, t_query: float) -> int:
        idx = int(np.searchsorted(self.t_data, t_query, side="right") - 1)
        return max(0, min(idx, len(self.t_data) - 1))

    def _interp_frame(self, idx: int, t_query: float) -> np.ndarray:
        if not self.args.interpolate_points:
            return np.asarray(self.points[idx], dtype=np.float64).reshape(-1)

        if idx >= len(self.t_data) - 1:
            return np.asarray(self.points[-1], dtype=np.float64).reshape(-1)

        t0 = self.t_data[idx]
        t1 = self.t_data[idx + 1]
        dt = max(float(t1 - t0), 1e-12)
        alpha = float(np.clip((t_query - t0) / dt, 0.0, 1.0))

        p0 = np.asarray(self.points[idx], dtype=np.float64).reshape(-1)
        p1 = np.asarray(self.points[idx + 1], dtype=np.float64).reshape(-1)

        if p0.shape != p1.shape:
            return p0

        return (1.0 - alpha) * p0 + alpha * p1

    def _confidence_at(self, t_query: float) -> int:
        if self.conf_t is None or self.confidence is None:
            return int(self.args.default_confidence)

        conf_t_rel = self.conf_t - float(self.conf_t[0])
        idx = int(np.searchsorted(conf_t_rel, t_query, side="right") - 1)
        idx = max(0, min(idx, len(self.confidence) - 1))
        self._last_conf_idx = idx

        val = int(self.confidence[idx])
        val = max(0, min(255, val))
        return val

    def _publish_points(self, frame: np.ndarray):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in frame]
        self.pub_points.publish(msg)

    def _publish_confidence(self, conf_val: int):
        msg = UInt8()
        msg.data = int(max(0, min(255, conf_val)))
        self.pub_conf.publish(msg)

    def _publish_frame(self, idx: int, t_query: float):
        frame = self._interp_frame(idx, t_query)
        conf = self._confidence_at(t_query)

        self._publish_points(frame)
        self._publish_confidence(conf)

        self._last_idx = idx

    def _tick(self):
        if self._done:
            return

        t_query = self._elapsed_data_time()
        if t_query is None:
            return

        if t_query >= self.duration:
            self._publish_frame(len(self.t_data) - 1, self.duration)

            if self.args.loop:
                self._loops_done += 1
                self.get_logger().info(f"Replay loop {self._loops_done} complete. Restarting.")
                self._wall_start = time.perf_counter()
                self._last_idx = -1
                self._last_conf_idx = -1
                return

            self._done = True
            self.get_logger().info("Replay complete.")
            return

        idx = self._find_index(t_query)

        # If interpolation is enabled, publish at fixed replay timer rate.
        # If interpolation is disabled, avoid re-publishing same source frame repeatedly.
        if (not self.args.interpolate_points) and idx == self._last_idx:
            return

        self._publish_frame(idx, t_query)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a BDCC skeleton-only segment."
    )

    parser.add_argument(
        "--segment",
        required=True,
        help="Segment directory, e.g. /ws/bdcc_exp/segments/S1_self_collision",
    )
    parser.add_argument(
        "--publish-mode",
        choices=["filtered", "raw"],
        default="filtered",
        help="Replay skeleton_filtered.npz or skeleton_raw.npz.",
    )

    parser.add_argument("--raw-points-topic", default="/skeleton/points")
    parser.add_argument("--filtered-points-topic", default="/skeleton/points_filtered")
    parser.add_argument("--confidence-topic", default="/skeleton/confidence")

    parser.add_argument(
        "--replay-rate-hz",
        type=float,
        default=60.0,
        help="ROS publish timer rate.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="1.0 = real time; 0.5 = slower; 2.0 = faster.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Delay before replay starts, useful for starting logger first.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop replay forever.",
    )
    parser.add_argument(
        "--interpolate-points",
        action="store_true",
        help="Linearly interpolate skeleton points between source frames.",
    )
    parser.add_argument(
        "--default-confidence",
        type=int,
        default=100,
        help="Used only if confidence.npz is missing or empty.",
    )
    parser.add_argument("--qos", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init(args=None)
    node = SkeletonSegmentReplay(args)

    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt received. Exiting replay.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()