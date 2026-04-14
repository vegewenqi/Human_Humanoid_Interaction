#!/usr/bin/env python3
import csv
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


CONTROLLED_JOINTS = [
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
]


@dataclass
class Sample:
    t_sec: float
    unsafe: np.ndarray
    nominal: np.ndarray
    capsules: np.ndarray


class SynchronizedMotionCapsuleReplay(Node):
    def __init__(self):
        super().__init__("synchronized_motion_capsule_replay")

        self.declare_parameter("input_csv", "/ws/ecc_sim/data/recorded_motion_capsules.csv")
        self.declare_parameter("unsafe_joint_topic", "/joint_commands_unsafe")
        self.declare_parameter("nominal_array_topic", "/g1_upperbody_q_des")
        self.declare_parameter("human_capsule_topic", "/sim/human_capsules_robot")
        self.declare_parameter("replay_rate_hz", 120.0)
        self.declare_parameter("time_scale", 1.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("interpolate_commands", True)
        self.declare_parameter("hold_capsules", True)
        self.declare_parameter("capsule_length", 70)

        self.input_csv = str(self.get_parameter("input_csv").value)
        self.unsafe_joint_topic = str(self.get_parameter("unsafe_joint_topic").value)
        self.nominal_array_topic = str(self.get_parameter("nominal_array_topic").value)
        self.human_capsule_topic = str(self.get_parameter("human_capsule_topic").value)
        self.replay_rate_hz = float(self.get_parameter("replay_rate_hz").value)
        self.time_scale = float(self.get_parameter("time_scale").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.interpolate_commands = bool(self.get_parameter("interpolate_commands").value)
        self.hold_capsules = bool(self.get_parameter("hold_capsules").value)
        self.capsule_length = int(self.get_parameter("capsule_length").value)

        self.samples = self._load_samples(self.input_csv)
        if len(self.samples) < 2:
            raise RuntimeError("Need at least 2 samples in input_csv")

        self.pub_unsafe = self.create_publisher(JointState, self.unsafe_joint_topic, 20)
        self.pub_nominal = self.create_publisher(Float32MultiArray, self.nominal_array_topic, 20)
        self.pub_capsules = self.create_publisher(Float32MultiArray, self.human_capsule_topic, 20)

        self._t0_wall = self.get_clock().now()
        self._published_done = False
        self._last_capsule_idx = 0
        self._duration = self.samples[-1].t_sec

        period = 1.0 / max(self.replay_rate_hz, 1e-6)
        self.create_timer(period, self._tick)

        self.get_logger().info(f"Loaded {len(self.samples)} synchronized samples from {self.input_csv}")
        self.get_logger().info(f"Replay topics: unsafe={self.unsafe_joint_topic}, nominal_array={self.nominal_array_topic}, human_capsules={self.human_capsule_topic}")
        self.get_logger().info(f"replay_rate_hz={self.replay_rate_hz:.1f}, interpolate_commands={self.interpolate_commands}, hold_capsules={self.hold_capsules}, duration={self._duration:.3f}s")

    def _load_samples(self, path: str) -> List[Sample]:
        samples: List[Sample] = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_sec = float(row["t_sec"])
                unsafe = np.array([float(row[f"unsafe_{j}"]) for j in CONTROLLED_JOINTS], dtype=np.float64)
                nominal = np.array([float(row[f"nominal_array_{i}"]) for i in range(8)], dtype=np.float64)
                capsules = np.array([float(row[f"human_capsule_{i}"]) for i in range(self.capsule_length)], dtype=np.float64)
                samples.append(Sample(t_sec=t_sec, unsafe=unsafe, nominal=nominal, capsules=capsules))
        samples.sort(key=lambda s: s.t_sec)
        return samples

    def _elapsed(self) -> float:
        now = self.get_clock().now()
        return (now - self._t0_wall).nanoseconds * 1e-9 * max(self.time_scale, 1e-9)

    def _reset(self):
        self._t0_wall = self.get_clock().now()
        self._published_done = False
        self._last_capsule_idx = 0

    def _find_bracketing_index(self, t: float) -> int:
        lo = 0
        hi = len(self.samples) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.samples[mid].t_sec <= t < self.samples[mid + 1].t_sec:
                return mid
            if t < self.samples[mid].t_sec:
                hi = mid - 1
            else:
                lo = mid + 1
        return max(0, min(len(self.samples) - 2, lo))

    def _interp(self, a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * a + alpha * b

    def _publish(self, unsafe: np.ndarray, nominal: np.ndarray, capsules: np.ndarray):
        stamp = self.get_clock().now().to_msg()

        msg_js = JointState()
        msg_js.header.stamp = stamp
        msg_js.name = list(CONTROLLED_JOINTS)
        msg_js.position = [float(x) for x in unsafe]
        msg_js.velocity = []
        msg_js.effort = []
        self.pub_unsafe.publish(msg_js)

        msg_arr = Float32MultiArray()
        msg_arr.data = [float(x) for x in nominal]
        self.pub_nominal.publish(msg_arr)

        msg_caps = Float32MultiArray()
        msg_caps.data = [float(x) for x in capsules]
        self.pub_capsules.publish(msg_caps)

    def _tick(self):
        if self._published_done:
            return

        t = self._elapsed()
        if t >= self._duration:
            last = self.samples[-1]
            self._publish(last.unsafe, last.nominal, last.capsules)
            if self.loop:
                self._reset()
            else:
                self._published_done = True
                self.get_logger().info("Replay complete")
            return

        idx = self._find_bracketing_index(t)
        s0 = self.samples[idx]
        s1 = self.samples[idx + 1]
        dt = max(s1.t_sec - s0.t_sec, 1e-9)
        alpha = np.clip((t - s0.t_sec) / dt, 0.0, 1.0)

        if self.interpolate_commands:
            unsafe = self._interp(s0.unsafe, s1.unsafe, alpha)
            nominal = self._interp(s0.nominal, s1.nominal, alpha)
        else:
            unsafe = s0.unsafe.copy()
            nominal = s0.nominal.copy()

        if self.hold_capsules:
            while self._last_capsule_idx + 1 < len(self.samples) and self.samples[self._last_capsule_idx + 1].t_sec <= t:
                self._last_capsule_idx += 1
            capsules = self.samples[self._last_capsule_idx].capsules.copy()
        else:
            capsules = self._interp(s0.capsules, s1.capsules, alpha)

        self._publish(unsafe, nominal, capsules)


def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedMotionCapsuleReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
