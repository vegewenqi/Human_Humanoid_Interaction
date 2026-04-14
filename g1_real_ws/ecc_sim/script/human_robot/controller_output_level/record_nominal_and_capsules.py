#!/usr/bin/env python3
import csv
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
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


class UnifiedMotionCapsuleRecorder(Node):
    def __init__(self):
        super().__init__("unified_motion_capsule_recorder")

        self.declare_parameter("output_csv", "/ws/ecc_sim/data/recorded_motion_capsules.csv")
        self.declare_parameter("unsafe_joint_topic", "/joint_commands_unsafe")
        self.declare_parameter("nominal_array_topic", "/g1_upperbody_q_des")
        self.declare_parameter("human_capsule_topic", "/sim/human_capsules_robot")
        self.declare_parameter("write_rate_hz", 60.0)
        self.declare_parameter("record_only_when_fresh", True)
        self.declare_parameter("fresh_timeout_sec", 0.10)
        self.declare_parameter("require_capsules", True)
        self.declare_parameter("require_command", True)
        self.declare_parameter("capsule_length", 70)

        self.output_csv = str(self.get_parameter("output_csv").value)
        self.unsafe_joint_topic = str(self.get_parameter("unsafe_joint_topic").value)
        self.nominal_array_topic = str(self.get_parameter("nominal_array_topic").value)
        self.human_capsule_topic = str(self.get_parameter("human_capsule_topic").value)
        self.write_rate_hz = float(self.get_parameter("write_rate_hz").value)
        self.record_only_when_fresh = bool(self.get_parameter("record_only_when_fresh").value)
        self.fresh_timeout_sec = float(self.get_parameter("fresh_timeout_sec").value)
        self.require_capsules = bool(self.get_parameter("require_capsules").value)
        self.require_command = bool(self.get_parameter("require_command").value)
        self.capsule_length = int(self.get_parameter("capsule_length").value)

        self.latest_unsafe_js: Optional[JointState] = None
        self.latest_nominal_array: Optional[List[float]] = None
        self.latest_capsules: Optional[List[float]] = None

        self.latest_unsafe_t: Optional[Time] = None
        self.latest_nominal_array_t: Optional[Time] = None
        self.latest_capsules_t: Optional[Time] = None

        self._start_time: Optional[Time] = None
        self._rows_written = 0

        self.create_subscription(JointState, self.unsafe_joint_topic, self._unsafe_cb, 50)
        self.create_subscription(Float32MultiArray, self.nominal_array_topic, self._array_cb, 50)
        self.create_subscription(Float32MultiArray, self.human_capsule_topic, self._capsule_cb, 50)

        self._csv_file = open(self.output_csv, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._write_header()

        period = 1.0 / max(self.write_rate_hz, 1e-6)
        self.create_timer(period, self._tick)

        self.get_logger().info(f"Writing synchronized samples to: {self.output_csv}")
        self.get_logger().info(f"Topics: unsafe={self.unsafe_joint_topic}, nominal_array={self.nominal_array_topic}, human_capsules={self.human_capsule_topic}")
        self.get_logger().info(f"write_rate_hz={self.write_rate_hz:.2f}, fresh_timeout_sec={self.fresh_timeout_sec:.3f}, capsule_length={self.capsule_length}")

    def _write_header(self):
        header = ["t_sec"]
        header += [f"unsafe_{j}" for j in CONTROLLED_JOINTS]
        header += [f"nominal_array_{i}" for i in range(8)]
        header += [f"human_capsule_{i}" for i in range(self.capsule_length)]
        self._writer.writerow(header)
        self._csv_file.flush()

    def _unsafe_cb(self, msg: JointState):
        self.latest_unsafe_js = msg
        self.latest_unsafe_t = self.get_clock().now()
        if self._start_time is None:
            self._start_time = self.latest_unsafe_t

    def _array_cb(self, msg: Float32MultiArray):
        self.latest_nominal_array = list(msg.data)
        self.latest_nominal_array_t = self.get_clock().now()
        if self._start_time is None:
            self._start_time = self.latest_nominal_array_t

    def _capsule_cb(self, msg: Float32MultiArray):
        self.latest_capsules = list(msg.data)
        self.latest_capsules_t = self.get_clock().now()
        if self._start_time is None:
            self._start_time = self.latest_capsules_t

    def _extract_unsafe_positions(self) -> Optional[List[float]]:
        if self.latest_unsafe_js is None:
            return None
        name_to_pos: Dict[str, float] = dict(zip(self.latest_unsafe_js.name, self.latest_unsafe_js.position))
        out = []
        for jn in CONTROLLED_JOINTS:
            if jn not in name_to_pos:
                self.get_logger().warn(f"Missing joint in unsafe JointState: {jn}", throttle_duration_sec=2.0)
                return None
            out.append(float(name_to_pos[jn]))
        return out

    def _is_fresh(self, ts: Optional[Time], now: Time) -> bool:
        if ts is None:
            return False
        age = (now - ts).nanoseconds * 1e-9
        return age <= self.fresh_timeout_sec

    def _tick(self):
        now = self.get_clock().now()
        if self._start_time is None:
            return

        unsafe_vals = self._extract_unsafe_positions()
        array_vals = self.latest_nominal_array
        capsule_vals = self.latest_capsules

        if self.require_command and unsafe_vals is None and array_vals is None:
            return
        if self.require_capsules and capsule_vals is None:
            return

        if self.record_only_when_fresh:
            cmd_fresh = self._is_fresh(self.latest_unsafe_t, now) or self._is_fresh(self.latest_nominal_array_t, now)
            cap_fresh = self._is_fresh(self.latest_capsules_t, now)
            if self.require_command and not cmd_fresh:
                return
            if self.require_capsules and not cap_fresh:
                return

        if unsafe_vals is None:
            unsafe_vals = [float("nan")] * 8
        if array_vals is None:
            array_vals = [float("nan")] * 8
        else:
            array_vals = list(array_vals[:8]) + [float("nan")] * max(0, 8 - len(array_vals))
            array_vals = array_vals[:8]

        if capsule_vals is None:
            capsule_vals = [float("nan")] * self.capsule_length
        else:
            if len(capsule_vals) < self.capsule_length:
                capsule_vals = list(capsule_vals) + [float("nan")] * (self.capsule_length - len(capsule_vals))
            else:
                capsule_vals = list(capsule_vals[:self.capsule_length])

        t_sec = (now - self._start_time).nanoseconds * 1e-9
        row = [t_sec] + unsafe_vals + array_vals + capsule_vals
        self._writer.writerow(row)
        self._rows_written += 1

        if self._rows_written % 50 == 0:
            self._csv_file.flush()
            self.get_logger().info(f"Recorded {self._rows_written} synchronized rows")

    def destroy_node(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedMotionCapsuleRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
