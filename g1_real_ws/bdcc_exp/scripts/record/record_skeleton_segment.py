#!/usr/bin/env python3
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import Float32MultiArray, UInt8, String
from sensor_msgs.msg import JointState


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


def stamp_to_sec(node: Node, msg) -> float:
    """
    Prefer message header stamp if available and nonzero.
    Otherwise use node clock time.
    """
    if hasattr(msg, "header"):
        stamp = msg.header.stamp
        sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if sec > 0.0:
            return sec
    now = node.get_clock().now()
    return now.nanoseconds * 1e-9


def ros_now_sec(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float_array(data) -> np.ndarray:
    return np.asarray(list(data), dtype=np.float64)


def jointstate_to_ordered_array(msg: JointState, joint_order: List[str]) -> np.ndarray:
    name_to_pos = dict(zip(msg.name, msg.position))
    out = []
    for jn in joint_order:
        out.append(float(name_to_pos.get(jn, np.nan)))
    return np.asarray(out, dtype=np.float64)


class SkeletonSegmentRecorder(Node):
    def __init__(self, args):
        super().__init__("bdcc_record_skeleton_segment")

        self.args = args
        self.outdir = Path(args.outdir).expanduser().resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.started = False
        self.finished = False
        self.start_wall_perf: Optional[float] = None
        self.start_ros_sec: Optional[float] = None

        # Main source data
        self.raw_t: List[float] = []
        self.raw_ros_t: List[float] = []
        self.raw_points: List[np.ndarray] = []

        self.filtered_t: List[float] = []
        self.filtered_ros_t: List[float] = []
        self.filtered_points: List[np.ndarray] = []

        self.conf_t: List[float] = []
        self.conf_ros_t: List[float] = []
        self.confidence: List[int] = []

        # Optional diagnostics
        self.diag: Dict[str, Dict[str, List[Any]]] = {
            "q_nom": {"t": [], "ros_t": [], "data": []},
            "q_cbf": {"t": [], "ros_t": [], "data": []},
            "q_act": {"t": [], "ros_t": [], "name": [], "position": [], "velocity": []},
            "human_capsules_robot": {"t": [], "ros_t": [], "data": []},
            "cbf_diagnostics": {"t": [], "ros_t": [], "data": []},
            "cbf_min_control_pair": {"t": [], "ros_t": [], "data": []},
        }

        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            args.raw_points_topic,
            self._raw_points_cb,
            args.qos,
        )
        self.create_subscription(
            Float32MultiArray,
            args.filtered_points_topic,
            self._filtered_points_cb,
            args.qos,
        )
        self.create_subscription(
            UInt8,
            args.confidence_topic,
            self._confidence_cb,
            args.qos,
        )

        if args.record_diagnostics:
            self.create_subscription(
                JointState,
                args.q_nom_topic,
                self._q_nom_cb,
                args.qos,
            )
            self.create_subscription(
                JointState,
                args.q_cbf_topic,
                self._q_cbf_cb,
                args.qos,
            )
            self.create_subscription(
                JointState,
                args.q_act_topic,
                self._q_act_cb,
                args.qos,
            )
            self.create_subscription(
                Float32MultiArray,
                args.human_capsule_topic,
                self._human_capsules_cb,
                args.qos,
            )
            self.create_subscription(
                Float32MultiArray,
                args.cbf_diagnostics_topic,
                self._cbf_diagnostics_cb,
                args.qos,
            )
            self.create_subscription(
                String,
                args.cbf_min_pair_topic,
                self._cbf_min_pair_cb,
                args.qos,
            )

        self.timer = self.create_timer(0.1, self._timer_cb)

        self.get_logger().info("BDCC skeleton segment recorder initialized.")
        self.get_logger().info(f"Output directory: {self.outdir}")
        self.get_logger().info(f"Scenario: {args.scenario}")
        self.get_logger().info(f"Duration: {args.duration:.3f} s")
        self.get_logger().info(f"Raw skeleton topic: {args.raw_points_topic}")
        self.get_logger().info(f"Filtered skeleton topic: {args.filtered_points_topic}")
        self.get_logger().info(f"Confidence topic: {args.confidence_topic}")
        self.get_logger().info(f"Record diagnostics: {args.record_diagnostics}")

    def start(self):
        self.started = True
        self.start_wall_perf = time.perf_counter()
        self.start_ros_sec = ros_now_sec(self)
        self.get_logger().info("Recording started.")

    def _relative_t(self) -> float:
        if self.start_wall_perf is None:
            return 0.0
        return time.perf_counter() - self.start_wall_perf

    def _accept(self) -> bool:
        return self.started and not self.finished

    def _raw_points_cb(self, msg: Float32MultiArray):
        if not self._accept():
            return
        self.raw_t.append(self._relative_t())
        self.raw_ros_t.append(stamp_to_sec(self, msg))
        self.raw_points.append(safe_float_array(msg.data))

    def _filtered_points_cb(self, msg: Float32MultiArray):
        if not self._accept():
            return
        self.filtered_t.append(self._relative_t())
        self.filtered_ros_t.append(stamp_to_sec(self, msg))
        self.filtered_points.append(safe_float_array(msg.data))

    def _confidence_cb(self, msg: UInt8):
        if not self._accept():
            return
        self.conf_t.append(self._relative_t())
        self.conf_ros_t.append(stamp_to_sec(self, msg))
        self.confidence.append(int(msg.data))

    def _q_nom_cb(self, msg: JointState):
        if not self._accept():
            return
        self.diag["q_nom"]["t"].append(self._relative_t())
        self.diag["q_nom"]["ros_t"].append(stamp_to_sec(self, msg))
        self.diag["q_nom"]["data"].append(jointstate_to_ordered_array(msg, CONTROLLED_JOINTS))

    def _q_cbf_cb(self, msg: JointState):
        if not self._accept():
            return
        self.diag["q_cbf"]["t"].append(self._relative_t())
        self.diag["q_cbf"]["ros_t"].append(stamp_to_sec(self, msg))
        self.diag["q_cbf"]["data"].append(jointstate_to_ordered_array(msg, CONTROLLED_JOINTS))

    def _q_act_cb(self, msg: JointState):
        if not self._accept():
            return
        self.diag["q_act"]["t"].append(self._relative_t())
        self.diag["q_act"]["ros_t"].append(stamp_to_sec(self, msg))
        self.diag["q_act"]["name"].append(list(msg.name))
        self.diag["q_act"]["position"].append(np.asarray(list(msg.position), dtype=np.float64))
        self.diag["q_act"]["velocity"].append(np.asarray(list(msg.velocity), dtype=np.float64))

    def _human_capsules_cb(self, msg: Float32MultiArray):
        if not self._accept():
            return
        self.diag["human_capsules_robot"]["t"].append(self._relative_t())
        self.diag["human_capsules_robot"]["ros_t"].append(stamp_to_sec(self, msg))
        self.diag["human_capsules_robot"]["data"].append(safe_float_array(msg.data))

    def _cbf_diagnostics_cb(self, msg: Float32MultiArray):
        if not self._accept():
            return
        self.diag["cbf_diagnostics"]["t"].append(self._relative_t())
        self.diag["cbf_diagnostics"]["ros_t"].append(stamp_to_sec(self, msg))
        self.diag["cbf_diagnostics"]["data"].append(safe_float_array(msg.data))

    def _cbf_min_pair_cb(self, msg: String):
        if not self._accept():
            return
        self.diag["cbf_min_control_pair"]["t"].append(self._relative_t())
        self.diag["cbf_min_control_pair"]["ros_t"].append(ros_now_sec(self))
        self.diag["cbf_min_control_pair"]["data"].append(str(msg.data))

    def _timer_cb(self):
        if not self.started or self.finished:
            return

        elapsed = self._relative_t()
        if elapsed >= self.args.duration:
            self.finished = True
            self.get_logger().info("Requested duration reached. Saving data...")
            self.save_all()
            self.print_summary()
            raise SystemExit

    def _stack_variable_length(self, arrays: List[np.ndarray], name: str) -> np.ndarray:
        """
        For normal skeleton messages, all arrays should have same length.
        If not, save as object array to avoid data loss.
        """
        if len(arrays) == 0:
            return np.empty((0,), dtype=np.float64)

        lengths = [a.size for a in arrays]
        if len(set(lengths)) == 1:
            return np.stack(arrays, axis=0)

        self.get_logger().warn(
            f"{name} has variable message lengths: {sorted(set(lengths))}. "
            "Saving as object array."
        )
        return np.asarray(arrays, dtype=object)

    def save_all(self):
        self._save_skeleton_raw()
        self._save_skeleton_filtered()
        self._save_confidence()
        if self.args.record_diagnostics:
            self._save_diagnostics()
        self._save_segment_yaml()
        self._save_manifest_json()

    def _save_skeleton_raw(self):
        path = self.outdir / "skeleton_raw.npz"
        np.savez_compressed(
            path,
            t=np.asarray(self.raw_t, dtype=np.float64),
            ros_t=np.asarray(self.raw_ros_t, dtype=np.float64),
            points=self._stack_variable_length(self.raw_points, "raw_points"),
            topic=self.args.raw_points_topic,
        )

    def _save_skeleton_filtered(self):
        path = self.outdir / "skeleton_filtered.npz"
        np.savez_compressed(
            path,
            t=np.asarray(self.filtered_t, dtype=np.float64),
            ros_t=np.asarray(self.filtered_ros_t, dtype=np.float64),
            points=self._stack_variable_length(self.filtered_points, "filtered_points"),
            topic=self.args.filtered_points_topic,
        )

    def _save_confidence(self):
        path = self.outdir / "confidence.npz"
        np.savez_compressed(
            path,
            t=np.asarray(self.conf_t, dtype=np.float64),
            ros_t=np.asarray(self.conf_ros_t, dtype=np.float64),
            confidence=np.asarray(self.confidence, dtype=np.int32),
            topic=self.args.confidence_topic,
        )

    def _save_diagnostics(self):
        path = self.outdir / "diagnostics.npz"

        save_dict = {}

        for key, item in self.diag.items():
            save_dict[f"{key}_t"] = np.asarray(item["t"], dtype=np.float64)
            save_dict[f"{key}_ros_t"] = np.asarray(item["ros_t"], dtype=np.float64)

            if key in ["q_nom", "q_cbf", "human_capsules_robot", "cbf_diagnostics"]:
                data = item["data"]
                save_dict[f"{key}_data"] = self._stack_variable_length(data, key)

            elif key == "q_act":
                save_dict[f"{key}_name"] = np.asarray(item["name"], dtype=object)
                save_dict[f"{key}_position"] = np.asarray(item["position"], dtype=object)
                save_dict[f"{key}_velocity"] = np.asarray(item["velocity"], dtype=object)

            elif key == "cbf_min_control_pair":
                save_dict[f"{key}_data"] = np.asarray(item["data"], dtype=object)

        np.savez_compressed(path, **save_dict)

    def _save_segment_yaml(self):
        """
        Avoid mandatory PyYAML dependency.
        Write a YAML-like text file manually.
        """
        path = self.outdir / "segment.yaml"

        lines = []
        lines.append(f"scenario_id: {self.args.scenario}")
        lines.append("recording_platform: real")
        lines.append("recording_mode: cbf_enabled")
        lines.append(f"duration_sec: {self.args.duration:.6f}")
        lines.append(f"created_unix_time: {time.time():.6f}")
        lines.append(f"created_readable_time: \"{time.strftime('%Y-%m-%d %H:%M:%S')}\"")
        lines.append("")
        lines.append("source_topics:")
        lines.append(f"  skeleton_raw: {self.args.raw_points_topic}")
        lines.append(f"  skeleton_filtered: {self.args.filtered_points_topic}")
        lines.append(f"  confidence: {self.args.confidence_topic}")
        lines.append("")
        lines.append("diagnostic_topics:")
        lines.append(f"  enabled: {str(self.args.record_diagnostics).lower()}")
        lines.append(f"  q_nom: {self.args.q_nom_topic}")
        lines.append(f"  q_cbf: {self.args.q_cbf_topic}")
        lines.append(f"  q_act: {self.args.q_act_topic}")
        lines.append(f"  human_capsules_robot: {self.args.human_capsule_topic}")
        lines.append(f"  cbf_diagnostics: {self.args.cbf_diagnostics_topic}")
        lines.append(f"  cbf_min_control_pair: {self.args.cbf_min_pair_topic}")
        lines.append("")
        lines.append("calibration:")
        lines.append("  frame_source: zed_world")
        lines.append("  target_frame: pelvis")
        lines.append(f"  extrinsic_tx: {self.args.extrinsic_tx}")
        lines.append(f"  extrinsic_ty: {self.args.extrinsic_ty}")
        lines.append(f"  extrinsic_tz: {self.args.extrinsic_tz}")
        lines.append(f"  extrinsic_qx: {self.args.extrinsic_qx}")
        lines.append(f"  extrinsic_qy: {self.args.extrinsic_qy}")
        lines.append(f"  extrinsic_qz: {self.args.extrinsic_qz}")
        lines.append(f"  extrinsic_qw: {self.args.extrinsic_qw}")
        lines.append("")
        lines.append("human_capsule_radius_real:")
        lines.append(f"  torso_radius: {self.args.human_torso_radius_real}")
        lines.append(f"  upper_arm_radius: {self.args.human_upper_arm_radius_real}")
        lines.append(f"  forearm_radius: {self.args.human_forearm_radius_real}")
        lines.append(f"  thigh_radius: {self.args.human_thigh_radius_real}")
        lines.append(f"  shin_radius: {self.args.human_shin_radius_real}")
        lines.append(f"  head_radius: {self.args.human_head_radius_real}")
        lines.append("")
        lines.append("cbf_default:")
        lines.append(f"  rr_safety_distance: {self.args.rr_safety_distance}")
        lines.append(f"  hr_safety_distance: {self.args.hr_safety_distance}")
        lines.append(f"  rr_gamma: {self.args.rr_gamma}")
        lines.append(f"  hr_gamma: {self.args.hr_gamma}")
        lines.append(f"  lpf_gain: {self.args.lpf_gain}")
        lines.append("")
        lines.append("counts:")
        lines.append(f"  raw_skeleton_frames: {len(self.raw_t)}")
        lines.append(f"  filtered_skeleton_frames: {len(self.filtered_t)}")
        lines.append(f"  confidence_frames: {len(self.conf_t)}")
        if self.args.record_diagnostics:
            for key, item in self.diag.items():
                lines.append(f"  {key}_frames: {len(item['t'])}")
        lines.append("")
        lines.append(f"notes: \"{self.args.notes}\"")
        lines.append("")

        path.write_text("\n".join(lines))

    def _save_manifest_json(self):
        path = self.outdir / "manifest.json"
        manifest = {
            "scenario_id": self.args.scenario,
            "outdir": str(self.outdir),
            "duration_sec": self.args.duration,
            "created_unix_time": time.time(),
            "topics": {
                "raw_points": self.args.raw_points_topic,
                "filtered_points": self.args.filtered_points_topic,
                "confidence": self.args.confidence_topic,
            },
            "record_diagnostics": self.args.record_diagnostics,
            "counts": {
                "raw_skeleton_frames": len(self.raw_t),
                "filtered_skeleton_frames": len(self.filtered_t),
                "confidence_frames": len(self.conf_t),
                "q_nom_frames": len(self.diag["q_nom"]["t"]),
                "q_cbf_frames": len(self.diag["q_cbf"]["t"]),
                "q_act_frames": len(self.diag["q_act"]["t"]),
                "human_capsules_robot_frames": len(self.diag["human_capsules_robot"]["t"]),
                "cbf_diagnostics_frames": len(self.diag["cbf_diagnostics"]["t"]),
                "cbf_min_control_pair_frames": len(self.diag["cbf_min_control_pair"]["t"]),
            },
        }
        path.write_text(json.dumps(manifest, indent=2))

    def print_summary(self):
        self.get_logger().info("========== Recording summary ==========")
        self.get_logger().info(f"Output: {self.outdir}")
        self.get_logger().info(f"Raw skeleton frames:      {len(self.raw_t)}")
        self.get_logger().info(f"Filtered skeleton frames: {len(self.filtered_t)}")
        self.get_logger().info(f"Confidence frames:        {len(self.conf_t)}")

        if self.args.record_diagnostics:
            for key, item in self.diag.items():
                self.get_logger().info(f"{key} frames: {len(item['t'])}")

        self.get_logger().info("Files written:")
        self.get_logger().info("  segment.yaml")
        self.get_logger().info("  manifest.json")
        self.get_logger().info("  skeleton_raw.npz")
        self.get_logger().info("  skeleton_filtered.npz")
        self.get_logger().info("  confidence.npz")
        if self.args.record_diagnostics:
            self.get_logger().info("  diagnostics.npz")
        self.get_logger().info("=======================================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a BDCC skeleton-only replay segment."
    )

    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario ID, e.g., S1_self_collision or S2_human_robot.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for this segment.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=40.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--start-on-enter",
        action="store_true",
        help="Wait for Enter before recording.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes saved into segment.yaml.",
    )

    # Source topics
    parser.add_argument("--raw-points-topic", default="/skeleton/points")
    parser.add_argument("--filtered-points-topic", default="/skeleton/points_filtered")
    parser.add_argument("--confidence-topic", default="/skeleton/confidence")
    parser.add_argument("--qos", type=int, default=50)

    # Diagnostics
    parser.add_argument("--record-diagnostics", action="store_true")
    parser.add_argument("--q-nom-topic", default="/joint_commands_unsafe")
    parser.add_argument("--q-cbf-topic", default="/real/joint_commands")
    parser.add_argument("--q-act-topic", default="/real/joint_states")
    parser.add_argument("--human-capsule-topic", default="/real/human_capsules_robot")
    parser.add_argument("--cbf-diagnostics-topic", default="/real/cbf/diagnostics")
    parser.add_argument("--cbf-min-pair-topic", default="/real/cbf/min_control_pair")

    # Calibration defaults from current launch file
    parser.add_argument("--extrinsic-tx", type=float, default=2.155132030254977)
    parser.add_argument("--extrinsic-ty", type=float, default=0.2126549700677248)
    parser.add_argument("--extrinsic-tz", type=float, default=-0.16733448394805378)
    parser.add_argument("--extrinsic-qx", type=float, default=0.11707934250730477)
    parser.add_argument("--extrinsic-qy", type=float, default=0.03377182306070312)
    parser.add_argument("--extrinsic-qz", type=float, default=0.992493225367401)
    parser.add_argument("--extrinsic-qw", type=float, default=0.010444573951437231)

    # Human real capsule radius defaults
    parser.add_argument("--human-torso-radius-real", type=float, default=0.15)
    parser.add_argument("--human-upper-arm-radius-real", type=float, default=0.09)
    parser.add_argument("--human-forearm-radius-real", type=float, default=0.10)
    parser.add_argument("--human-thigh-radius-real", type=float, default=0.08)
    parser.add_argument("--human-shin-radius-real", type=float, default=0.07)
    parser.add_argument("--human-head-radius-real", type=float, default=0.08)

    # Default CBF parameters at recording time
    parser.add_argument("--rr-safety-distance", type=float, default=0.015)
    parser.add_argument("--hr-safety-distance", type=float, default=0.10)
    parser.add_argument("--rr-gamma", type=float, default=2.0)
    parser.add_argument("--hr-gamma", type=float, default=4.0)
    parser.add_argument("--lpf-gain", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.start_on_enter:
        print("")
        print("======================================================")
        print("BDCC skeleton segment recorder")
        print(f"Scenario: {args.scenario}")
        print(f"Output:   {outdir}")
        print(f"Duration: {args.duration:.3f} s")
        print("Press Enter to start recording.")
        print("======================================================")
        input()

    rclpy.init(args=None)
    node = SkeletonSegmentRecorder(args)

    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt received. Saving partial data...")
        node.finished = True
        node.save_all()
        node.print_summary()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()