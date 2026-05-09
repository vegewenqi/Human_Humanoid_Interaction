#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node

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


def ros_now_sec(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def stamp_to_sec(node: Node, msg: Any) -> float:
    if hasattr(msg, "header"):
        stamp = msg.header.stamp
        sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if sec > 0.0:
            return sec
    return ros_now_sec(node)


def jointstate_to_ordered_array(
    msg: JointState,
    joint_order: List[str],
    fill_missing: float = np.nan,
) -> np.ndarray:
    name_to_pos = dict(zip(msg.name, msg.position))
    out = []
    for jn in joint_order:
        out.append(float(name_to_pos.get(jn, fill_missing)))
    return np.asarray(out, dtype=np.float64)


def msg_float_array(msg: Float32MultiArray) -> np.ndarray:
    return np.asarray(list(msg.data), dtype=np.float64)


def stack_or_object(arrays: List[np.ndarray]) -> np.ndarray:
    if len(arrays) == 0:
        return np.empty((0,), dtype=np.float64)
    shapes = [tuple(np.asarray(a).shape) for a in arrays]
    if len(set(shapes)) == 1:
        return np.stack(arrays, axis=0)
    return np.asarray(arrays, dtype=object)


class TrialTopicLogger(Node):
    def __init__(self, args):
        super().__init__("bdcc_trial_topic_logger")

        self.args = args
        self.outdir = Path(args.outdir).expanduser().resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.started = False
        self.finished = False
        self.start_wall_perf: Optional[float] = None
        self.start_ros_sec: Optional[float] = None

        self.data: Dict[str, Dict[str, List[Any]]] = {
            "q_nom": {"t": [], "ros_t": [], "data": []},
            "q_cbf": {"t": [], "ros_t": [], "data": []},
            "q_act": {"t": [], "ros_t": [], "name": [], "position": [], "velocity": []},
            "skeleton_filtered": {"t": [], "ros_t": [], "data": []},
            "confidence": {"t": [], "ros_t": [], "data": []},
            "human_capsules_robot": {"t": [], "ros_t": [], "data": []},
            "cbf_diagnostics": {"t": [], "ros_t": [], "data": []},
            "cbf_min_control_pair": {"t": [], "ros_t": [], "data": []},
        }

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

        if args.record_q_act:
            self.create_subscription(
                JointState,
                args.q_act_topic,
                self._q_act_cb,
                args.qos,
            )

        self.create_subscription(
            Float32MultiArray,
            args.skeleton_filtered_topic,
            self._skeleton_filtered_cb,
            args.qos,
        )
        self.create_subscription(
            UInt8,
            args.confidence_topic,
            self._confidence_cb,
            args.qos,
        )
        self.create_subscription(
            Float32MultiArray,
            args.human_capsule_topic,
            self._human_capsule_cb,
            args.qos,
        )

        if args.record_cbf_diagnostics:
            self.create_subscription(
                Float32MultiArray,
                args.cbf_diagnostics_topic,
                self._cbf_diag_cb,
                args.qos,
            )
            self.create_subscription(
                String,
                args.cbf_min_pair_topic,
                self._cbf_min_pair_cb,
                args.qos,
            )

        self.timer = self.create_timer(0.1, self._timer_cb)

        self.get_logger().info("BDCC trial topic logger initialized.")
        self.get_logger().info(f"Platform: {args.platform}")
        self.get_logger().info(f"Output directory: {self.outdir}")
        self.get_logger().info(f"Duration: {args.duration:.3f} s")
        self.get_logger().info(f"q_nom topic: {args.q_nom_topic}")
        self.get_logger().info(f"q_cbf topic: {args.q_cbf_topic}")
        self.get_logger().info(f"q_act topic: {args.q_act_topic}, enabled={args.record_q_act}")
        self.get_logger().info(f"skeleton topic: {args.skeleton_filtered_topic}")
        self.get_logger().info(f"confidence topic: {args.confidence_topic}")
        self.get_logger().info(f"human capsule topic: {args.human_capsule_topic}")
        self.get_logger().info(f"CBF diagnostics enabled: {args.record_cbf_diagnostics}")
        self.get_logger().info(f"CBF diagnostics topic: {args.cbf_diagnostics_topic}")
        self.get_logger().info(f"CBF min pair topic: {args.cbf_min_pair_topic}")

    def start(self):
        self.started = True
        self.start_wall_perf = time.perf_counter()
        self.start_ros_sec = ros_now_sec(self)
        self.get_logger().info("Trial logging started.")

    def _relative_t(self) -> float:
        if self.start_wall_perf is None:
            return 0.0
        return time.perf_counter() - self.start_wall_perf

    def _accept(self) -> bool:
        return self.started and not self.finished

    def _append_array_msg(self, key: str, msg: Float32MultiArray):
        if not self._accept():
            return
        self.data[key]["t"].append(self._relative_t())
        self.data[key]["ros_t"].append(stamp_to_sec(self, msg))
        self.data[key]["data"].append(msg_float_array(msg))

    def _q_nom_cb(self, msg: JointState):
        if not self._accept():
            return
        self.data["q_nom"]["t"].append(self._relative_t())
        self.data["q_nom"]["ros_t"].append(stamp_to_sec(self, msg))
        self.data["q_nom"]["data"].append(
            jointstate_to_ordered_array(msg, CONTROLLED_JOINTS)
        )

    def _q_cbf_cb(self, msg: JointState):
        if not self._accept():
            return
        self.data["q_cbf"]["t"].append(self._relative_t())
        self.data["q_cbf"]["ros_t"].append(stamp_to_sec(self, msg))
        self.data["q_cbf"]["data"].append(
            jointstate_to_ordered_array(msg, CONTROLLED_JOINTS)
        )

    def _q_act_cb(self, msg: JointState):
        if not self._accept():
            return
        self.data["q_act"]["t"].append(self._relative_t())
        self.data["q_act"]["ros_t"].append(stamp_to_sec(self, msg))
        self.data["q_act"]["name"].append(list(msg.name))
        self.data["q_act"]["position"].append(
            np.asarray(list(msg.position), dtype=np.float64)
        )
        self.data["q_act"]["velocity"].append(
            np.asarray(list(msg.velocity), dtype=np.float64)
        )

    def _skeleton_filtered_cb(self, msg: Float32MultiArray):
        self._append_array_msg("skeleton_filtered", msg)

    def _confidence_cb(self, msg: UInt8):
        if not self._accept():
            return
        self.data["confidence"]["t"].append(self._relative_t())
        self.data["confidence"]["ros_t"].append(stamp_to_sec(self, msg))
        self.data["confidence"]["data"].append(int(msg.data))

    def _human_capsule_cb(self, msg: Float32MultiArray):
        self._append_array_msg("human_capsules_robot", msg)

    def _cbf_diag_cb(self, msg: Float32MultiArray):
        self._append_array_msg("cbf_diagnostics", msg)

    def _cbf_min_pair_cb(self, msg: String):
        if not self._accept():
            return
        self.data["cbf_min_control_pair"]["t"].append(self._relative_t())
        self.data["cbf_min_control_pair"]["ros_t"].append(ros_now_sec(self))
        self.data["cbf_min_control_pair"]["data"].append(str(msg.data))

    def _timer_cb(self):
        if not self.started or self.finished:
            return

        elapsed = self._relative_t()
        if elapsed >= self.args.duration:
            self.finished = True
            self.get_logger().info("Requested duration reached. Saving trial data...")
            self.save_all()
            self.print_summary()
            # Do not call rclpy.shutdown() here; main loop handles exit.

    def save_all(self):
        self._save_topics_npz()
        self._save_run_yaml()
        self._save_manifest_json()

    def _save_topics_npz(self):
        path = self.outdir / "topics.npz"
        save_dict: Dict[str, Any] = {}

        for key, item in self.data.items():
            save_dict[f"{key}_t"] = np.asarray(item["t"], dtype=np.float64)
            save_dict[f"{key}_ros_t"] = np.asarray(item["ros_t"], dtype=np.float64)

            if key in [
                "q_nom",
                "q_cbf",
                "skeleton_filtered",
                "human_capsules_robot",
                "cbf_diagnostics",
            ]:
                save_dict[f"{key}_data"] = stack_or_object(item["data"])

            elif key == "confidence":
                save_dict[f"{key}_data"] = np.asarray(item["data"], dtype=np.int32)

            elif key == "q_act":
                save_dict[f"{key}_name"] = np.asarray(item["name"], dtype=object)
                save_dict[f"{key}_position"] = np.asarray(item["position"], dtype=object)
                save_dict[f"{key}_velocity"] = np.asarray(item["velocity"], dtype=object)

            elif key == "cbf_min_control_pair":
                save_dict[f"{key}_data"] = np.asarray(item["data"], dtype=object)

        np.savez_compressed(path, **save_dict)

    def _save_run_yaml(self):
        path = self.outdir / "run.yaml"

        lines = []
        lines.append(f"run_id: {self.args.run_id}")
        lines.append(f"scenario_id: {self.args.scenario}")
        lines.append(f"platform: {self.args.platform}")
        lines.append(f"mode: {self.args.mode}")
        lines.append(f"duration_sec: {self.args.duration:.6f}")
        lines.append(f"created_unix_time: {time.time():.6f}")
        lines.append(f"created_readable_time: \"{time.strftime('%Y-%m-%d %H:%M:%S')}\"")
        lines.append("")
        lines.append("topics:")
        lines.append(f"  q_nom: {self.args.q_nom_topic}")
        lines.append(f"  q_cbf: {self.args.q_cbf_topic}")
        lines.append(f"  q_act: {self.args.q_act_topic}")
        lines.append(f"  skeleton_filtered: {self.args.skeleton_filtered_topic}")
        lines.append(f"  confidence: {self.args.confidence_topic}")
        lines.append(f"  human_capsules_robot: {self.args.human_capsule_topic}")
        lines.append(f"  cbf_diagnostics: {self.args.cbf_diagnostics_topic}")
        lines.append(f"  cbf_min_control_pair: {self.args.cbf_min_pair_topic}")
        lines.append("")
        lines.append("parameters:")
        lines.append(f"  rr_safety_distance: {self.args.rr_safety_distance}")
        lines.append(f"  hr_safety_distance: {self.args.hr_safety_distance}")
        lines.append(f"  rr_gamma: {self.args.rr_gamma}")
        lines.append(f"  hr_gamma: {self.args.hr_gamma}")
        lines.append(f"  lpf_gain: {self.args.lpf_gain}")
        lines.append("")
        lines.append("flags:")
        lines.append(f"  record_q_act: {str(self.args.record_q_act).lower()}")
        lines.append(f"  record_cbf_diagnostics: {str(self.args.record_cbf_diagnostics).lower()}")
        lines.append("")
        lines.append("counts:")
        for key, item in self.data.items():
            lines.append(f"  {key}_frames: {len(item['t'])}")
        lines.append("")
        lines.append(f"notes: \"{self.args.notes}\"")
        lines.append("")

        path.write_text("\n".join(lines))

    def _save_manifest_json(self):
        path = self.outdir / "manifest.json"

        manifest = {
            "run_id": self.args.run_id,
            "scenario_id": self.args.scenario,
            "platform": self.args.platform,
            "mode": self.args.mode,
            "outdir": str(self.outdir),
            "duration_sec": self.args.duration,
            "created_unix_time": time.time(),
            "topics": {
                "q_nom": self.args.q_nom_topic,
                "q_cbf": self.args.q_cbf_topic,
                "q_act": self.args.q_act_topic,
                "skeleton_filtered": self.args.skeleton_filtered_topic,
                "confidence": self.args.confidence_topic,
                "human_capsules_robot": self.args.human_capsule_topic,
                "cbf_diagnostics": self.args.cbf_diagnostics_topic,
                "cbf_min_control_pair": self.args.cbf_min_pair_topic,
            },
            "parameters": {
                "rr_safety_distance": self.args.rr_safety_distance,
                "hr_safety_distance": self.args.hr_safety_distance,
                "rr_gamma": self.args.rr_gamma,
                "hr_gamma": self.args.hr_gamma,
                "lpf_gain": self.args.lpf_gain,
            },
            "counts": {
                key: len(item["t"]) for key, item in self.data.items()
            },
            "notes": self.args.notes,
        }

        path.write_text(json.dumps(manifest, indent=2))

    def print_summary(self):
        self.get_logger().info("========== Trial logging summary ==========")
        self.get_logger().info(f"Output: {self.outdir}")
        for key, item in self.data.items():
            self.get_logger().info(f"{key} frames: {len(item['t'])}")
        self.get_logger().info("Files written:")
        self.get_logger().info("  topics.npz")
        self.get_logger().info("  run.yaml")
        self.get_logger().info("  manifest.json")
        self.get_logger().info("===========================================")


def default_topics_for_platform(platform: str) -> Dict[str, str]:
    if platform == "real":
        return {
            "q_cbf_topic": "/real/joint_commands",
            "q_act_topic": "/real/joint_states",
            "human_capsule_topic": "/real/human_capsules_robot",
            "cbf_diagnostics_topic": "/real/cbf/diagnostics",
            "cbf_min_pair_topic": "/real/cbf/min_control_pair",
        }

    if platform == "sim":
        return {
            "q_cbf_topic": "/sim/joint_commands",
            "q_act_topic": "/sim/joint_states",
            "human_capsule_topic": "/sim/human_capsules_robot",
            "cbf_diagnostics_topic": "/sim/cbf/diagnostics",
            "cbf_min_pair_topic": "/sim/cbf/min_control_pair",
        }

    raise ValueError(f"Unsupported platform: {platform}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log raw topics for one BDCC replay trial. No FK or metrics are computed here."
    )

    parser.add_argument(
        "--platform",
        choices=["real", "sim"],
        required=True,
        help="Select default topic set.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario ID, e.g. S1_self_collision or S2_human_robot.",
    )
    parser.add_argument(
        "--mode",
        default="cbf",
        choices=["cbf", "nominal", "sweep"],
        help="Run mode label saved into metadata.",
    )
    parser.add_argument(
        "--run-id",
        default="run_001",
        help="Run ID saved into metadata.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for this run.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=45.0,
        help="Logging duration in seconds. Usually replay duration + start-delay buffer.",
    )
    parser.add_argument(
        "--start-on-enter",
        action="store_true",
        help="Wait for Enter before starting logger.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes saved into run.yaml.",
    )

    # Shared topics
    parser.add_argument("--q-nom-topic", default="/joint_commands_unsafe")
    parser.add_argument("--skeleton-filtered-topic", default="/skeleton/points_filtered")
    parser.add_argument("--confidence-topic", default="/skeleton/confidence")

    # Platform topics, default is filled after parse if user does not override.
    parser.add_argument("--q-cbf-topic", default=None)
    parser.add_argument("--q-act-topic", default=None)
    parser.add_argument("--human-capsule-topic", default=None)
    parser.add_argument("--cbf-diagnostics-topic", default=None)
    parser.add_argument("--cbf-min-pair-topic", default=None)

    parser.add_argument(
        "--record-q-act",
        action="store_true",
        help="Record /sim/joint_states or /real/joint_states. Optional.",
    )
    parser.add_argument(
        "--record-cbf-diagnostics",
        action="store_true",
        help="Record CBF diagnostics topics.",
    )

    # Metadata only. These should match launch parameters.
    parser.add_argument("--rr-safety-distance", type=float, default=np.nan)
    parser.add_argument("--hr-safety-distance", type=float, default=np.nan)
    parser.add_argument("--rr-gamma", type=float, default=np.nan)
    parser.add_argument("--hr-gamma", type=float, default=np.nan)
    parser.add_argument("--lpf-gain", type=float, default=1.0)

    parser.add_argument("--qos", type=int, default=50)

    args = parser.parse_args()

    defaults = default_topics_for_platform(args.platform)
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.start_on_enter:
        print("")
        print("======================================================")
        print("BDCC trial topic logger")
        print(f"Platform: {args.platform}")
        print(f"Scenario: {args.scenario}")
        print(f"Output:   {outdir}")
        print(f"Duration: {args.duration:.3f} s")
        print("Press Enter to start logging.")
        print("======================================================")
        input()

    rclpy.init(args=None)
    node = TrialTopicLogger(args)

    try:
        node.start()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt received. Saving partial trial data...")
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