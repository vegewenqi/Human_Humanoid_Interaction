#!/usr/bin/env python3
"""Run BDCC2026 CBF-QP parameter sweeps.

Typical real-robot collection, without offline metrics:

python3 /ws/bdcc_exp/scripts/sweep/run_parameter_sweep.py \
  --platform real \
  --sweep-type phi_grid \
  --segment /ws/bdcc_exp/segments/S2_human_robot_2 \
  --out-root /ws/bdcc_exp/sweeps/real_merge_phi_grid \
  --repeats 3 \
  --launch-wait-sec 10 \
  --shutdown-wait-sec 10 \
  --max-runs-per-invocation 1 \
  --duration 70 \
  --eval-start-sec 10 \
  --eval-end-sec 62 \
  --rviz false

Add --run-offline only when metrics should be computed immediately after each
run. The normal real-robot workflow is to collect first, then use
aggregate_sweep_results.py --compute-missing for offline metrics.

Shutdown note: by default the sweep stops launch/replay/logger with SIGINT
first, matching Ctrl-C. This is important for the real G1 arm bridge, whose
orderly return-home/release sequence is entered from its SIGINT handler.

Supervised pause/stop:
  touch <out-root>/PAUSE_SWEEP   # pause before the next run starts
  rm <out-root>/PAUSE_SWEEP      # continue
  touch <out-root>/STOP_SWEEP    # stop before the next run starts

Use Ctrl-C for immediate safe interruption of the current run.

By default this script runs only one non-completed run per invocation. Re-run
the same command to advance to the next repeat/parameter point through the
existing skip/resume logic.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_SEGMENT = "/ws/bdcc_exp/segments/S2_human_robot_2"
DEFAULT_URDF_PATH = "/ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf"
DEFAULT_BDCC_ROOT = Path("/ws/bdcc_exp")
LOCAL_BDCC_ROOT = Path(__file__).resolve().parents[2]

PHI_RR_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04]
PHI_HR_VALUES = [0.09, 0.12, 0.15, 0.17, 0.19]
GAMMA_RR_VALUES = [0.5, 1.0, 2.0, 3.0, 3.5]
GAMMA_HR_VALUES = [1.5, 2.0, 3.0, 4.0, 4.5]

PARETO_CANDIDATES = [
    (0.005, 0.09, 0.5, 1.5),
    (0.01, 0.12, 1.0, 2.0),
    (0.02, 0.15, 2.0, 3.0),
    (0.03, 0.15, 2.0, 3.0),
    (0.04, 0.19, 3.5, 4.5),
    (0.04, 0.17, 2.0, 3.0),
    (0.03, 0.19, 2.0, 4.5),
    (0.005, 0.19, 0.5, 4.5),
    (0.04, 0.09, 3.5, 1.5),
    (0.02, 0.17, 3.0, 4.0),
    (0.010, 0.135, 2.0, 3.0),
    (0.020, 0.135, 2.0, 3.0),
    (0.030, 0.135, 2.0, 3.0),
    (0.020, 0.145, 2.0, 3.0),
    (0.030, 0.145, 2.0, 3.0),
    (0.010, 0.150, 3.0, 4.0),
    (0.020, 0.150, 3.0, 4.0),
    (0.015, 0.150, 2.0, 3.0),
    (0.015, 0.170, 2.0, 3.0),
    (0.030, 0.170, 3.0, 4.0),
]

REAL_CRITICAL_NODE_NAMES = [
    "/zed_skeleton_points_preprocessor",
    "/human_skeleton_capsule",
    "/human_capsule_frame_transform_real",
    "/human_angle_estimator",
    "/g1_joint_mapper",
    "/g1_cbf_node_real",
    "/jointstate_to_array_qdes_real",
    "/g1_arm_sdk_bridge_real",
    "/robot_state_publisher_real",
    "/bdcc_trial_topic_logger",
    "/bdcc_replay_skeleton_segment",
]

SIM_CRITICAL_NODE_NAMES = [
    "/zed_skeleton_points_preprocessor",
    "/human_skeleton_capsule",
    "/human_capsule_frame_transform_sim",
    "/human_angle_estimator",
    "/g1_joint_mapper",
    "/g1_cbf_node_sim",
    "/jointstate_to_array_qdes_sim",
    "/g1_controller_sim",
    "/robot_state_publisher_sim",
    "/ghost_publisher_node",
    "/ghost_robot_state_publisher",
    "/bdcc_trial_topic_logger",
    "/bdcc_replay_skeleton_segment",
]

REAL_CRITICAL_COMMAND_TOPICS = [
    "/arm_sdk",
    "/joint_commands_unsafe",
    "/g1_upperbody_q_des",
    "/real/joint_commands",
    "/real/g1_upperbody_q_des_safe",
]

SIM_CRITICAL_COMMAND_TOPICS = [
    "/joint_commands_unsafe",
    "/g1_upperbody_q_des",
    "/sim/joint_commands",
    "/sim/g1_upperbody_q_des_safe",
]


@dataclass(frozen=True)
class SweepParams:
    phi_rr: float
    phi_hr: float
    gamma_rr: float
    gamma_hr: float

    @property
    def key(self) -> str:
        return (
            f"prr_{format_key_float(self.phi_rr, 3)}_"
            f"phr_{format_key_float(self.phi_hr, 3)}_"
            f"grr_{format_key_float(self.gamma_rr, 1)}_"
            f"ghr_{format_key_float(self.gamma_hr, 1)}"
        )


class SweepRunFailed(RuntimeError):
    pass


class SweepStopRequested(RuntimeError):
    pass


class RosGraphNotClear(SweepRunFailed):
    pass


def format_key_float(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace("-", "m").replace(".", "p")


def str_to_bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {text}")


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def signal_from_name(name: str) -> signal.Signals:
    lowered = str(name).strip().lower()
    if lowered in {"int", "sigint", "ctrl-c", "ctrl_c"}:
        return signal.SIGINT
    if lowered in {"term", "sigterm"}:
        return signal.SIGTERM
    raise ValueError(f"Unsupported signal name: {name}")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def scenario_from_segment(segment: str) -> str:
    return Path(segment).expanduser().name


def default_script_path(relative: str) -> Path:
    ws_path = DEFAULT_BDCC_ROOT / relative
    if ws_path.exists():
        return ws_path
    return LOCAL_BDCC_ROOT / relative


def shell_join(cmd: Iterable[str]) -> str:
    return shlex.join([str(x) for x in cmd])


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True))


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_sweep(sweep_type: str) -> List[SweepParams]:
    if sweep_type == "phi_grid":
        return [
            SweepParams(phi_rr=prr, phi_hr=phr, gamma_rr=2.0, gamma_hr=3.0)
            for prr in PHI_RR_VALUES
            for phr in PHI_HR_VALUES
        ]

    if sweep_type == "gamma_grid":
        return [
            SweepParams(phi_rr=0.03, phi_hr=0.15, gamma_rr=grr, gamma_hr=ghr)
            for grr in GAMMA_RR_VALUES
            for ghr in GAMMA_HR_VALUES
        ]

    if sweep_type == "pareto_samples":
        return [SweepParams(*vals) for vals in PARETO_CANDIDATES]

    raise ValueError(f"Unsupported sweep type: {sweep_type}")


def build_launch_cmd(args: argparse.Namespace, params: SweepParams) -> List[str]:
    run_sim = args.platform == "sim"
    run_real = args.platform == "real"
    cmd = [
        "ros2",
        "launch",
        args.launch_package,
        args.launch_file,
        f"run_sim:={bool_arg(run_sim)}",
        f"run_real:={bool_arg(run_real)}",
        "use_cbf:=true",
        f"rviz:={bool_arg(args.rviz)}",
    ]

    prefix = "real" if args.platform == "real" else "sim"
    cmd.extend(
        [
            f"{prefix}_rr_safety_distance:={params.phi_rr}",
            f"{prefix}_hr_safety_distance:={params.phi_hr}",
            f"{prefix}_rr_gamma:={params.gamma_rr}",
            f"{prefix}_hr_gamma:={params.gamma_hr}",
        ]
    )
    return cmd


def build_logger_cmd(
    args: argparse.Namespace,
    params: SweepParams,
    run_dir: Path,
    run_id: str,
    scenario: str,
) -> List[str]:
    return [
        sys.executable,
        str(args.logger_script),
        "--platform",
        args.platform,
        "--scenario",
        scenario,
        "--mode",
        "sweep",
        "--run-id",
        run_id,
        "--outdir",
        str(run_dir),
        "--duration",
        str(args.duration),
        "--record-q-act",
        "--record-cbf-diagnostics",
        "--rr-safety-distance",
        str(params.phi_rr),
        "--hr-safety-distance",
        str(params.phi_hr),
        "--rr-gamma",
        str(params.gamma_rr),
        "--hr-gamma",
        str(params.gamma_hr),
    ]


def build_replay_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(args.replay_script),
        "--segment",
        args.segment,
        "--publish-mode",
        "filtered",
        "--start-delay",
        "0.0",
        "--replay-rate-hz",
        "60",
        "--time-scale",
        "1.0",
    ]


def build_offline_cmd(args: argparse.Namespace, run_dir: Path) -> List[str]:
    return [
        sys.executable,
        str(args.offline_script),
        "--run-dir",
        str(run_dir),
        "--urdf-path",
        args.urdf_path,
        "--mode",
        "both",
        "--sample-rate-hz",
        str(args.sample_rate_hz),
        "--max-lag-sec",
        str(args.max_lag_sec),
        "--lag-step-sec",
        str(args.lag_step_sec),
        "--eval-start-sec",
        str(args.eval_start_sec),
        "--eval-end-sec",
        str(args.eval_end_sec),
    ]


def popen_group(cmd: List[str]) -> subprocess.Popen:
    print(f"[sweep] start: {shell_join(cmd)}", flush=True)
    return subprocess.Popen(cmd, start_new_session=True)


def process_is_running(proc: Optional[subprocess.Popen]) -> bool:
    return proc is not None and proc.poll() is None


def send_process_group_signal(
    proc: Optional[subprocess.Popen],
    name: str,
    sig: signal.Signals,
) -> None:
    if not process_is_running(proc):
        return
    print(
        f"[sweep] sending {sig.name} to {name} process group (pid={proc.pid})",
        flush=True,
    )
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def wait_process_group_exit(
    proc: Optional[subprocess.Popen],
    name: str,
    first_signal: signal.Signals = signal.SIGINT,
    graceful_timeout_sec: float = 10.0,
    term_timeout_sec: float = 5.0,
    kill_timeout_sec: float = 5.0,
) -> Optional[int]:
    if proc is None:
        return None
    if proc.poll() is not None:
        return int(proc.returncode)

    print(
        f"[sweep] waiting for {name} after {first_signal.name} "
        f"({graceful_timeout_sec:.1f}s)",
        flush=True,
    )
    try:
        proc.wait(timeout=graceful_timeout_sec)
    except subprocess.TimeoutExpired:
        if first_signal != signal.SIGTERM:
            print(
                f"[sweep] {name} did not exit after {first_signal.name}; "
                f"sending SIGTERM (wait={term_timeout_sec:.1f}s)",
                flush=True,
            )
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=term_timeout_sec)
            except subprocess.TimeoutExpired:
                pass

    if proc.poll() is None:
        print(
            f"[sweep] killing {name} with SIGKILL "
            f"(wait={kill_timeout_sec:.1f}s)",
            flush=True,
        )
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=kill_timeout_sec)

    return int(proc.returncode)


def stop_process_group(
    proc: Optional[subprocess.Popen],
    name: str,
    first_signal: signal.Signals = signal.SIGINT,
    graceful_timeout_sec: float = 10.0,
    term_timeout_sec: float = 5.0,
    kill_timeout_sec: float = 5.0,
) -> Optional[int]:
    send_process_group_signal(proc, name, first_signal)
    return wait_process_group_exit(
        proc,
        name,
        first_signal=first_signal,
        graceful_timeout_sec=graceful_timeout_sec,
        term_timeout_sec=term_timeout_sec,
        kill_timeout_sec=kill_timeout_sec,
    )


def stop_process_groups(
    processes: List[tuple[str, Optional[subprocess.Popen]]],
    first_signal: signal.Signals = signal.SIGINT,
    graceful_timeout_sec: float = 10.0,
    term_timeout_sec: float = 5.0,
    kill_timeout_sec: float = 5.0,
) -> Dict[str, Optional[int]]:
    for name, proc in processes:
        send_process_group_signal(proc, name, first_signal)

    return_codes: Dict[str, Optional[int]] = {}
    for name, proc in processes:
        return_codes[name] = wait_process_group_exit(
            proc,
            name,
            first_signal=first_signal,
            graceful_timeout_sec=graceful_timeout_sec,
            term_timeout_sec=term_timeout_sec,
            kill_timeout_sec=kill_timeout_sec,
        )
    return return_codes


def cleanup_process_groups(
    args: argparse.Namespace,
    processes: List[tuple[str, Optional[subprocess.Popen]]],
) -> Dict[str, Optional[int]]:
    first_signal = signal_from_name(args.shutdown_signal)
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        return stop_process_groups(
            processes,
            first_signal=first_signal,
            graceful_timeout_sec=args.shutdown_wait_sec,
            term_timeout_sec=args.term_wait_sec,
            kill_timeout_sec=args.kill_wait_sec,
        )
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def run_capture(cmd: List[str], timeout_sec: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )


def ros2_node_list(args: argparse.Namespace) -> List[str]:
    try:
        proc = run_capture(["ros2", "node", "list"], args.ros_graph_cmd_timeout_sec)
    except FileNotFoundError as exc:
        raise RosGraphNotClear("ros2 command not found; source the ROS workspace first") from exc
    except subprocess.TimeoutExpired as exc:
        raise RosGraphNotClear("ros2 node list timed out") from exc

    if proc.returncode != 0:
        raise RosGraphNotClear(
            "ros2 node list failed: " + (proc.stderr.strip() or proc.stdout.strip())
        )

    nodes = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("/"):
            nodes.append(line)
    return nodes


def ros2_topic_publisher_count(args: argparse.Namespace, topic: str) -> int:
    try:
        proc = run_capture(["ros2", "topic", "info", topic], args.ros_graph_cmd_timeout_sec)
    except FileNotFoundError as exc:
        raise RosGraphNotClear("ros2 command not found; source the ROS workspace first") from exc
    except subprocess.TimeoutExpired as exc:
        raise RosGraphNotClear(f"ros2 topic info {topic} timed out") from exc

    output = "\n".join([proc.stdout, proc.stderr])
    if proc.returncode != 0:
        unknown_markers = [
            "Unknown topic",
            "Unable to find topic",
            "not found",
            "does not appear to be published",
        ]
        if any(marker.lower() in output.lower() for marker in unknown_markers):
            return 0
        # Some ROS 2 distros return a nonzero code when a topic has no endpoint
        # yet. Treat missing Publisher count as zero only if no node currently
        # reports it.
        if "Publisher count:" not in output:
            return 0

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Publisher count:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                break
    return 0


def critical_node_names(args: argparse.Namespace) -> List[str]:
    base = REAL_CRITICAL_NODE_NAMES if args.platform == "real" else SIM_CRITICAL_NODE_NAMES
    extra = [x.strip() for x in args.extra_critical_nodes if x.strip()]
    return base + extra


def critical_command_topics(args: argparse.Namespace) -> List[str]:
    base = REAL_CRITICAL_COMMAND_TOPICS if args.platform == "real" else SIM_CRITICAL_COMMAND_TOPICS
    extra = [x.strip() for x in args.extra_critical_topics if x.strip()]
    return base + extra


def ros_graph_blockers(args: argparse.Namespace) -> List[str]:
    blockers: List[str] = []
    topic_baseline: Dict[str, int] = getattr(args, "ros_topic_baseline", {})

    nodes = ros2_node_list(args)
    node_counts: Dict[str, int] = {}
    for node in nodes:
        node_counts[node] = node_counts.get(node, 0) + 1

    for node_name in critical_node_names(args):
        count = node_counts.get(node_name, 0)
        if count > 0:
            blockers.append(f"node {node_name} still present (count={count})")

    for topic in critical_command_topics(args):
        pub_count = ros2_topic_publisher_count(args, topic)
        baseline_count = int(topic_baseline.get(topic, 0))
        if pub_count > baseline_count:
            blockers.append(
                f"topic {topic} has {pub_count} publisher(s), "
                f"baseline before this invocation was {baseline_count}"
            )

    return blockers


def capture_ros_topic_baseline(args: argparse.Namespace) -> Dict[str, int]:
    if not args.verify_ros_graph or args.dry_run:
        return {}

    baseline: Dict[str, int] = {}
    for topic in critical_command_topics(args):
        baseline[topic] = ros2_topic_publisher_count(args, topic)
    return baseline


def assert_ros_graph_clear(args: argparse.Namespace, label: str) -> None:
    if not args.verify_ros_graph:
        return

    deadline = time.monotonic() + max(0.0, args.ros_graph_clear_timeout_sec)
    last_blockers: List[str] = []
    while True:
        last_blockers = ros_graph_blockers(args)
        if not last_blockers:
            print(f"[sweep] ROS graph clear: {label}", flush=True)
            return
        if time.monotonic() >= deadline:
            break
        print(
            f"[sweep] waiting for ROS graph to clear ({label}): "
            + "; ".join(last_blockers),
            flush=True,
        )
        time.sleep(max(0.1, args.ros_graph_check_period_sec))

    raise RosGraphNotClear(f"{label}: " + "; ".join(last_blockers))


def warn_if_ros_graph_not_clear(args: argparse.Namespace, label: str) -> None:
    if not args.verify_ros_graph:
        return
    try:
        assert_ros_graph_clear(args, label)
    except Exception as exc:
        print(f"[sweep] WARNING: ROS graph not clear after cleanup: {exc}", file=sys.stderr)


def sleep_while_launch_alive(proc: subprocess.Popen, seconds: float) -> Optional[int]:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            return int(rc)
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    return None


def control_file_path(args: argparse.Namespace, path_value: Optional[str], filename: str) -> Path:
    if path_value:
        return Path(path_value).expanduser()
    return args.out_root / filename


def stop_file_exists(args: argparse.Namespace) -> bool:
    return control_file_path(args, args.stop_file, "STOP_SWEEP").exists()


def pause_file_exists(args: argparse.Namespace) -> bool:
    return control_file_path(args, args.pause_file, "PAUSE_SWEEP").exists()


def wait_for_pause_file_removal(args: argparse.Namespace) -> None:
    pause_path = control_file_path(args, args.pause_file, "PAUSE_SWEEP")
    stop_path = control_file_path(args, args.stop_file, "STOP_SWEEP")
    if not pause_path.exists():
        return

    print("")
    print(f"[sweep] pause requested: {pause_path}")
    print(f"[sweep] remove the pause file to continue, or create {stop_path} to stop.")
    while pause_path.exists():
        if stop_path.exists():
            raise SweepStopRequested(f"stop file exists: {stop_path}")
        time.sleep(1.0)
    print("[sweep] pause file removed; continuing.", flush=True)


def wait_for_enter_if_requested(args: argparse.Namespace) -> None:
    if not args.pause_between_runs:
        return
    if args.dry_run:
        return
    input("[sweep] pause-between-runs enabled. Press Enter to start the next run...")


def check_manual_stop_or_pause(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    stop_path = control_file_path(args, args.stop_file, "STOP_SWEEP")
    if stop_path.exists():
        raise SweepStopRequested(f"stop file exists: {stop_path}")
    wait_for_pause_file_removal(args)


def completed_run_exists(run_dir: Path) -> bool:
    cfg = read_json(run_dir / "sweep_run_config.json")
    if not cfg:
        return False
    return cfg.get("status") == "completed" and (run_dir / "topics.npz").exists()


def archive_incomplete_run(run_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_dir.with_name(f"{run_dir.name}_failed_{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = run_dir.with_name(f"{run_dir.name}_failed_{stamp}_{suffix:02d}")
        suffix += 1
    run_dir.rename(candidate)
    return candidate


def prepare_run_dir(args: argparse.Namespace, run_dir: Path) -> None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    if completed_run_exists(run_dir):
        return

    if not args.overwrite_incomplete:
        raise SweepRunFailed(
            f"Incomplete run exists and --overwrite-incomplete is false: {run_dir}"
        )

    archived = archive_incomplete_run(run_dir)
    print(f"[sweep] archived incomplete run: {archived}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)


def initial_config(
    args: argparse.Namespace,
    params: SweepParams,
    run_dir: Path,
    run_id: str,
    repeat_id: int,
    scenario: str,
    commands: Dict[str, List[str]],
) -> Dict[str, Any]:
    return {
        "status": "pending",
        "platform": args.platform,
        "sweep_type": args.sweep_type,
        "scenario": scenario,
        "segment": args.segment,
        "out_root": str(args.out_root),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "repeat_id": repeat_id,
        "param_key": params.key,
        "parameters": {
            "phi_rr": params.phi_rr,
            "phi_hr": params.phi_hr,
            "gamma_rr": params.gamma_rr,
            "gamma_hr": params.gamma_hr,
            "rr_safety_distance": params.phi_rr,
            "hr_safety_distance": params.phi_hr,
            "rr_gamma": params.gamma_rr,
            "hr_gamma": params.gamma_hr,
        },
        "timing": {
            "launch_wait_sec": args.launch_wait_sec,
            "reset_wait_sec": args.reset_wait_sec,
            "shutdown_wait_sec": args.shutdown_wait_sec,
            "term_wait_sec": args.term_wait_sec,
            "kill_wait_sec": args.kill_wait_sec,
            "duration_sec": args.duration,
            "eval_start_sec": args.eval_start_sec,
            "eval_end_sec": args.eval_end_sec,
        },
        "shutdown": {
            "first_signal": args.shutdown_signal,
            "note": "launch/replay/logger are stopped with SIGINT by default to match Ctrl-C safe shutdown",
        },
        "ros_graph_verification": {
            "enabled": bool(args.verify_ros_graph),
            "clear_timeout_sec": args.ros_graph_clear_timeout_sec,
            "critical_nodes": critical_node_names(args) if args.verify_ros_graph else [],
            "critical_command_topics": critical_command_topics(args) if args.verify_ros_graph else [],
            "topic_publisher_baseline": getattr(args, "ros_topic_baseline", {}),
        },
        "offline": {
            "run_offline": bool(args.run_offline),
            "sample_rate_hz": args.sample_rate_hz,
            "max_lag_sec": args.max_lag_sec,
            "lag_step_sec": args.lag_step_sec,
            "urdf_path": args.urdf_path,
            "mode": "both",
        },
        "commands": {k: [str(x) for x in v] for k, v in commands.items()},
        "command_strings": {k: shell_join(v) for k, v in commands.items()},
        "timestamps": {
            "created": now_iso(),
            "started": None,
            "completed": None,
            "failed": None,
        },
        "return_codes": {
            "launch": None,
            "logger": None,
            "replay": None,
            "offline": None,
        },
        "warnings": [],
        "failure_reason": None,
    }


def mark_failed(config: Dict[str, Any], reason: str, config_path: Path) -> None:
    config["status"] = "failed"
    config["failure_reason"] = reason
    config["timestamps"]["failed"] = now_iso()
    write_json(config_path, config)


def run_one(args: argparse.Namespace, params: SweepParams, repeat_id: int) -> bool:
    run_id = f"run_{repeat_id:03d}"
    scenario = scenario_from_segment(args.segment)
    param_dir = args.out_root / params.key
    run_dir = param_dir / run_id

    launch_cmd = build_launch_cmd(args, params)
    logger_cmd = build_logger_cmd(args, params, run_dir, run_id, scenario)
    replay_cmd = build_replay_cmd(args)
    offline_cmd = build_offline_cmd(args, run_dir)
    commands = {
        "launch": launch_cmd,
        "logger": logger_cmd,
        "replay": replay_cmd,
    }
    if args.run_offline:
        commands["offline"] = offline_cmd

    if args.dry_run:
        print("")
        print(f"[dry-run] {params.key}/{run_id}")
        print(f"[dry-run] run_dir: {run_dir}")
        for name, cmd in commands.items():
            print(f"[dry-run] {name}: {shell_join(cmd)}")
        return True

    if args.skip_existing_completed and completed_run_exists(run_dir):
        print(f"[sweep] skip completed: {run_dir}", flush=True)
        return False

    assert_ros_graph_clear(args, "preflight before starting run")

    prepare_run_dir(args, run_dir)

    config_path = run_dir / "sweep_run_config.json"
    config = initial_config(args, params, run_dir, run_id, repeat_id, scenario, commands)
    write_json(config_path, config)

    launch_proc: Optional[subprocess.Popen] = None
    replay_proc: Optional[subprocess.Popen] = None
    logger_proc: Optional[subprocess.Popen] = None
    should_reset_sleep = False

    try:
        print("")
        print("=" * 78)
        print(
            f"[sweep] running {args.sweep_type} {params.key} {run_id} "
            f"({repeat_id}/{args.repeats})",
            flush=True,
        )
        print("=" * 78)

        config["status"] = "running"
        config["timestamps"]["started"] = now_iso()
        write_json(config_path, config)

        launch_proc = popen_group(launch_cmd)
        should_reset_sleep = True
        launch_early_rc = sleep_while_launch_alive(launch_proc, args.launch_wait_sec)
        if launch_early_rc is not None:
            config["return_codes"]["launch"] = launch_early_rc
            raise SweepRunFailed(
                f"Launch exited during launch wait with return code {launch_early_rc}"
            )

        logger_proc = popen_group(logger_cmd)
        replay_proc = popen_group(replay_cmd)

        logger_rc = int(logger_proc.wait())
        config["return_codes"]["logger"] = logger_rc
        write_json(config_path, config)

        replay_pre_stop_rc = replay_proc.poll() if replay_proc is not None else None
        launch_pre_stop_rc = launch_proc.poll() if launch_proc is not None else None

        return_codes = cleanup_process_groups(
            args,
            [
                ("launch", launch_proc),
                ("replay", replay_proc),
            ],
        )
        config["return_codes"]["replay"] = return_codes.get("replay")
        config["return_codes"]["launch"] = return_codes.get("launch")
        write_json(config_path, config)

        if args.reset_wait_sec > 0:
            print(
                f"[sweep] waiting {args.reset_wait_sec:.1f}s for robot reset/stabilization",
                flush=True,
            )
            time.sleep(args.reset_wait_sec)

        assert_ros_graph_clear(args, "post-shutdown before next run")

        topics_path = run_dir / "topics.npz"
        if logger_rc != 0:
            raise SweepRunFailed(f"Logger returned non-zero code {logger_rc}")
        if launch_pre_stop_rc is not None and launch_pre_stop_rc != 0:
            raise SweepRunFailed(
                f"Launch exited before cleanup with non-zero code {launch_pre_stop_rc}"
            )
        if not topics_path.exists():
            raise SweepRunFailed(f"Missing topics.npz after logger completed: {topics_path}")
        if replay_pre_stop_rc is not None and replay_pre_stop_rc != 0:
            config["warnings"].append(
                f"Replay exited before cleanup with code {replay_pre_stop_rc}; "
                "topics.npz exists, so run is kept."
            )
        if launch_pre_stop_rc == 0:
            config["warnings"].append("Launch exited before cleanup with code 0.")

        if args.run_offline:
            print(f"[sweep] offline: {shell_join(offline_cmd)}", flush=True)
            offline = subprocess.run(offline_cmd)
            config["return_codes"]["offline"] = int(offline.returncode)
            write_json(config_path, config)
            if offline.returncode != 0:
                raise SweepRunFailed(
                    f"offline_compute_metrics.py returned non-zero code {offline.returncode}"
                )
            if not (run_dir / "metrics_summary.json").exists():
                raise SweepRunFailed("Offline compute finished but metrics_summary.json is missing")

        config["status"] = "completed"
        config["timestamps"]["completed"] = now_iso()
        write_json(config_path, config)
        print(f"[sweep] completed: {run_dir}", flush=True)
        return True

    except KeyboardInterrupt as exc:
        cleanup_process_groups(
            args,
            [
                ("launch", launch_proc),
                ("logger", logger_proc),
                ("replay", replay_proc),
            ],
        )
        warn_if_ros_graph_not_clear(args, "after interrupted cleanup")
        if should_reset_sleep and args.reset_wait_sec > 0:
            print(
                f"[sweep] interrupted; waiting {args.reset_wait_sec:.1f}s for reset/stabilization",
                flush=True,
            )
            time.sleep(args.reset_wait_sec)
        mark_failed(config, "Interrupted by user", config_path)
        raise SweepRunFailed("Interrupted by user") from exc

    except Exception as exc:
        cleanup_process_groups(
            args,
            [
                ("launch", launch_proc),
                ("logger", logger_proc),
                ("replay", replay_proc),
            ],
        )
        warn_if_ros_graph_not_clear(args, "after failure cleanup")
        if should_reset_sleep and args.reset_wait_sec > 0:
            print(
                f"[sweep] failure cleanup; waiting {args.reset_wait_sec:.1f}s for reset/stabilization",
                flush=True,
            )
            time.sleep(args.reset_wait_sec)
        reason = str(exc)
        mark_failed(config, reason, config_path)
        raise SweepRunFailed(reason) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BDCC real/sim CBF-QP parameter sweeps with resume support."
    )
    parser.add_argument("--platform", choices=["real", "sim"], default="real")
    parser.add_argument(
        "--sweep-type",
        choices=["phi_grid", "gamma_grid", "pareto_samples"],
        required=True,
    )
    parser.add_argument("--segment", default=DEFAULT_SEGMENT)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--launch-wait-sec", type=float, default=10.0)
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=None,
        help="Deprecated alias for --launch-wait-sec.",
    )
    parser.add_argument(
        "--reset-wait-sec",
        type=float,
        default=0.0,
        help="Optional wait after launch/replay have exited. Default 0 because reset is handled during SIGINT shutdown.",
    )
    parser.add_argument(
        "--shutdown-signal",
        choices=["sigint", "sigterm"],
        default="sigint",
        help="First signal used to stop replay/logger/launch. sigint matches Ctrl-C.",
    )
    parser.add_argument(
        "--shutdown-wait-sec",
        type=float,
        default=10.0,
        help="Seconds to wait after the first shutdown signal before fallback.",
    )
    parser.add_argument(
        "--term-wait-sec",
        type=float,
        default=5.0,
        help="Seconds to wait after SIGTERM fallback.",
    )
    parser.add_argument(
        "--kill-wait-sec",
        type=float,
        default=5.0,
        help="Seconds to wait after SIGKILL fallback.",
    )
    parser.add_argument(
        "--pause-between-runs",
        action="store_true",
        help="Wait for Enter before each next run. Useful for supervised real-robot sweeps.",
    )
    parser.add_argument(
        "--pause-file",
        default=None,
        help="If this file exists, pause before starting the next run. Default: <out-root>/PAUSE_SWEEP.",
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="If this file exists, stop before starting the next run. Default: <out-root>/STOP_SWEEP.",
    )
    parser.add_argument(
        "--max-runs-per-invocation",
        type=int,
        default=1,
        help="Maximum non-completed runs to execute before exiting. Use 0 to run the whole sweep.",
    )
    parser.add_argument(
        "--verify-ros-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Verify critical ROS nodes/topics are clear before/after each run. Default: true for real, false for sim.",
    )
    parser.add_argument(
        "--ros-graph-clear-timeout-sec",
        type=float,
        default=5.0,
        help="Seconds to wait for critical ROS nodes/topics to disappear after shutdown.",
    )
    parser.add_argument(
        "--ros-graph-check-period-sec",
        type=float,
        default=0.5,
        help="Polling period for ROS graph cleanup checks.",
    )
    parser.add_argument(
        "--ros-graph-cmd-timeout-sec",
        type=float,
        default=3.0,
        help="Timeout for individual ros2 node/topic inspection commands.",
    )
    parser.add_argument(
        "--extra-critical-nodes",
        nargs="*",
        default=[],
        help="Additional exact ROS node names that must not be present before/after runs.",
    )
    parser.add_argument(
        "--extra-critical-topics",
        nargs="*",
        default=[],
        help="Additional ROS topics whose publisher count must return to the invocation baseline.",
    )
    parser.add_argument("--duration", type=float, default=70.0)
    parser.add_argument("--eval-start-sec", type=float, default=10.0)
    parser.add_argument("--eval-end-sec", type=float, default=62.0)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--max-lag-sec", type=float, default=2.0)
    parser.add_argument("--lag-step-sec", type=float, default=0.02)
    parser.add_argument("--urdf-path", default=DEFAULT_URDF_PATH)
    parser.add_argument("--run-offline", action="store_true")
    parser.add_argument(
        "--skip-existing-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip run dirs with status=completed and topics.npz.",
    )
    parser.add_argument(
        "--overwrite-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Archive incomplete/failed run dirs and rerun them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rviz", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--launch-package", default="mujoco_g1")
    parser.add_argument("--launch-file", default="bdcc_unified_pipeline.launch.py")

    parser.add_argument(
        "--logger-script",
        type=Path,
        default=default_script_path("scripts/log/trial_topic_logger.py"),
        help="Path to trial_topic_logger.py.",
    )
    parser.add_argument(
        "--replay-script",
        type=Path,
        default=default_script_path("scripts/replay/replay_skeleton_segment.py"),
        help="Path to replay_skeleton_segment.py.",
    )
    parser.add_argument(
        "--offline-script",
        type=Path,
        default=default_script_path("scripts/offline/offline_compute_metrics.py"),
        help="Path to offline_compute_metrics.py.",
    )

    args = parser.parse_args()

    if args.warmup_sec is not None:
        print("[sweep] --warmup-sec is deprecated; using it as --launch-wait-sec")
        args.launch_wait_sec = args.warmup_sec

    if args.out_root is None:
        args.out_root = f"/ws/bdcc_exp/sweeps/{args.platform}_merge_{args.sweep_type}"
    args.out_root = Path(args.out_root).expanduser()

    if args.verify_ros_graph is None:
        args.verify_ros_graph = args.platform == "real"

    args.logger_script = Path(args.logger_script).expanduser()
    args.replay_script = Path(args.replay_script).expanduser()
    args.offline_script = Path(args.offline_script).expanduser()

    return args


def main() -> int:
    def _external_stop_handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _external_stop_handler)

    args = parse_args()
    sweep = build_sweep(args.sweep_type)

    print("[sweep] BDCC2026 parameter sweep")
    print(f"[sweep] platform: {args.platform}")
    print(f"[sweep] sweep_type: {args.sweep_type}")
    print(f"[sweep] segment: {args.segment}")
    print(f"[sweep] out_root: {args.out_root}")
    print(f"[sweep] repeats: {args.repeats}")
    print(f"[sweep] parameter points: {len(sweep)}")
    print(f"[sweep] run_offline: {args.run_offline}")
    print(f"[sweep] max_runs_per_invocation: {args.max_runs_per_invocation}")
    print(f"[sweep] verify_ros_graph: {args.verify_ros_graph}")
    print(f"[sweep] dry_run: {args.dry_run}")
    print(f"[sweep] pause_file: {control_file_path(args, args.pause_file, 'PAUSE_SWEEP')}")
    print(f"[sweep] stop_file: {control_file_path(args, args.stop_file, 'STOP_SWEEP')}")

    try:
        if args.dry_run:
            print("[sweep] dry run only; no directories or processes will be created.")
            args.ros_topic_baseline = {}
        else:
            args.out_root.mkdir(parents=True, exist_ok=True)
            args.ros_topic_baseline = capture_ros_topic_baseline(args)
            if args.verify_ros_graph:
                print("[sweep] ROS topic publisher baseline:")
                for topic, count in args.ros_topic_baseline.items():
                    print(f"[sweep]   {topic}: {count}")

        executed_runs = 0
        run_index = 0
        total_runs = len(sweep) * args.repeats
        for params in sweep:
            for repeat_id in range(1, args.repeats + 1):
                run_index += 1
                check_manual_stop_or_pause(args)
                did_execute = run_one(args, params, repeat_id)
                if did_execute:
                    executed_runs += 1
                    if args.max_runs_per_invocation > 0 and executed_runs >= args.max_runs_per_invocation:
                        print(
                            f"[sweep] reached --max-runs-per-invocation={args.max_runs_per_invocation}; exiting."
                        )
                        return 0
                if run_index < total_runs:
                    wait_for_enter_if_requested(args)
                    check_manual_stop_or_pause(args)
    except SweepStopRequested as exc:
        print(f"[sweep] STOP requested: {exc}", flush=True)
        print("[sweep] no new runs will be started.")
        return 0
    except KeyboardInterrupt:
        print("[sweep] interrupted between runs; no child launch is active.", file=sys.stderr)
        return 130
    except SweepRunFailed as exc:
        print(f"[sweep] FAILED: {exc}", file=sys.stderr, flush=True)
        print("[sweep] stopping sweep so robot/log state can be inspected.", file=sys.stderr)
        return 1

    print("[sweep] all requested runs handled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
