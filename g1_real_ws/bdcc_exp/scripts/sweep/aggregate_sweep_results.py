#!/usr/bin/env python3
"""Aggregate BDCC2026 sweep runs into per-run and per-parameter CSV files.

Typical post-processing after real-robot collection:

python3 /ws/bdcc_exp/scripts/sweep/aggregate_sweep_results.py \
  --sweep-root /ws/bdcc_exp/sweeps/real_merge_phi_grid \
  --compute-missing \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --eval-start-sec 10 \
  --eval-end-sec 62

The script keeps raw metrics unchanged, derives angle/MSE helper columns, and
adds optional composite scores S_safe and S_imit for plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_URDF_PATH = "/ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf"
DEFAULT_BDCC_ROOT = Path("/ws/bdcc_exp")
LOCAL_BDCC_ROOT = Path(__file__).resolve().parents[2]
RUN_RE = re.compile(r"^run_(\d+)$")

CORE_COLUMNS = [
    "platform",
    "sweep_type",
    "scenario",
    "run_id",
    "repeat_id",
    "param_key",
    "phi_rr",
    "phi_hr",
    "gamma_rr",
    "gamma_hr",
    "run_dir",
    "status",
]

METRIC_COLUMNS = [
    "merge_safe_M_clear_m",
    "merge_safe_M_ctr",
    "merge_safe_M_cc",
    "self_safe_M_clear_m",
    "self_safe_M_ctr",
    "self_safe_M_cc",
    "hr_safe_M_clear_m",
    "hr_safe_M_ctr",
    "hr_safe_M_cc",
    "rmse_q0_rad",
    "rmse_q0_deg",
    "mse_q0_rad",
    "mse_q0_deg",
    "safe_ndtw_link_rad_mean",
    "safe_ndtw_link_deg_mean",
    "unsafe_ndtw_link_rad_mean",
    "unsafe_ndtw_link_deg_mean",
    "mean_correction_norm_rad",
    "max_correction_norm_rad",
]

SCORE_COLUMNS = ["S_safe", "S_imit"]


def default_script_path(relative: str) -> Path:
    ws_path = DEFAULT_BDCC_ROOT / relative
    if ws_path.exists():
        return ws_path
    return LOCAL_BDCC_ROOT / relative


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[aggregate] warning: could not read {path}: {exc}", file=sys.stderr)
        return None


def as_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def shell_join(cmd: Iterable[str]) -> str:
    import shlex

    return shlex.join([str(x) for x in cmd])


def read_run_yaml_simple(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    out: Dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and ":" in raw_line:
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if value == "":
                current_section = key
                out.setdefault(current_section, {})
            else:
                out[key] = value
                current_section = None
        elif current_section and raw_line.startswith("  ") and ":" in raw_line:
            key, value = raw_line.split(":", 1)
            out.setdefault(current_section, {})[key.strip()] = value.strip().strip('"')
    return out


def parse_params_from_key(param_key: str) -> Dict[str, float]:
    # Example: prr_0p030_phr_0p150_grr_2p0_ghr_3p0
    parts = param_key.split("_")
    parsed: Dict[str, float] = {}
    for idx in range(0, len(parts) - 1, 2):
        name = parts[idx]
        value = parts[idx + 1].replace("m", "-").replace("p", ".")
        try:
            parsed[name] = float(value)
        except ValueError:
            continue
    return {
        "phi_rr": parsed.get("prr", math.nan),
        "phi_hr": parsed.get("phr", math.nan),
        "gamma_rr": parsed.get("grr", math.nan),
        "gamma_hr": parsed.get("ghr", math.nan),
    }


def find_run_dirs(sweep_root: Path) -> List[Path]:
    run_dirs = []
    for path in sweep_root.rglob("run_*"):
        if path.is_dir() and RUN_RE.match(path.name):
            run_dirs.append(path)
    return sorted(run_dirs)


def run_offline_compute(args: argparse.Namespace, run_dir: Path) -> int:
    cmd = [
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
    print(f"[aggregate] compute missing: {shell_join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def load_run_row(args: argparse.Namespace, run_dir: Path) -> Optional[Dict[str, Any]]:
    config = read_json(run_dir / "sweep_run_config.json") or {}
    manifest = read_json(run_dir / "manifest.json") or {}
    run_yaml = read_run_yaml_simple(run_dir / "run.yaml")

    metrics_path = run_dir / "metrics_summary.json"
    if args.compute_missing and not metrics_path.exists() and (run_dir / "topics.npz").exists():
        rc = run_offline_compute(args, run_dir)
        if rc != 0:
            print(
                f"[aggregate] warning: offline compute failed for {run_dir} with code {rc}",
                file=sys.stderr,
            )

    metrics = read_json(metrics_path) or {}
    if args.require_metrics and not metrics:
        print(f"[aggregate] skip without metrics: {run_dir}", file=sys.stderr)
        return None

    run_match = RUN_RE.match(run_dir.name)
    repeat_id_from_dir = int(run_match.group(1)) if run_match else math.nan
    params = config.get("parameters", {}) or {}
    manifest_params = manifest.get("parameters", {}) or {}
    yaml_params = run_yaml.get("parameters", {}) if isinstance(run_yaml.get("parameters"), dict) else {}

    param_key = first_present(config.get("param_key"), run_dir.parent.name, default="")
    parsed_key = parse_params_from_key(str(param_key))

    row: Dict[str, Any] = {
        "platform": first_present(
            config.get("platform"),
            manifest.get("platform"),
            run_yaml.get("platform"),
            default="",
        ),
        "sweep_type": first_present(config.get("sweep_type"), default=""),
        "scenario": first_present(
            config.get("scenario"),
            manifest.get("scenario_id"),
            run_yaml.get("scenario_id"),
            default="",
        ),
        "run_id": first_present(config.get("run_id"), manifest.get("run_id"), run_dir.name),
        "repeat_id": first_present(config.get("repeat_id"), repeat_id_from_dir),
        "param_key": param_key,
        "phi_rr": as_float(
            first_present(
                params.get("phi_rr"),
                params.get("rr_safety_distance"),
                manifest_params.get("rr_safety_distance"),
                yaml_params.get("rr_safety_distance"),
                parsed_key["phi_rr"],
            )
        ),
        "phi_hr": as_float(
            first_present(
                params.get("phi_hr"),
                params.get("hr_safety_distance"),
                manifest_params.get("hr_safety_distance"),
                yaml_params.get("hr_safety_distance"),
                parsed_key["phi_hr"],
            )
        ),
        "gamma_rr": as_float(
            first_present(
                params.get("gamma_rr"),
                params.get("rr_gamma"),
                manifest_params.get("rr_gamma"),
                yaml_params.get("rr_gamma"),
                parsed_key["gamma_rr"],
            )
        ),
        "gamma_hr": as_float(
            first_present(
                params.get("gamma_hr"),
                params.get("hr_gamma"),
                manifest_params.get("hr_gamma"),
                yaml_params.get("hr_gamma"),
                parsed_key["gamma_hr"],
            )
        ),
        "run_dir": str(run_dir),
        "status": first_present(config.get("status"), default="unknown"),
    }

    for col in METRIC_COLUMNS:
        row[col] = math.nan
    for key, value in metrics.items():
        if key in row:
            row[key] = value

    rmse_rad = as_float(row.get("rmse_q0_rad"))
    row["rmse_q0_deg"] = rmse_rad * 180.0 / math.pi if math.isfinite(rmse_rad) else math.nan
    row["mse_q0_rad"] = rmse_rad**2 if math.isfinite(rmse_rad) else math.nan
    row["mse_q0_deg"] = row["rmse_q0_deg"] ** 2 if math.isfinite(as_float(row["rmse_q0_deg"])) else math.nan

    safe_ndtw_rad = as_float(row.get("safe_ndtw_link_rad_mean"))
    unsafe_ndtw_rad = as_float(row.get("unsafe_ndtw_link_rad_mean"))
    row["safe_ndtw_link_deg_mean"] = (
        safe_ndtw_rad * 180.0 / math.pi if math.isfinite(safe_ndtw_rad) else math.nan
    )
    row["unsafe_ndtw_link_deg_mean"] = (
        unsafe_ndtw_rad * 180.0 / math.pi if math.isfinite(unsafe_ndtw_rad) else math.nan
    )

    return row


def finite_array(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([as_float(row.get(key)) for row in rows], dtype=np.float64)


def minmax_error(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    vmin = float(np.min(values[finite]))
    vmax = float(np.max(values[finite]))
    if math.isclose(vmin, vmax):
        out[finite] = 0.0
    else:
        out[finite] = (values[finite] - vmin) / (vmax - vmin)
    return out


def minmax_good(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    vmin = float(np.min(values[finite]))
    vmax = float(np.max(values[finite]))
    if math.isclose(vmin, vmax):
        out[finite] = 1.0
    else:
        out[finite] = (values[finite] - vmin) / (vmax - vmin)
    return out


def weighted_sum_or_nan(components: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if not components:
        return np.asarray([], dtype=np.float64)
    out = np.zeros_like(components[0], dtype=np.float64)
    valid = np.ones_like(components[0], dtype=bool)
    for comp, weight in zip(components, weights):
        valid &= np.isfinite(comp)
        out += float(weight) * comp
    out[~valid] = np.nan
    return out


def compute_scores(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    if args.score_normalization != "minmax":
        raise ValueError(f"Unsupported score normalization: {args.score_normalization}")

    sw = np.asarray(args.safety_weights, dtype=np.float64)
    iw = np.asarray(args.imitation_weights, dtype=np.float64)
    sw = sw / np.sum(sw)
    iw = iw / np.sum(iw)

    clear_good = minmax_good(finite_array(rows, "merge_safe_M_clear_m"))
    ctr_good = 1.0 - minmax_error(finite_array(rows, "merge_safe_M_ctr"))
    cc_good = 1.0 - minmax_error(finite_array(rows, "merge_safe_M_cc"))
    s_safe = weighted_sum_or_nan([clear_good, ctr_good, cc_good], sw)

    rmse_error = minmax_error(finite_array(rows, args.score_rmse_metric))
    ndtw_error = minmax_error(finite_array(rows, args.score_ndtw_metric))
    imitation_penalty = weighted_sum_or_nan([rmse_error, ndtw_error], iw)
    s_imit = 1.0 - imitation_penalty
    s_imit[~np.isfinite(imitation_penalty)] = np.nan

    for idx, row in enumerate(rows):
        row["S_safe"] = float(s_safe[idx]) if np.isfinite(s_safe[idx]) else math.nan
        row["S_imit"] = float(s_imit[idx]) if np.isfinite(s_imit[idx]) else math.nan


def stats_for(values: Sequence[Any]) -> Dict[str, float]:
    arr = np.asarray([as_float(v) for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "q25": math.nan, "q75": math.nan}
    std = 0.0 if finite.size == 1 else float(np.std(finite, ddof=1))
    return {
        "mean": float(np.mean(finite)),
        "std": std,
        "median": float(np.median(finite)),
        "q25": float(np.percentile(finite, 25)),
        "q75": float(np.percentile(finite, 75)),
    }


def group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("param_key"),
        round(as_float(row.get("phi_rr")), 9),
        round(as_float(row.get("phi_hr")), 9),
        round(as_float(row.get("gamma_rr")), 9),
        round(as_float(row.get("gamma_hr")), 9),
    )


def aggregate_rows(rows: List[Dict[str, Any]], numeric_cols: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    agg_rows: List[Dict[str, Any]] = []
    for _, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        first = items[0]
        out: Dict[str, Any] = {
            "platform": first.get("platform", ""),
            "sweep_type": first.get("sweep_type", ""),
            "scenario": first.get("scenario", ""),
            "param_key": first.get("param_key", ""),
            "phi_rr": first.get("phi_rr", math.nan),
            "phi_hr": first.get("phi_hr", math.nan),
            "gamma_rr": first.get("gamma_rr", math.nan),
            "gamma_hr": first.get("gamma_hr", math.nan),
            "n_runs": len(items),
        }
        for col in numeric_cols:
            stat = stats_for([row.get(col) for row in items])
            for suffix, value in stat.items():
                out[f"{col}_{suffix}"] = value
        agg_rows.append(out)

    return sorted(
        agg_rows,
        key=lambda row: (
            as_float(row.get("phi_rr")),
            as_float(row.get("phi_hr")),
            as_float(row.get("gamma_rr")),
            as_float(row.get("gamma_hr")),
        ),
    )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.out_csv is None:
        return (
            args.sweep_root / "sweep_summary_per_run.csv",
            args.sweep_root / "sweep_summary_agg.csv",
        )

    out = Path(args.out_csv).expanduser()
    if out.suffix == ".csv":
        stem = out.stem
        if stem.endswith("_per_run"):
            agg_name = stem[: -len("_per_run")] + "_agg.csv"
        else:
            agg_name = stem + "_agg.csv"
        return out, out.with_name(agg_name)

    return out / "sweep_summary_per_run.csv", out / "sweep_summary_agg.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate BDCC sweep results.")
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument(
        "--mode",
        choices=["both", "merge", "self_collision", "human_robot"],
        default="both",
        help="Metadata/filter label; missing metric computation still uses --mode both.",
    )
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--compute-missing", action="store_true")
    parser.add_argument("--urdf-path", default=DEFAULT_URDF_PATH)
    parser.add_argument("--eval-start-sec", type=float, default=10.0)
    parser.add_argument("--eval-end-sec", type=float, default=62.0)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--max-lag-sec", type=float, default=2.0)
    parser.add_argument("--lag-step-sec", type=float, default=0.02)
    parser.add_argument("--no-compute-scores", action="store_true")
    parser.add_argument("--safety-weights", type=float, nargs=3, default=[0.45, 0.45, 0.10])
    parser.add_argument("--imitation-weights", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--score-normalization", choices=["minmax"], default="minmax")
    parser.add_argument("--score-rmse-metric", default="rmse_q0_deg")
    parser.add_argument("--score-ndtw-metric", default="safe_ndtw_link_deg_mean")
    parser.add_argument(
        "--offline-script",
        type=Path,
        default=default_script_path("scripts/offline/offline_compute_metrics.py"),
        help="Path to offline_compute_metrics.py.",
    )
    args = parser.parse_args()
    args.sweep_root = args.sweep_root.expanduser()
    args.offline_script = args.offline_script.expanduser()
    return args


def main() -> int:
    args = parse_args()
    run_dirs = find_run_dirs(args.sweep_root)
    print(f"[aggregate] sweep_root: {args.sweep_root}")
    print(f"[aggregate] run dirs: {len(run_dirs)}")

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        row = load_run_row(args, run_dir)
        if row is not None:
            rows.append(row)

    if not args.no_compute_scores:
        compute_scores(args, rows)

    score_cols = [] if args.no_compute_scores else SCORE_COLUMNS
    per_run_columns = CORE_COLUMNS + METRIC_COLUMNS + score_cols
    agg_numeric_cols = METRIC_COLUMNS + score_cols
    agg_rows = aggregate_rows(rows, agg_numeric_cols)

    per_run_csv, agg_csv = output_paths(args)
    write_csv(per_run_csv, rows, per_run_columns)

    agg_columns = [
        "platform",
        "sweep_type",
        "scenario",
        "param_key",
        "phi_rr",
        "phi_hr",
        "gamma_rr",
        "gamma_hr",
        "n_runs",
    ]
    for col in agg_numeric_cols:
        agg_columns.extend(
            [f"{col}_mean", f"{col}_std", f"{col}_median", f"{col}_q25", f"{col}_q75"]
        )
    write_csv(agg_csv, agg_rows, agg_columns)

    print(f"[aggregate] wrote: {per_run_csv}")
    print(f"[aggregate] wrote: {agg_csv}")
    print(f"[aggregate] rows: per_run={len(rows)}, agg={len(agg_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
