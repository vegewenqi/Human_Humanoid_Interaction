#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DIAG_LAYOUT = {
    0: "n_constraints",
    1: "total_ms",
    2: "qp_ms",
    3: "dq_ref_norm",
    4: "dq_safe_norm",
    5: "correction_norm",
    6: "min_control_barrier_value",
    7: "qp_status_code",
    8: "min_control_pair_id",
}


def load_npz(path: Path):
    if path.is_dir():
        path = path / "topics.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(str(path), allow_pickle=True), path


def finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def stats(x) -> Dict[str, Any]:
    x = finite(x)
    if len(x) == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan, "min": math.nan}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


def normalize_pair_label(s: str) -> str:
    s = str(s)
    prefix = ""
    if s.startswith("rr:"):
        prefix = "self"
        s = s[3:]
    elif s.startswith("hr:"):
        prefix = "human-robot"
        s = s[3:]
    s = s.replace("__", " vs ").replace("_", " ")
    return f"{prefix}: {s}" if prefix else s


def filter_window(t, *arrays, start=None, end=None):
    t = np.asarray(t, dtype=float)
    mask = np.isfinite(t)
    if start is not None:
        mask &= t >= float(start)
    if end is not None:
        mask &= t <= float(end)
    out = [t[mask]]
    for a in arrays:
        out.append(np.asarray(a)[mask])
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Summarize CBF diagnostics from topics.npz.")
    p.add_argument("--input", required=True, help="topics.npz or run directory containing topics.npz")
    p.add_argument("--eval-start-sec", type=float, default=None)
    p.add_argument("--eval-end-sec", type=float, default=None)
    p.add_argument("--rr-safety-distance", type=float, default=None)
    p.add_argument("--hr-safety-distance", type=float, default=None)
    p.add_argument("--rr-gamma", type=float, default=None)
    p.add_argument("--hr-gamma", type=float, default=None)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--ema-alpha", type=float, default=0.25)
    p.add_argument("--max-joint-velocity", type=float, default=0.7, help="rad/s")
    p.add_argument("--home-transition-velocity", type=float, default=0.20, help="rad/s")
    p.add_argument("--outdir", default=None)
    p.add_argument("--prefix", default="cbf_runtime_stats")
    p.add_argument("--top-k-pairs", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    data, input_path = load_npz(Path(args.input).expanduser())
    outdir = Path(args.outdir).expanduser() if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    required = ["cbf_diagnostics_t", "cbf_diagnostics_data"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing required fields in topics.npz: {missing}")

    t = np.asarray(data["cbf_diagnostics_t"], dtype=float)
    diag = np.asarray(data["cbf_diagnostics_data"], dtype=float)
    if diag.ndim != 2 or diag.shape[1] < 9:
        raise ValueError(f"cbf_diagnostics_data should be Nx9, got {diag.shape}")

    t, diag = filter_window(t, diag, start=args.eval_start_sec, end=args.eval_end_sec)
    if len(t) == 0:
        raise ValueError("No CBF diagnostics samples in selected window.")

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    mean_rate = float((len(t) - 1) / duration) if duration > 0 and len(t) > 1 else math.nan
    median_rate = float(1.0 / np.median(dt)) if len(dt) else math.nan

    n_constraints = diag[:, 0]
    total_ms = diag[:, 1]
    qp_ms = diag[:, 2]
    correction_norm = diag[:, 5]
    min_h = diag[:, 6]
    qp_status = diag[:, 7]

    # Pair topic is optional. It has the string label, which is more useful than numeric pair id.
    pair_counts = []
    if "cbf_min_control_pair_t" in data.files and "cbf_min_control_pair_data" in data.files:
        pair_t = np.asarray(data["cbf_min_control_pair_t"], dtype=float)
        pair_s = np.asarray(data["cbf_min_control_pair_data"], dtype=object)
        pair_t, pair_s = filter_window(pair_t, pair_s, start=args.eval_start_sec, end=args.eval_end_sec)
        c = Counter([normalize_pair_label(x) for x in pair_s if str(x) != ""])
        total_pairs = sum(c.values())
        for label, count in c.most_common(args.top_k_pairs):
            pct = 100.0 * count / total_pairs if total_pairs else math.nan
            pair_counts.append({"pair": label, "count": int(count), "percent": float(pct)})

    success_mask = np.isfinite(qp_status) & (qp_status >= 0.5)
    qp_success_pct = 100.0 * float(np.mean(success_mask)) if len(qp_status) else math.nan

    result = {
        "input": str(input_path),
        "eval_start_sec": float(t[0]),
        "eval_end_sec": float(t[-1]),
        "n_samples": int(len(t)),
        "duration_sec": duration,
        "cbf_rate_hz_mean": mean_rate,
        "cbf_rate_hz_median": median_rate,
        "n_constraints": stats(n_constraints),
        "total_ms": stats(total_ms),
        "qp_ms": stats(qp_ms),
        "correction_norm_rad": stats(correction_norm),
        "min_control_barrier_value": stats(min_h),
        "qp_success_percent": qp_success_pct,
        "top_min_control_pairs": pair_counts,
        "settings": {
            "rr_safety_distance_m": args.rr_safety_distance,
            "hr_safety_distance_m": args.hr_safety_distance,
            "rr_gamma": args.rr_gamma,
            "hr_gamma": args.hr_gamma,
            "control_dt_sec": args.control_dt,
            "controller_rate_hz": 1.0 / args.control_dt if args.control_dt and args.control_dt > 0 else math.nan,
            "ema_alpha": args.ema_alpha,
            "max_joint_velocity_rad_s": args.max_joint_velocity,
            "max_joint_velocity_deg_s": float(np.rad2deg(args.max_joint_velocity)),
            "home_transition_velocity_rad_s": args.home_transition_velocity,
            "home_transition_velocity_deg_s": float(np.rad2deg(args.home_transition_velocity)),
        },
    }

    json_path = outdir / f"{args.prefix}.json"
    md_path = outdir / f"{args.prefix}.md"
    csv_path = outdir / f"{args.prefix}.csv"

    json_path.write_text(json.dumps(result, indent=2))

    rows = []
    def add(k, v): rows.append((k, v))
    add("Eval window", f"{result['eval_start_sec']:.2f}-{result['eval_end_sec']:.2f} s")
    if args.rr_safety_distance is not None: add("phi_s^rr", f"{args.rr_safety_distance:.3f} m")
    if args.hr_safety_distance is not None: add("phi_s^hr", f"{args.hr_safety_distance:.3f} m")
    if args.rr_gamma is not None: add("gamma^rr", f"{args.rr_gamma:.2f}")
    if args.hr_gamma is not None: add("gamma^hr", f"{args.hr_gamma:.2f}")
    add("CBF rate", f"{mean_rate:.1f} Hz mean / {median_rate:.1f} Hz median")
    add("QP solve time", f"{result['qp_ms']['median']:.2f} ms median / {result['qp_ms']['p95']:.2f} ms p95")
    add("Total CBF time", f"{result['total_ms']['median']:.2f} ms median / {result['total_ms']['p95']:.2f} ms p95")
    add("Active constraints", f"{result['n_constraints']['median']:.0f} median / {result['n_constraints']['max']:.0f} max")
    add("QP success", f"{qp_success_pct:.1f}%")
    add("Controller", f"{1.0/args.control_dt:.0f} Hz, EMA alpha={args.ema_alpha:.2f}")
    add("Velocity limit", f"{args.max_joint_velocity:.2f} rad/s ({np.rad2deg(args.max_joint_velocity):.1f} deg/s)")
    if pair_counts:
        add("Most critical pair", f"{pair_counts[0]['pair']} ({pair_counts[0]['percent']:.1f}%)")

    with csv_path.open("w") as f:
        f.write("item,value\n")
        for k, v in rows:
            f.write(f'"{k}","{v}"\n')

    lines = ["| Item | Value |", "|---|---:|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    md_path.write_text("\n".join(lines) + "\n")

    print("CBF runtime summary")
    print(f"Input: {input_path}")
    print(f"Window: {result['eval_start_sec']:.3f}-{result['eval_end_sec']:.3f} s, samples={len(t)}")
    print(f"CBF rate: {mean_rate:.2f} Hz mean / {median_rate:.2f} Hz median")
    print(f"QP time: {result['qp_ms']['median']:.2f} ms median / {result['qp_ms']['p95']:.2f} ms p95")
    print(f"Total CBF time: {result['total_ms']['median']:.2f} ms median / {result['total_ms']['p95']:.2f} ms p95")
    print(f"Active constraints: {result['n_constraints']['median']:.0f} median / {result['n_constraints']['max']:.0f} max")
    print(f"QP success: {qp_success_pct:.1f}%")
    if pair_counts:
        print("Top critical pairs:")
        for p in pair_counts:
            print(f"  {p['pair']}: {p['percent']:.1f}%")
    print("Wrote:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
