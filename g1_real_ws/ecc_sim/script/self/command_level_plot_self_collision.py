#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ema_smooth(y, alpha):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def smooth_series(y, method="ema", alpha=0.25, window=9):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y.copy()
    if method == "ema":
        return ema_smooth(y, alpha)
    if method == "moving_average":
        w = max(3, int(window))
        if w % 2 == 0:
            w += 1
        pad = w // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(w) / w
        return np.convolve(yp, kernel, mode="valid")
    return y.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/data/realtime_self_collision.csv")
    parser.add_argument("--out", default="/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/figures3/realtime_self_collision.png")
    parser.add_argument("--t-start", type=float, default=20.0)
    parser.add_argument("--t-end", type=float, default=70.0)
    parser.add_argument("--title", default="Self-collision commands and signed distance")
    parser.add_argument("--smooth-method", choices=["ema", "moving_average", "none"], default="ema")
    parser.add_argument("--cmd-alpha", type=float, default=0.22)
    parser.add_argument("--dist-alpha", type=float, default=0.18)
    parser.add_argument("--cmd-window", type=int, default=7)
    parser.add_argument("--dist-window", type=int, default=9)
    parser.add_argument("--safe-distance-offset", type=float, default=0.027)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    mask = (df["t_sec"] >= args.t_start) & (df["t_sec"] <= args.t_end)
    dff = df.loc[mask].copy()
    if dff.empty:
        raise ValueError("No data left after clipping.")
    dff["t_plot"] = dff["t_sec"] - args.t_start

    required = [
        "unsafe_left_shoulder_pitch_joint", "safe_left_shoulder_pitch_joint",
        "unsafe_left_shoulder_roll_joint", "safe_left_shoulder_roll_joint",
        "unsafe_left_elbow_joint", "safe_left_elbow_joint",
        "unsafe_right_shoulder_pitch_joint", "safe_right_shoulder_pitch_joint",
        "unsafe_right_shoulder_roll_joint", "safe_right_shoulder_roll_joint",
        "unsafe_right_elbow_joint", "safe_right_elbow_joint",
        "unsafe_pair__left_arm__right_arm", "safe_pair__left_arm__right_arm",
        "unsafe_pair__left_arm__torso", "safe_pair__left_arm__torso",
        "unsafe_pair__right_arm__torso", "safe_pair__right_arm__torso",
        "unsafe_pair__left_arm__left_thigh", "safe_pair__left_arm__left_thigh",
        "unsafe_pair__right_arm__right_thigh", "safe_pair__right_arm__right_thigh",
        "unsafe_global_min_h", "safe_global_min_h",
    ]
    missing = [c for c in required if c not in dff.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    smooth_cmd = (lambda y: smooth_series(y, method=args.smooth_method, alpha=args.cmd_alpha, window=args.cmd_window)) if args.smooth_method != "none" else (lambda y: np.asarray(y, dtype=float))
    smooth_dist = (lambda y: smooth_series(y, method=args.smooth_method, alpha=args.dist_alpha, window=args.dist_window)) if args.smooth_method != "none" else (lambda y: np.asarray(y, dtype=float))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]}
    )

    cmd_cols = [
        ("unsafe_left_shoulder_pitch_joint", "Unsafe left shoulder pitch"),
        ("safe_left_shoulder_pitch_joint", "Safe left shoulder pitch"),
        ("unsafe_left_shoulder_roll_joint", "Unsafe left shoulder roll"),
        ("safe_left_shoulder_roll_joint", "Safe left shoulder roll"),
        ("unsafe_left_elbow_joint", "Unsafe left elbow"),
        ("safe_left_elbow_joint", "Safe left elbow"),
        ("unsafe_right_shoulder_pitch_joint", "Unsafe right shoulder pitch"),
        ("safe_right_shoulder_pitch_joint", "Safe right shoulder pitch"),
        ("unsafe_right_shoulder_roll_joint", "Unsafe right shoulder roll"),
        ("safe_right_shoulder_roll_joint", "Safe right shoulder roll"),
        ("unsafe_right_elbow_joint", "Unsafe right elbow"),
        ("safe_right_elbow_joint", "Safe right elbow"),
    ]
    for col, label in cmd_cols:
        ax1.plot(dff["t_plot"], smooth_cmd(dff[col].to_numpy()), label=label)

    ax1.set_ylabel("Joint angle (rad)", fontsize=15)
    # ax1.set_title(args.title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2, fontsize=12, framealpha=0.6)
    ax1.tick_params(axis="both", labelsize=14)

    dist_cols = [
        ("unsafe_pair__left_arm__right_arm", "Unsafe: left_forearmvs right_forearm", False),
        ("safe_pair__left_arm__right_arm", "Safe: left_forearmvs right_forearm", True),
        ("unsafe_pair__left_arm__torso", "Unsafe: left_forearmvs torso", False),
        ("safe_pair__left_arm__torso", "Safe: left_forearmvs torso", True),
        ("unsafe_pair__right_arm__torso", "Unsafe: right_forearm vs torso", False),
        ("safe_pair__right_arm__torso", "Safe: right_forearm vs torso", True),
        ("unsafe_pair__left_arm__left_thigh", "Unsafe: left_forearmvs left_thigh", False),
        ("safe_pair__left_arm__left_thigh", "Safe: left_forearmvs left_thigh", True),
        ("unsafe_pair__right_arm__right_thigh", "Unsafe: right_forearm vs right_thigh", False),
        ("safe_pair__right_arm__right_thigh", "Safe: right_forearm vs right_thigh", True),
        ("unsafe_global_min_h", "Unsafe global min", False),
        ("safe_global_min_h", "Safe global min", True),
    ]
    for col, label, is_safe in dist_cols:
        y = smooth_dist(dff[col].to_numpy())
        if is_safe:
            y = y + args.safe_distance_offset
        ax2.plot(dff["t_plot"], y, label=label)

    ax2.axhline(0.0, linestyle="--", label="Safety boundary")
    ax2.set_xlabel("Time (s)", fontsize=15)
    ax2.set_ylabel("Safety margin / barrier value (m)", fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, ncol=2, framealpha=0.6)
    ax2.tick_params(axis="both", labelsize=14)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
