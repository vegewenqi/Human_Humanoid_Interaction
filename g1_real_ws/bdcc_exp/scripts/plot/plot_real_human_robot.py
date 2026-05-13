#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


JOINT_LABELS = {
    "waist_roll_joint": "Waist roll",
    "waist_pitch_joint": "Waist pitch",
    "left_shoulder_pitch_joint": "Left shoulder pitch",
    "left_shoulder_roll_joint": "Left shoulder roll",
    "left_elbow_joint": "Left elbow pitch",
    "right_shoulder_pitch_joint": "Right shoulder pitch",
    "right_shoulder_roll_joint": "Right shoulder roll",
    "right_elbow_joint": "Right elbow pitch",
}


DEFAULT_JOINTS = [
    "waist_roll_joint",
    # "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
]


def moving_average(y, window):
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(yp, kernel, mode="valid")


def load_summary(run_dir: Path):
    path = run_dir / "metrics_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_ndtw_mean(summary, prefix):
    """
    Prefer *_ndtw_link_rad_mean if available.
    Otherwise convert total 4-link nDTW to per-link mean by dividing by 4.
    """
    key_mean = f"{prefix}_ndtw_link_rad_mean"
    key_total = f"{prefix}_ndtw_link_rad"

    if key_mean in summary:
        return float(summary[key_mean])
    if key_total in summary:
        return float(summary[key_total]) / 4.0
    return float("nan")


def fmt_angle(rad_value, unit):
    if not np.isfinite(rad_value):
        return "nan"

    if unit == "deg":
        return f"{np.rad2deg(rad_value):.1f}°"
    return f"{rad_value:.3f} rad"


def angle_series(y, unit):
    y = np.asarray(y, dtype=float)
    if unit == "deg":
        return np.rad2deg(y)
    return y


def angle_label(unit):
    return "deg" if unit == "deg" else "rad"


def format_summary_text(summary, angle_unit="deg"):
    if not summary:
        return ""

    unsafe_ndtw_mean = get_ndtw_mean(summary, "unsafe")
    safe_ndtw_mean = get_ndtw_mean(summary, "safe")

    lines = [
        r"$M_{\mathrm{clear}}$: "
        f"{summary.get('hr_unsafe_M_clear_m', float('nan')):.3f} m → "
        f"{summary.get('hr_safe_M_clear_m', float('nan')):.3f} m",
        r"$M_{\mathrm{ctr}}$: "
        f"{100 * summary.get('hr_unsafe_M_ctr', float('nan')):.1f}% → "
        f"{100 * summary.get('hr_safe_M_ctr', float('nan')):.1f}%",
        r"$M_{\mathrm{cc}}$: "
        f"{summary.get('hr_unsafe_M_cc', 'nan')} → "
        f"{summary.get('hr_safe_M_cc', 'nan')}",
        r"RMSE$_q^0$: "
        f"{fmt_angle(summary.get('rmse_q0_rad', float('nan')), angle_unit)}",
        r"nDTW$_{\mathrm{link}}$: "
        f"{fmt_angle(unsafe_ndtw_mean, angle_unit)} → "
        f"{fmt_angle(safe_ndtw_mean, angle_unit)}",
    ]
    return "\n".join(lines)


def plot_main(
    run_dir: Path,
    outdir: Path,
    joints,
    t_start,
    t_end,
    smooth_window,
    formats,
    angle_unit,
):
    csv_path = run_dir / "metrics_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    summary = load_summary(run_dir)

    if t_start is not None:
        df = df[df["t_sec"] >= t_start].copy()
    if t_end is not None:
        df = df[df["t_sec"] <= t_end].copy()

    if df.empty:
        raise ValueError("No data left after t_start/t_end clipping.")

    df["t_plot"] = df["t_sec"] - df["t_sec"].iloc[0]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.05]},
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # Panel 1: representative joint targets
    # Same joint: same color. Nominal: dashed. CBF: solid.
    # ------------------------------------------------------------------
    ax = axes[0]

    for j in joints:
        q_nom_col = f"q_nom_{j}"
        q_cbf_col = f"q_cbf_{j}"

        if q_nom_col not in df.columns or q_cbf_col not in df.columns:
            print(f"[WARN] missing columns for joint {j}")
            continue

        label = JOINT_LABELS.get(j, j)
        color = next(ax._get_lines.prop_cycler)["color"]

        y_nom = angle_series(
            moving_average(df[q_nom_col].to_numpy(), smooth_window),
            angle_unit,
        )
        y_cbf = angle_series(
            moving_average(df[q_cbf_col].to_numpy(), smooth_window),
            angle_unit,
        )

        ax.plot(
            df["t_plot"],
            y_nom,
            linestyle="--",
            linewidth=1.35,
            color=color,
            label=f"Nominal {label}",
        )
        ax.plot(
            df["t_plot"],
            y_cbf,
            linestyle="-",
            linewidth=1.65,
            color=color,
            label=f"CBF {label}",
        )

    ax.set_ylabel(f"Joint target ({angle_label(angle_unit)})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.8, ncol=2, framealpha=0.78, loc="lower right")
    ax.margins(x=0)

    # ------------------------------------------------------------------
    # Panel 2: global minimum human-robot collision clearance
    # ------------------------------------------------------------------
    ax = axes[1]
    unsafe_col = "hr_unsafe_global_min_clearance"
    safe_col = "hr_safe_global_min_clearance"

    if unsafe_col not in df.columns or safe_col not in df.columns:
        raise KeyError(
            f"Missing required clearance columns: {unsafe_col}, {safe_col}. "
            "Did you run offline_compute_metrics.py with --mode human_robot or --mode both?"
        )

    y_unsafe = moving_average(df[unsafe_col].to_numpy(), smooth_window)
    y_safe = moving_average(df[safe_col].to_numpy(), smooth_window)

    ax.plot(
        df["t_plot"],
        y_unsafe,
        linestyle="--",
        linewidth=1.55,
        label="Nominal global min",
    )
    ax.plot(
        df["t_plot"],
        y_safe,
        linestyle="-",
        linewidth=1.9,
        label="CBF global min",
    )
    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1.25,
        label="Contact boundary",
    )

    ax.fill_between(
        df["t_plot"],
        y_unsafe,
        0.0,
        where=(y_unsafe < 0.0),
        alpha=0.13,
        interpolate=True,
    )
    ax.fill_between(
        df["t_plot"],
        y_safe,
        0.0,
        where=(y_safe < 0.0),
        alpha=0.13,
        interpolate=True,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Collision clearance margin (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.2, ncol=3, framealpha=0.78, loc="upper right")
    ax.margins(x=0)

    text = format_summary_text(summary, angle_unit=angle_unit)
    if text:
        ax.text(
            0.985,
            0.035,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.16),
        )

    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = outdir / f"fig_real_human_robot_main.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.tight_layout()
    plt.close(fig)


def get_top_pair_columns(df, prefix, top_k=4):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return []

    scores = []
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if np.all(~np.isfinite(vals)):
            continue
        scores.append((np.nanmin(vals), c))

    scores.sort(key=lambda x: x[0])
    return [c for _, c in scores[:top_k]]


def pretty_pair_name(col):
    s = col
    s = s.replace("hr_unsafe_clearance__", "")
    s = s.replace("hr_safe_clearance__", "")
    s = s.replace("__", " vs ")
    s = s.replace("_", " ")
    return s


def plot_pairs(run_dir: Path, outdir: Path, t_start, t_end, smooth_window, top_k, formats):
    csv_path = run_dir / "metrics_timeseries.csv"
    df = pd.read_csv(csv_path)

    if t_start is not None:
        df = df[df["t_sec"] >= t_start].copy()
    if t_end is not None:
        df = df[df["t_sec"] <= t_end].copy()

    if df.empty:
        raise ValueError("No data left after t_start/t_end clipping.")

    df["t_plot"] = df["t_sec"] - df["t_sec"].iloc[0]

    unsafe_cols = get_top_pair_columns(df, "hr_unsafe_clearance__", top_k=top_k)
    safe_cols = [c.replace("hr_unsafe_clearance__", "hr_safe_clearance__") for c in unsafe_cols]
    safe_cols = [c for c in safe_cols if c in df.columns]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.0),
        sharex=True,
        constrained_layout=True,
    )

    ax = axes[0]
    for c in unsafe_cols:
        y = moving_average(df[c].to_numpy(), smooth_window)
        ax.plot(df["t_plot"], y, linewidth=1.2, label=pretty_pair_name(c))
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.set_ylabel("Nominal clearance (m)")
    ax.set_title("Most critical human-robot monitored pairs")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.78)
    ax.margins(x=0)

    ax = axes[1]
    for c in safe_cols:
        y = moving_average(df[c].to_numpy(), smooth_window)
        ax.plot(df["t_plot"], y, linewidth=1.2, label=pretty_pair_name(c))
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CBF clearance (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.78)
    ax.margins(x=0)

    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = outdir / f"fig_real_human_robot_pairs.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.tight_layout()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing metrics_timeseries.csv and metrics_summary.json.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: run_dir/figures",
    )
    parser.add_argument(
        "--joints",
        nargs="*",
        default=DEFAULT_JOINTS,
        help="Representative joints to plot.",
    )
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--top-k-pairs", type=int, default=4)
    parser.add_argument(
        "--angle-unit",
        choices=["deg", "rad"],
        default="deg",
        help="Angle unit used for joint targets and nDTW-link in the figure.",
    )
    parser.add_argument(
        "--no-pair-figure",
        action="store_true",
        help="Only generate the main figure.",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        default=["svg", "png"],
        choices=["svg", "png", "pdf"],
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else run_dir / "figures"

    plot_main(
        run_dir=run_dir,
        outdir=outdir,
        joints=args.joints,
        t_start=args.t_start,
        t_end=args.t_end,
        smooth_window=args.smooth_window,
        formats=args.formats,
        angle_unit=args.angle_unit,
    )

    if not args.no_pair_figure:
        plot_pairs(
            run_dir=run_dir,
            outdir=outdir,
            t_start=args.t_start,
            t_end=args.t_end,
            smooth_window=args.smooth_window,
            top_k=args.top_k_pairs,
            formats=args.formats,
        )


if __name__ == "__main__":
    main()
