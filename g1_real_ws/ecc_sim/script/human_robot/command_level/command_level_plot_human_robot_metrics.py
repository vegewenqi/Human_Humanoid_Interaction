# #!/usr/bin/env python3
# import argparse
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def ema_smooth(y, alpha):
#     y = np.asarray(y, dtype=float)
#     out = np.empty_like(y)
#     out[0] = y[0]
#     for i in range(1, len(y)):
#         out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
#     return out


# def smooth_series(y, method="ema", alpha=0.25, window=9):
#     y = np.asarray(y, dtype=float)
#     if len(y) < 3:
#         return y.copy()
#     if method == "ema":
#         return ema_smooth(y, alpha)
#     if method == "moving_average":
#         w = max(3, int(window))
#         if w % 2 == 0:
#             w += 1
#         pad = w // 2
#         yp = np.pad(y, (pad, pad), mode="edge")
#         kernel = np.ones(w) / w
#         return np.convolve(yp, kernel, mode="valid")
#     return y.copy()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", default='/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/result2/command_level_human_robot_metrics.csv')
#     parser.add_argument("--out", default='/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/figures2/command_level_human_robot_metrics1.png')
#     parser.add_argument("--t-start", type=float, default=10.0)
#     parser.add_argument("--t-end", type=float, default=50.0)
#     parser.add_argument("--title", default="Commands and signed distance")
#     parser.add_argument("--smooth-method", choices=["ema", "moving_average", "none"], default="ema")
#     parser.add_argument("--cmd-alpha", type=float, default=0.22)
#     parser.add_argument("--dist-alpha", type=float, default=0.18)
#     parser.add_argument("--cmd-window", type=int, default=7)
#     parser.add_argument("--dist-window", type=int, default=9)
#     parser.add_argument(
#         "--safe-distance-offset", type=float, default=0.00,
#         help="Vertical offset applied only to the safe/CBF distance curves."
#     )
#     args = parser.parse_args()

#     df = pd.read_csv(args.csv)

#     required = [
#         "t_sec",
#         "unsafe_waist_roll_joint",
#         "unsafe_left_shoulder_roll_joint",
#         "unsafe_left_elbow_joint",
#         "safe_waist_roll_joint",
#         "safe_left_shoulder_roll_joint",
#         "safe_left_elbow_joint",
#         "unsafe_rep1_h",
#         "unsafe_rep2_h",
#         "unsafe_global_min_h",
#         "safe_rep1_h",
#         "safe_rep2_h",
#         "safe_global_min_h",
#     ]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise KeyError(f"Missing required columns: {missing}")

#     mask = (df["t_sec"] >= args.t_start) & (df["t_sec"] <= args.t_end)
#     dff = df.loc[mask].copy()
#     if dff.empty:
#         raise ValueError("No data left after clipping. Check --t-start and --t-end.")
#     dff["t_plot"] = dff["t_sec"] - args.t_start

#     smooth_cmd = (lambda y: smooth_series(
#         y, method=args.smooth_method, alpha=args.cmd_alpha, window=args.cmd_window
#     )) if args.smooth_method != "none" else (lambda y: np.asarray(y, dtype=float))

#     smooth_dist = (lambda y: smooth_series(
#         y, method=args.smooth_method, alpha=args.dist_alpha, window=args.dist_window
#     )) if args.smooth_method != "none" else (lambda y: np.asarray(y, dtype=float))

#     fig, (ax1, ax2) = plt.subplots(
#         2, 1, figsize=(11, 8), sharex=True,
#         gridspec_kw={"height_ratios": [1, 1.15]}
#     )

#     cmd_cols = [
#         ("unsafe_waist_roll_joint", "Unsafe torso roll"),
#         ("safe_waist_roll_joint", "Safe torso roll"),
#         ("unsafe_left_shoulder_roll_joint", "Unsafe left shoulder roll"),
#         ("safe_left_shoulder_roll_joint", "Safe left shoulder roll"),
#         ("unsafe_left_elbow_joint", "Unsafe left elbow"),
#         ("safe_left_elbow_joint", "Safe left elbow"),
#     ]
#     for col, label in cmd_cols:
#         ax1.plot(dff["t_plot"], smooth_cmd(dff[col].to_numpy()), label=label)

#     ax1.set_ylabel("Joint angle (rad)")
#     ax1.set_title(args.title)
#     ax1.grid(True, alpha=0.3)
#     ax1.legend(ncol=2, fontsize=9)

#     dist_specs = [
#         ("unsafe_rep1_h", "Unsafe: left_arm vs right_forearm_hand", False),
#         ("safe_rep1_h", "Safe: left_arm vs right_forearm_hand", True),
#         ("unsafe_rep2_h", "Unsafe: left_upper_arm vs right_forearm_hand", False),
#         ("safe_rep2_h", "Safe: left_upper_arm vs right_forearm_hand", True),
#         ("unsafe_global_min_h", "Unsafe global min", False),
#         ("safe_global_min_h", "Safe global min", True),
#     ]

#     for col, label, is_safe in dist_specs:
#         y = smooth_dist(dff[col].to_numpy())
#         if is_safe:
#             y = y + args.safe_distance_offset
#         ax2.plot(dff["t_plot"], y, label=label)

#     ax2.axhline(0.0, linestyle="--", label="Safety boundary")
#     ax2.set_xlabel("Time (s)")
#     ax2.set_ylabel("Signed distance / barrier value (m)")
#     ax2.grid(True, alpha=0.3)
#     ax2.legend(fontsize=9, ncol=2)

#     out = Path(args.out)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     fig.tight_layout()
#     fig.savefig(out, dpi=300, bbox_inches="tight")
#     print(f"Saved figure to: {out}")
#     print(f"Applied smoothing method: {args.smooth_method}")
#     print(f"Applied safe/CBF-only distance offset: {args.safe_distance_offset} m")


# if __name__ == "__main__":
#     main()




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
    parser.add_argument(
        "--csv",
        default="/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/result2/command_level_human_robot_metrics.csv"
    )
    parser.add_argument(
        "--out",
        default="/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/figures2/command_level_human_robot_metrics1.svg"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="svg",
        help="Output format. If not set, inferred from --out suffix."
    )
    parser.add_argument("--t-start", type=float, default=10.0)
    parser.add_argument("--t-end", type=float, default=50.0)
    parser.add_argument("--title", default="Commands and signed distance")
    parser.add_argument("--smooth-method", choices=["ema", "moving_average", "none"], default="ema")
    parser.add_argument("--cmd-alpha", type=float, default=0.22)
    parser.add_argument("--dist-alpha", type=float, default=0.18)
    parser.add_argument("--cmd-window", type=int, default=7)
    parser.add_argument("--dist-window", type=int, default=9)
    parser.add_argument(
        "--safe-distance-offset", type=float, default=0.00,
        help="Vertical offset applied only to the safe/CBF distance curves."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = [
        "t_sec",
        "unsafe_waist_roll_joint",
        "unsafe_left_shoulder_roll_joint",
        "unsafe_left_elbow_joint",
        "safe_waist_roll_joint",
        "safe_left_shoulder_roll_joint",
        "safe_left_elbow_joint",
        "unsafe_rep1_h",
        "unsafe_rep2_h",
        "unsafe_global_min_h",
        "safe_rep1_h",
        "safe_rep2_h",
        "safe_global_min_h",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    mask = (df["t_sec"] >= args.t_start) & (df["t_sec"] <= args.t_end)
    dff = df.loc[mask].copy()
    if dff.empty:
        raise ValueError("No data left after clipping. Check --t-start and --t-end.")
    dff["t_plot"] = dff["t_sec"] - args.t_start

    smooth_cmd = (lambda y: smooth_series(
        y, method=args.smooth_method, alpha=args.cmd_alpha, window=args.cmd_window
    )) if args.smooth_method != "none" else (lambda y: np.asarray(y, dtype=float))

    smooth_dist = (lambda y: smooth_series(
        y, method=args.smooth_method, alpha=args.dist_alpha, window=args.dist_window
    )) if args.smooth_method != "none" else (lambda y: np.asarray(y, dtype=float))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]}
    )

    cmd_cols = [
        ("unsafe_waist_roll_joint", "Unsafe torso roll"),
        ("safe_waist_roll_joint", "Safe torso roll"),
        ("unsafe_left_shoulder_roll_joint", "Unsafe left shoulder roll"),
        ("safe_left_shoulder_roll_joint", "Safe left shoulder roll"),
        ("unsafe_left_elbow_joint", "Unsafe left elbow"),
        ("safe_left_elbow_joint", "Safe left elbow"),
    ]
    for col, label in cmd_cols:
        ax1.plot(dff["t_plot"], smooth_cmd(dff[col].to_numpy()), label=label)

    ax1.set_ylabel("Joint angle (rad)", fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2, fontsize=12)
    ax1.tick_params(axis="both", labelsize=14)

    dist_specs = [
        ("unsafe_rep1_h", "Unsafe: left_forearm vs right_forearm", False),
        ("safe_rep1_h", "Safe: left_forearm vs right_forearm", True),
        ("unsafe_rep2_h", "Unsafe: left_upper_arm vs right_forearm", False),
        ("safe_rep2_h", "Safe: left_upper_arm vs right_forearm", True),
        ("unsafe_global_min_h", "Unsafe global min", False),
        ("safe_global_min_h", "Safe global min", True),
    ]

    for col, label, is_safe in dist_specs:
        y = smooth_dist(dff[col].to_numpy())
        if is_safe:
            y = y + args.safe_distance_offset
        ax2.plot(dff["t_plot"], y, label=label)

    ax2.axhline(0.0, linestyle="--", label="Safety boundary")
    ax2.set_xlabel("Time (s)", fontsize=15)
    ax2.set_ylabel("Safety margin / barrier value (m)", fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, ncol=2,loc="upper right")
    ax2.tick_params(axis="both", labelsize=14)

    out = Path(args.out)

    if out.suffix == "":
        fmt = args.format if args.format is not None else "pdf"
        out = out.with_suffix(f".{fmt}")
    else:
        fmt = args.format if args.format is not None else out.suffix[1:].lower()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()

    if fmt == "svg":
        fig.savefig(out, format="svg", bbox_inches="tight")
    elif fmt == "pdf":
        fig.savefig(out, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(out, format="png", dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {out}")
    print(f"Output format: {fmt}")
    print(f"Applied smoothing method: {args.smooth_method}")
    print(f"Applied safe/CBF-only distance offset: {args.safe_distance_offset} m")


if __name__ == "__main__":
    main()
