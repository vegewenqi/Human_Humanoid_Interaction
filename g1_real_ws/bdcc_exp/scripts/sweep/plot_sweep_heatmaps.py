#!/usr/bin/env python3
"""Plot BDCC2026 sweep heatmaps from sweep_summary_agg.csv.

Example:

python3 /ws/bdcc_exp/scripts/sweep/plot_sweep_heatmaps.py \
  --summary-csv /ws/bdcc_exp/sweeps/real_merge_phi_grid/sweep_summary_agg.csv \
  --sweep-type phi_grid \
  --outdir /ws/bdcc_exp/figures/sweeps/real_merge_phi_grid \
  --formats png svg pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np


METRICS = [
    {
        "metric": "merge_safe_M_clear_m",
        "title": "Minimum clearance [m]",
        "scale": 1.0,
        "fmt_small": "{:.3f}",
        "fmt_large": "{:.2f}",
        "norm": "auto",
    },
    {
        "metric": "merge_safe_M_ctr",
        "title": "Collision-time ratio [%]",
        "scale": 100.0,
        "fmt_small": "{:.2f}",
        "fmt_large": "{:.1f}",
        "norm": "zero_min",
    },
    {
        "metric": "rmse_q0_deg",
        "title": r"$\rm{RMSE}_q^0$ [deg]",
        "scale": 1.0,
        "fmt_small": "{:.2f}",
        "fmt_large": "{:.1f}",
        "norm": "auto",
    },
    {
        "metric": "safe_ndtw_link_deg_mean",
        "title": "nDTW-link [deg]",
        "scale": 1.0,
        "fmt_small": "{:.2f}",
        "fmt_large": "{:.1f}",
        "norm": "auto",
    },
]


def as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metric_column(base_metric: str, use_median: bool) -> str:
    return f"{base_metric}_{'median' if use_median else 'mean'}"


def get_panel_cmap(args: argparse.Namespace) -> colors.Colormap:
    if args.palette == "custom":
        return plt.get_cmap(args.cmap)
    if args.palette == "cividis":
        return plt.get_cmap("cividis")
    if args.palette == "soft_purple":
        return colors.LinearSegmentedColormap.from_list(
            "bdcc_soft_purple",
            ["#f7f4fb", "#eae4f0", "#dcd0ed", "#cfc0ef", "#c1aeea"],
        )
    return colors.LinearSegmentedColormap.from_list(
        "bdcc_soft_blue",
        ["#f8fbfd", "#eaf3f8", "#d8e9f2", "#c2dbe9", "#a7c9dc"],
    )


def sorted_unique(values: Sequence[float]) -> List[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return sorted(set(round(v, 12) for v in finite))


def pivot_matrix(
    rows: Sequence[Dict[str, str]],
    x_key: str,
    y_key: str,
    value_key: str,
) -> Tuple[List[float], List[float], np.ndarray]:
    xs = sorted_unique([as_float(row.get(x_key)) for row in rows])
    ys = sorted_unique([as_float(row.get(y_key)) for row in rows])
    matrix = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    x_index = {x: idx for idx, x in enumerate(xs)}
    y_index = {y: idx for idx, y in enumerate(ys)}

    for row in rows:
        x = round(as_float(row.get(x_key)), 12)
        y = round(as_float(row.get(y_key)), 12)
        value = as_float(row.get(value_key))
        if x in x_index and y in y_index:
            matrix[y_index[y], x_index[x]] = value
    return xs, ys, matrix


def format_x_tick(value: float, sweep_type: str) -> str:
    if sweep_type == "phi_grid":
        return f"{value:.2f}"
    return f"{value:.1f}"

def format_y_tick(value: float, sweep_type: str) -> str:
    if sweep_type == "phi_grid":
        return f"{value:.3f}"
    return f"{value:.1f}"

def annotate_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    norm: colors.Normalize,
    cmap: colors.Colormap,
    fontsize: float,
    fmt_small: str,
    fmt_large: str,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    span = float(np.nanmax(finite) - np.nanmin(finite))
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if not np.isfinite(value):
                continue
            if abs(value) >= 10 or span >= 10:
                text = fmt_large.format(value)
            else:
                text = fmt_small.format(value)
            normed = norm(value) if norm is not None else math.nan
            if isinstance(normed, np.ma.MaskedArray):
                normed = float(normed.filled(math.nan))
            if math.isfinite(float(normed)):
                rgba = cmap(float(np.clip(normed, 0.0, 1.0)))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "white" if luminance < 0.48 else "black"
            else:
                text_color = "black"
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def make_norm(matrix: np.ndarray, spec: Dict[str, Any], args: argparse.Namespace) -> colors.Normalize:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0)

    kind = spec.get("norm", "auto")
    if kind == "clearance":
        vmin = args.clearance_vmin if args.clearance_vmin is not None else float(np.nanmin(finite))
        vmax = args.clearance_vmax if args.clearance_vmax is not None else float(np.nanmax(finite))
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    if kind == "zero_min":
        vmax = float(np.nanmax(finite))
        if math.isclose(vmax, 0.0):
            vmax = 1.0
        return colors.Normalize(vmin=0.0, vmax=vmax)

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    return colors.Normalize(vmin=vmin, vmax=vmax)


def baseline_indices(xs: Sequence[float], ys: Sequence[float], sweep_type: str) -> Tuple[int, int]:
    if sweep_type == "phi_grid":
        bx, by = 0.15, 0.03
    else:
        bx, by = 3.0, 2.0
    x_idx = int(np.argmin(np.abs(np.asarray(xs, dtype=float) - bx)))
    y_idx = int(np.argmin(np.abs(np.asarray(ys, dtype=float) - by)))
    return x_idx, y_idx


def point_indices(xs: Sequence[float], ys: Sequence[float], x_value: float, y_value: float) -> Tuple[int, int]:
    x_idx = int(np.argmin(np.abs(np.asarray(xs, dtype=float) - x_value)))
    y_idx = int(np.argmin(np.abs(np.asarray(ys, dtype=float) - y_value)))
    return x_idx, y_idx


def scatter_marker(
    ax: plt.Axes,
    x_idx: int,
    y_idx: int,
    *,
    marker: str,
    size: float,
    linewidth: float,
    offset: Tuple[float, float],
) -> None:
    ax.scatter(
        [x_idx + offset[0]],
        [y_idx + offset[1]],
        marker=marker,
        s=size,
        c="none",
        edgecolors="black",
        linewidths=linewidth,
        zorder=4,
    )


def plot_heatmaps(args: argparse.Namespace, rows: List[Dict[str, str]]) -> None:
    if args.sweep_type == "phi_grid":
        x_key, y_key = "phi_hr", "phi_rr"
        x_label = r"$\phi_s^{hr}$ [m]"
        y_label = r"$\phi_s^{rr}$ [m]"
        out_name = "phi_grid_heatmaps"
    else:
        x_key, y_key = "gamma_hr", "gamma_rr"
        x_label = r"$\gamma^{hr}$"
        y_label = r"$\gamma^{rr}$"
        out_name = "gamma_grid_heatmaps"

    fig, axes = plt.subplots(2, 2, figsize=args.figsize, constrained_layout=True)
    axes_flat = list(axes.flat)
    panel_cmap = get_panel_cmap(args)

    for ax, spec in zip(axes_flat, METRICS):
        spec = dict(spec)
        if args.clearance_diverging and spec["metric"] == "merge_safe_M_clear_m":
            spec["norm"] = "clearance"
            cmap = plt.get_cmap(args.clearance_cmap)
        else:
            cmap = panel_cmap
        col = metric_column(spec["metric"], args.use_median)
        xs, ys, matrix = pivot_matrix(rows, x_key, y_key, col)
        matrix = matrix * float(spec["scale"])
        masked = np.ma.masked_invalid(matrix)
        norm = make_norm(matrix, spec, args)
        im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=args.colorbar_shrink)
        cbar.ax.tick_params(labelsize=args.tick_fontsize)

        ax.set_title(spec["title"], fontsize=args.title_fontsize)
        ax.set_xlabel(x_label, fontsize=args.axis_label_fontsize)
        ax.set_ylabel(y_label, fontsize=args.axis_label_fontsize)
        ax.set_xticks(np.arange(len(xs)))
        ax.set_yticks(np.arange(len(ys)))
        ax.set_xticklabels([format_x_tick(x, args.sweep_type) for x in xs], fontsize=args.tick_fontsize)
        ax.set_yticklabels([format_y_tick(y, args.sweep_type) for y in ys], fontsize=args.tick_fontsize)

        if args.annotate:
            annotate_cells(
                ax,
                matrix,
                norm=norm,
                cmap=cmap,
                fontsize=args.annotation_fontsize,
                fmt_small=spec["fmt_small"],
                fmt_large=spec["fmt_large"],
            )

        if xs and ys and args.mark_baseline:
            bx, by = baseline_indices(xs, ys, args.sweep_type)
            scatter_marker(
                ax,
                bx,
                by,
                marker=args.baseline_marker,
                size=args.baseline_marker_size,
                linewidth=args.baseline_marker_linewidth,
                offset=tuple(args.marker_offset),
            )

        if xs and ys and args.mark_candidate and args.sweep_type == "phi_grid":
            cx, cy = point_indices(xs, ys, args.candidate_phi_hr, args.candidate_phi_rr)
            scatter_marker(
                ax,
                cx,
                cy,
                marker=args.candidate_marker,
                size=args.candidate_marker_size,
                linewidth=args.candidate_marker_linewidth,
                offset=tuple(args.marker_offset),
            )

    if args.marker_legend:
        handles = []
        if args.mark_baseline:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=args.baseline_marker,
                    color="black",
                    markerfacecolor="none",
                    linestyle="None",
                    markersize=5.5,
                    markeredgewidth=args.baseline_marker_linewidth,
                    label="Hardware point",
                )
            )
        if args.mark_candidate and args.sweep_type == "phi_grid":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=args.candidate_marker,
                    color="black",
                    markerfacecolor="none",
                    linestyle="None",
                    markersize=6,
                    markeredgewidth=args.candidate_marker_linewidth,
                    label="Candidate point",
                )
            )
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=len(handles),
                frameon=False,
                fontsize=args.legend_fontsize,
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        path = args.outdir / f"{out_name}.{fmt}"
        fig.savefig(path, dpi=args.dpi)
        print(f"[heatmap] wrote: {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BDCC sweep heatmaps.")
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--sweep-type", choices=["phi_grid", "gamma_grid"], required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"])
    parser.add_argument("--metric-set", choices=["main"], default="main")
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-cell values. Use --no-annotate for a cleaner figure.",
    )
    parser.add_argument("--annotation-fontsize", type=float, default=6.5)
    parser.add_argument("--use-median", action="store_true")
    parser.add_argument(
        "--palette",
        choices=["soft_blue", "soft_purple", "cividis", "custom"],
        default="soft_blue",
        help="Unified color palette for all panels. Use custom to honor --cmap.",
    )
    parser.add_argument(
        "--cmap",
        default="cividis",
        help="Matplotlib colormap used when --palette custom.",
    )
    parser.add_argument(
        "--clearance-diverging",
        action="store_true",
        help="Use a separate diverging colormap for minimum clearance centered at zero.",
    )
    parser.add_argument("--clearance-cmap", default="RdBu")
    parser.add_argument("--clearance-vmin", type=float, default=-0.04)
    parser.add_argument("--clearance-vmax", type=float, default=0.03)
    parser.add_argument("--title-fontsize", type=float, default=9.0)
    parser.add_argument("--axis-label-fontsize", type=float, default=10.5)
    parser.add_argument("--tick-fontsize", type=float, default=9.0)
    parser.add_argument("--mark-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-marker", default="D")
    parser.add_argument("--baseline-marker-size", type=float, default=20.0)
    parser.add_argument("--baseline-marker-linewidth", type=float, default=0.7)
    parser.add_argument("--mark-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-phi-rr", type=float, default=0.01)
    parser.add_argument("--candidate-phi-hr", type=float, default=0.15)
    parser.add_argument("--candidate-marker", default="*")
    parser.add_argument("--candidate-marker-size", type=float, default=52.0)
    parser.add_argument("--candidate-marker-linewidth", type=float, default=0.7)
    parser.add_argument("--marker-offset", type=float, nargs=2, default=[0.28, 0.28])
    parser.add_argument("--marker-legend", action="store_true")
    parser.add_argument("--legend-fontsize", type=float, default=7.5)
    parser.add_argument("--colorbar-shrink", type=float, default=0.85)
    parser.add_argument("--figsize", type=float, nargs=2, default=[7.2, 5.8])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.summary_csv.expanduser())
    if not rows:
        raise RuntimeError(f"No rows found in {args.summary_csv}")
    plot_heatmaps(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
