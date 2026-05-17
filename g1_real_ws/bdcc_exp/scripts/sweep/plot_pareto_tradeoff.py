#!/usr/bin/env python3
"""Plot BDCC2026 safety-imitation Pareto trade-offs.

Composite default:

python3 /ws/bdcc_exp/scripts/sweep/plot_pareto_tradeoff.py \
  --summary-csv /ws/bdcc_exp/sweeps/real_merge_phi_grid/sweep_summary_agg.csv \
  --summary-csv /ws/bdcc_exp/sweeps/real_merge_gamma_grid/sweep_summary_agg.csv \
  --summary-csv /ws/bdcc_exp/sweeps/real_merge_pareto_samples/sweep_summary_agg.csv \
  --pareto-mode composite \
  --outdir /ws/bdcc_exp/figures/sweeps/real_merge_pareto \
  --formats png svg pdf \
  --show-errorbars \
  --mark-baseline
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "safe_ndtw_link_deg_mean_mean": "nDTW-link [deg]",
    "merge_safe_M_ctr_mean": "Collision-time ratio",
    "merge_safe_M_clear_m_mean": "Minimum clearance [m]",
    "merge_safe_M_cc_mean": "Collision count",
    "S_imit_mean": "Imitation score",
    "S_safe_mean": "Safety score",
}

MARKERS = {
    "phi_grid": "o",
    "gamma_grid": "s",
    "pareto_samples": "^",
}


def str_to_bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {text}")


def as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_summary(path: Path, source_label: str) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_csv"] = str(path)
        row["source_label"] = source_label
    return rows


def metric_label(metric: str) -> str:
    if metric in LABELS:
        return LABELS[metric]
    label = metric
    for suffix in ["_mean", "_median", "_std", "_q25", "_q75"]:
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    return label.replace("_", " ")


def std_column(metric: str) -> Optional[str]:
    if metric.endswith("_mean"):
        return metric[: -len("_mean")] + "_std"
    if metric.endswith("_median"):
        return metric[: -len("_median")] + "_std"
    return None


def metric_or_mean_column(color_by: str, rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return color_by
    if color_by in rows[0]:
        return color_by
    mean_col = f"{color_by}_mean"
    if mean_col in rows[0]:
        return mean_col
    return color_by


def source_label(row: Dict[str, Any]) -> str:
    return str(row.get("source_label") or row.get("sweep_type") or "sweep")


def marker_source(row: Dict[str, Any]) -> str:
    return str(row.get("sweep_type") or row.get("source_label") or "sweep")


def pareto_front_indices(
    xs: np.ndarray,
    ys: np.ndarray,
    lower_x_better: bool,
    lower_y_better: bool,
) -> List[int]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    indices = np.where(finite)[0]
    if indices.size == 0:
        return []

    score_x = -xs if lower_x_better else xs
    score_y = -ys if lower_y_better else ys
    front: List[int] = []
    for idx in indices:
        dominated = False
        for other in indices:
            if other == idx:
                continue
            at_least = score_x[other] >= score_x[idx] and score_y[other] >= score_y[idx]
            strictly = score_x[other] > score_x[idx] or score_y[other] > score_y[idx]
            if at_least and strictly:
                dominated = True
                break
        if not dominated:
            front.append(int(idx))
    return sorted(front, key=lambda i: xs[i])


def is_baseline(row: Dict[str, Any]) -> bool:
    return (
        math.isclose(as_float(row.get("phi_rr")), 0.03, abs_tol=1e-9)
        and math.isclose(as_float(row.get("phi_hr")), 0.15, abs_tol=1e-9)
        and math.isclose(as_float(row.get("gamma_rr")), 2.0, abs_tol=1e-9)
        and math.isclose(as_float(row.get("gamma_hr")), 3.0, abs_tol=1e-9)
    )


def param_annotation(row: Dict[str, Any]) -> str:
    key = str(row.get("param_key", ""))
    if key:
        return key
    return (
        f"({as_float(row.get('phi_rr')):.3f}, {as_float(row.get('phi_hr')):.3f}, "
        f"{as_float(row.get('gamma_rr')):.1f}, {as_float(row.get('gamma_hr')):.1f})"
    )


def setup_defaults(args: argparse.Namespace) -> None:
    if args.pareto_mode == "composite":
        if args.x_metric is None:
            args.x_metric = "S_imit_mean"
        if args.y_metric is None:
            args.y_metric = "S_safe_mean"
        if args.lower_x_better is None:
            args.lower_x_better = False
        if args.lower_y_better is None:
            args.lower_y_better = False
    else:
        if args.x_metric is None:
            args.x_metric = "safe_ndtw_link_deg_mean_mean"
        if args.y_metric is None:
            args.y_metric = "merge_safe_M_ctr_mean"
        if args.lower_x_better is None:
            args.lower_x_better = True
        if args.lower_y_better is None:
            args.lower_y_better = True


def plot_errorbars(
    ax: plt.Axes,
    rows: Sequence[Dict[str, Any]],
    xs: np.ndarray,
    ys: np.ndarray,
    args: argparse.Namespace,
) -> None:
    x_std_col = std_column(args.x_metric)
    y_std_col = std_column(args.y_metric)
    xerr = None
    yerr = None
    if x_std_col:
        vals = np.asarray([as_float(row.get(x_std_col)) for row in rows], dtype=float)
        xerr = np.where(np.isfinite(vals), vals, 0.0)
    if y_std_col:
        vals = np.asarray([as_float(row.get(y_std_col)) for row in rows], dtype=float)
        yerr = np.where(np.isfinite(vals), vals, 0.0)
    ax.errorbar(
        xs,
        ys,
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor="0.65",
        elinewidth=0.8,
        capsize=2,
        zorder=1,
    )


def plot_pareto(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    setup_defaults(args)
    color_col = metric_or_mean_column(args.color_by, rows)
    xs = np.asarray([as_float(row.get(args.x_metric)) for row in rows], dtype=float)
    ys = np.asarray([as_float(row.get(args.y_metric)) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=args.figsize, constrained_layout=True)

    if args.show_errorbars:
        plot_errorbars(ax, rows, xs, ys, args)

    color_values = [row.get(color_col, "") for row in rows]
    color_numeric = np.asarray([as_float(value) for value in color_values], dtype=float)
    use_numeric_color = np.any(np.isfinite(color_numeric)) and all(
        np.isfinite(v) or str(raw).strip() in {"", "nan", "None"}
        for v, raw in zip(color_numeric, color_values)
    )

    sources = sorted(set(source_label(row) for row in rows))
    if use_numeric_color:
        finite_color = np.where(np.isfinite(color_numeric), color_numeric, np.nan)
        for source in sources:
            idx = [i for i, row in enumerate(rows) if source_label(row) == source]
            if not idx:
                continue
            marker = MARKERS.get(marker_source(rows[idx[0]]), "o")
            sc = ax.scatter(
                xs[idx],
                ys[idx],
                c=finite_color[idx],
                cmap="viridis",
                marker=marker,
                s=52,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                label=source,
                zorder=2,
            )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(metric_label(color_col))
    else:
        categories = sorted(set(str(v) for v in color_values))
        cmap = plt.get_cmap("tab10")
        cat_color = {cat: cmap(i % 10) for i, cat in enumerate(categories)}
        for source in sources:
            for cat in categories:
                idx = [
                    i
                    for i, row in enumerate(rows)
                    if source_label(row) == source and str(row.get(color_col, "")) == cat
                ]
                if not idx:
                    continue
                marker = MARKERS.get(marker_source(rows[idx[0]]), "o")
                label = source if color_col == "sweep_type" else f"{source}: {cat}"
                ax.scatter(
                    xs[idx],
                    ys[idx],
                    c=[cat_color[cat]],
                    marker=marker,
                    s=52,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    label=label,
                    zorder=2,
                )

    front = pareto_front_indices(xs, ys, args.lower_x_better, args.lower_y_better)
    if front:
        ax.plot(xs[front], ys[front], color="black", linewidth=1.2, zorder=3)
        ax.scatter(xs[front], ys[front], facecolors="none", edgecolors="black", s=92, zorder=4)
        if args.annotate_front:
            for idx in front:
                ax.annotate(
                    param_annotation(rows[idx]),
                    (xs[idx], ys[idx]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

    if args.mark_baseline:
        baseline_idx = [i for i, row in enumerate(rows) if is_baseline(row)]
        if baseline_idx:
            ax.scatter(
                xs[baseline_idx],
                ys[baseline_idx],
                marker="*",
                s=210,
                c="gold",
                edgecolors="black",
                linewidths=0.9,
                label="Baseline",
                zorder=5,
            )

    ax.set_xlabel(metric_label(args.x_metric))
    ax.set_ylabel(metric_label(args.y_metric))
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="best")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_name = f"pareto_tradeoff_{args.pareto_mode}"
    for fmt in args.formats:
        path = args.outdir / f"{out_name}.{fmt}"
        fig.savefig(path, dpi=args.dpi)
        print(f"[pareto] wrote: {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BDCC Pareto trade-off scatter.")
    parser.add_argument("--summary-csv", action="append", required=True, type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"])
    parser.add_argument("--pareto-mode", choices=["raw", "composite"], default="composite")
    parser.add_argument("--x-metric", default=None)
    parser.add_argument("--y-metric", default=None)
    parser.add_argument("--color-by", default="sweep_type")
    parser.add_argument("--mark-baseline", action="store_true")
    parser.add_argument("--show-errorbars", action="store_true")
    parser.add_argument("--annotate-front", action="store_true")
    parser.add_argument("--lower-x-better", type=str_to_bool, default=None)
    parser.add_argument("--lower-y-better", type=str_to_bool, default=None)
    parser.add_argument("--figsize", type=float, nargs=2, default=[6.4, 4.8])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.summary_csv):
        raise ValueError("--labels length must match the number of --summary-csv entries")

    rows: List[Dict[str, Any]] = []
    for idx, path in enumerate(args.summary_csv):
        label = args.labels[idx] if args.labels else path.expanduser().parent.name
        rows.extend(read_summary(path.expanduser(), label))

    if not rows:
        raise RuntimeError("No rows found in summary CSV inputs")
    plot_pareto(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
