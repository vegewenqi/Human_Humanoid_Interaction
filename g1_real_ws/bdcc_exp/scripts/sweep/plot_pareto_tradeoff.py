#!/usr/bin/env python3
"""Plot BDCC2026 safety-imitation Pareto trade-offs.

Composite default:

python3 /ws/bdcc_exp/scripts/sweep/plot_pareto_tradeoff.py \
  --sweep-root /ws/bdcc_exp/sweeps \
  --pareto-mode composite \
  --outdir /ws/bdcc_exp/figures/sweeps/real_merge_pareto \
  --formats png svg pdf \
  --show-errorbars

Composite mode recomputes S_safe and S_imit from all per-run CSV rows with one
global min-max normalization before grouping repeats by the four CBF parameters.
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
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


LABELS = {
    "safe_ndtw_link_deg_mean_mean": "nDTW-link [deg]",
    "merge_safe_M_ctr_mean": "Collision-time ratio",
    "merge_safe_M_clear_m_mean": "Minimum clearance [m]",
    "merge_safe_M_cc_mean": "Collision count",
    "S_imit_mean": r"Imitation score ($S_{\rm{imit}}$)",
    "S_safe_mean": r"Safety score ($S_{\rm{safe}}$)",
    "phi_hr": r"Human--robot margin $\phi_s^{hr}$ [m]",
    "phi_hr_mean": r"Human--robot margin $\phi_s^{hr}$ [m]",
    "gamma_rr": r"$\gamma^{rr}$",
    "gamma_rr_mean": r"$\gamma^{rr}$",
}

PARAM_COLUMNS = ["phi_rr", "phi_hr", "gamma_rr", "gamma_hr"]
SWEEP_TYPES_IN_ORDER = ["phi_grid", "gamma_grid", "pareto_samples"]

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


def per_run_csv_for(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.name == "sweep_summary_per_run.csv":
        return expanded
    sibling = expanded.parent / "sweep_summary_per_run.csv"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"Composite Pareto needs per-run data for global score normalization. "
        f"Could not find {sibling} for input {expanded}"
    )


def discover_summary_csvs(sweep_root: Path) -> List[Path]:
    root = sweep_root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Sweep root does not exist: {root}")

    if (root / "sweep_summary_agg.csv").exists():
        return [root / "sweep_summary_agg.csv"]

    found: List[Path] = []
    seen = set()
    for sweep_type in SWEEP_TYPES_IN_ORDER:
        matches = sorted(
            child / "sweep_summary_agg.csv"
            for child in root.iterdir()
            if child.is_dir()
            and sweep_type in child.name
            and (child / "sweep_summary_agg.csv").exists()
        )
        if not matches:
            print(f"[pareto] warning: no {sweep_type} summary found under {root}")
        for path in matches:
            resolved = path.resolve()
            if resolved not in seen:
                found.append(path)
                seen.add(resolved)

    if not found:
        raise FileNotFoundError(
            f"No sweep_summary_agg.csv files found under {root}. "
            f"Expected child directories containing: {', '.join(SWEEP_TYPES_IN_ORDER)}"
        )
    return found


def resolve_summary_csvs(args: argparse.Namespace) -> None:
    paths: List[Path] = []
    if args.sweep_root is not None:
        paths.extend(discover_summary_csvs(args.sweep_root))
    if args.summary_csv:
        for path in args.summary_csv:
            expanded = path.expanduser()
            if expanded.is_dir():
                paths.extend(discover_summary_csvs(expanded))
            else:
                paths.append(expanded)

    if not paths:
        raise ValueError("Provide either --sweep-root or one or more --summary-csv inputs.")

    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(path)
        seen.add(resolved)

    args.summary_csv = deduped


def metric_label(metric: str) -> str:
    if metric in LABELS:
        return LABELS[metric]
    label = metric
    for suffix in ["_mean", "_median", "_std", "_q25", "_q75"]:
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    return label.replace("_", " ")


def metric_lower_better(metric: Optional[str]) -> bool:
    """Infer whether a plotted metric is an error/violation metric."""
    name = str(metric or "").lower()
    if not name:
        return True
    if "s_safe" in name or "s_imit" in name:
        return False
    if "clear" in name:
        return False
    lower_better_tokens = (
        "ctr",
        "cc",
        "collision_count",
        "collision-time",
        "rmse",
        "mse",
        "ndtw",
        "correction_norm",
        "error",
    )
    return any(token in name for token in lower_better_tokens)


def colorbar_label(metric: str) -> str:
    if metric in {"phi_hr", "phi_hr_mean"}:
        return r"$\phi_s^{hr}$"
    return metric_label(metric)


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


def derive_per_run_metrics(row: Dict[str, Any]) -> None:
    rmse_rad = as_float(row.get("rmse_q0_rad"))
    rmse_deg = as_float(row.get("rmse_q0_deg"))
    if not math.isfinite(rmse_deg) and math.isfinite(rmse_rad):
        rmse_deg = rmse_rad * 180.0 / math.pi
        row["rmse_q0_deg"] = rmse_deg

    if not math.isfinite(as_float(row.get("mse_q0_rad"))) and math.isfinite(rmse_rad):
        row["mse_q0_rad"] = rmse_rad**2
    if not math.isfinite(as_float(row.get("mse_q0_deg"))) and math.isfinite(rmse_deg):
        row["mse_q0_deg"] = rmse_deg**2

    safe_ndtw_rad = as_float(row.get("safe_ndtw_link_rad_mean"))
    if not math.isfinite(as_float(row.get("safe_ndtw_link_deg_mean"))) and math.isfinite(safe_ndtw_rad):
        row["safe_ndtw_link_deg_mean"] = safe_ndtw_rad * 180.0 / math.pi

    unsafe_ndtw_rad = as_float(row.get("unsafe_ndtw_link_rad_mean"))
    if not math.isfinite(as_float(row.get("unsafe_ndtw_link_deg_mean"))) and math.isfinite(unsafe_ndtw_rad):
        row["unsafe_ndtw_link_deg_mean"] = unsafe_ndtw_rad * 180.0 / math.pi


def compute_global_composite_scores(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
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


def parameter_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return tuple(round(as_float(row.get(col)), 12) for col in PARAM_COLUMNS)  # type: ignore[return-value]


def numeric_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    cols = sorted({key for row in rows for key in row.keys()})
    out = []
    for col in cols:
        if col in PARAM_COLUMNS:
            continue
        if any(math.isfinite(as_float(row.get(col))) for row in rows):
            out.append(col)
    return out


def aggregate_parameter_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, float, float, float], List[Dict[str, Any]]] = {}
    for row in rows:
        key = parameter_key(row)
        if all(math.isfinite(v) for v in key):
            groups.setdefault(key, []).append(row)

    num_cols = numeric_columns(rows)
    out_rows: List[Dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        out: Dict[str, Any] = {col: value for col, value in zip(PARAM_COLUMNS, key)}
        out["param_key"] = str(group[0].get("param_key") or "")
        out["sweep_type"] = "+".join(sorted(set(str(row.get("sweep_type", "")) for row in group if row.get("sweep_type"))))
        out["source_label"] = "+".join(
            sorted(set(str(row.get("source_label", "")) for row in group if row.get("source_label")))
        )
        out["n_runs"] = len(group)
        for col in num_cols:
            stats = stats_for([row.get(col) for row in group])
            for stat_name, value in stats.items():
                out[f"{col}_{stat_name}"] = value
        out_rows.append(out)
    return out_rows


def load_global_composite_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, path in enumerate(args.summary_csv):
        label = args.labels[idx] if args.labels else path.expanduser().parent.name
        per_run_path = per_run_csv_for(path)
        loaded = read_summary(per_run_path, label)
        for row in loaded:
            derive_per_run_metrics(row)
        rows.extend(loaded)

    if not rows:
        raise RuntimeError("No rows found in per-run CSV inputs")
    compute_global_composite_scores(args, rows)
    return aggregate_parameter_rows(rows)


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


def scaled_marker_sizes(
    rows: Sequence[Dict[str, Any]],
    size_col: str,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray([as_float(row.get(size_col)) for row in rows], dtype=float)
    sizes = np.full(values.shape, 0.5 * (args.size_min + args.size_max), dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return values, sizes

    gamma_min = float(np.nanmin(values[finite]))
    gamma_max = float(np.nanmax(values[finite]))
    gamma_norm = (values[finite] - gamma_min) / (gamma_max - gamma_min + 1e-12)
    sizes[finite] = args.size_min + (gamma_norm**args.size_exp) * (args.size_max - args.size_min)
    return values, sizes


def size_for_value(value: float, values: np.ndarray, args: argparse.Namespace) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.5 * (args.size_min + args.size_max)
    gamma_min = float(np.nanmin(finite))
    gamma_max = float(np.nanmax(finite))
    gamma_norm = (float(value) - gamma_min) / (gamma_max - gamma_min + 1e-12)
    gamma_norm = float(np.clip(gamma_norm, 0.0, 1.0))
    return args.size_min + (gamma_norm**args.size_exp) * (args.size_max - args.size_min)


def gamma_reference_values(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if args.gamma_size_refs:
        return np.asarray(args.gamma_size_refs, dtype=float)
    if finite.size == 0:
        return np.asarray([], dtype=float)
    gamma_min = float(np.nanmin(finite))
    gamma_max = float(np.nanmax(finite))
    return np.linspace(gamma_min, gamma_max, 5)


def size_legend_handles(values: np.ndarray, args: argparse.Namespace) -> List[Any]:
    refs = gamma_reference_values(values, args)
    handles = []
    for value in refs:
        handles.append(
            plt.scatter(
                [],
                [],
                s=size_for_value(float(value), values, args),
                facecolors="white",
                edgecolors="black",
                linewidths=args.legend_marker_linewidth,
                label=f"{value:.1f}",
            )
        )
    return handles


def add_size_ramp(ax: plt.Axes, values: np.ndarray, args: argparse.Namespace) -> None:
    refs = gamma_reference_values(values, args)
    if refs.size == 0:
        return

    inset = ax.inset_axes(args.size_ramp_box)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.set_facecolor((1.0, 1.0, 1.0, args.size_ramp_face_alpha))
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(args.size_ramp_frame_linewidth)
        spine.set_linestyle(":")

    xs = np.linspace(0.14, 0.88, refs.size)
    ys = np.full(refs.shape, 0.50, dtype=float)
    sizes = [size_for_value(float(value), values, args) * args.size_ramp_scale for value in refs]

    inset.plot(
        [float(xs[0]), float(xs[-1])],
        [0.50, 0.50],
        color="0.55",
        linewidth=0.8,
        alpha=0.65,
        zorder=1,
    )
    inset.scatter(
        xs,
        ys,
        s=sizes,
        facecolors="white",
        edgecolors="black",
        linewidths=args.legend_marker_linewidth,
        zorder=2,
    )
    inset.text(
        0.0,
        0.78,
        r"$\gamma^{rr}$",
        ha="left",
        va="center",
        fontsize=args.legend_fontsize,
        color="black",
        transform=inset.transAxes,
    )
    for x, value in zip(xs, refs):
        inset.text(
            float(x),
            0.05,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=max(args.legend_fontsize - 1.0, 5.0),
            color="black",
        )


def add_parameter_note(ax: plt.Axes, args: argparse.Namespace) -> None:
    if not args.parameter_note:
        return
    ax.text(
        args.parameter_note_xy[0],
        args.parameter_note_xy[1],
        r"Each point: $(\phi_s^{rr}, \phi_s^{hr}, \gamma^{rr}, \gamma^{hr})$",
        transform=ax.transAxes,
        ha=args.parameter_note_ha,
        va="top",
        fontsize=args.parameter_note_fontsize,
        color="black",
        zorder=7,
    )


def style_dotted_legend(legend: Any, args: argparse.Namespace) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(args.legend_frame_alpha)
    frame.set_edgecolor("black")
    frame.set_linewidth(args.legend_frame_linewidth)
    frame.set_linestyle(":")


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


def catmull_rom_smooth(xs: np.ndarray, ys: np.ndarray, samples_per_segment: int) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    pts = np.column_stack([xs[finite], ys[finite]])
    if pts.shape[0] < 3 or samples_per_segment <= 1:
        return xs[finite], ys[finite]

    # Duplicate endpoints so the spline starts and ends at empirical front points.
    extended = np.vstack([pts[0], pts, pts[-1]])
    curve: List[np.ndarray] = []
    for idx in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[idx - 1], extended[idx], extended[idx + 1], extended[idx + 2]
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            curve.append(point)
    curve.append(pts[-1])
    smoothed = np.asarray(curve, dtype=float)
    return smoothed[:, 0], smoothed[:, 1]


def fitted_pareto_curve(xs: np.ndarray, ys: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    fx = np.asarray(xs[finite], dtype=float)
    fy = np.asarray(ys[finite], dtype=float)
    if fx.size < 3:
        return fx, fy

    order = np.argsort(fx)
    fx = fx[order]
    fy = fy[order]

    if args.front_fit_method == "catmull":
        return catmull_rom_smooth(fx, fy, args.front_smooth_samples)

    # Least-squares polynomial trend: this is intentionally a fitted guide,
    # not a replacement for the empirical non-dominated front.
    degree = max(1, min(int(args.front_fit_degree), fx.size - 1))
    coeff = np.polyfit(fx, fy, degree)
    curve_x = np.linspace(float(np.min(fx)), float(np.max(fx)), args.front_fit_samples)
    curve_y = np.polyval(coeff, curve_x)
    if args.clip_fitted_front and args.pareto_mode == "composite":
        curve_y = np.clip(curve_y, args.ylim[0], args.ylim[1])
    return curve_x, curve_y


def closest_to_ideal_pareto_index(
    front: Sequence[int],
    xs: np.ndarray,
    ys: np.ndarray,
    lower_x_better: bool,
    lower_y_better: bool,
) -> Optional[int]:
    if not front:
        return None
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return None

    score_x = -xs if lower_x_better else xs
    score_y = -ys if lower_y_better else ys

    sx = score_x.copy()
    sy = score_y.copy()
    sx_min, sx_max = float(np.nanmin(sx[finite])), float(np.nanmax(sx[finite]))
    sy_min, sy_max = float(np.nanmin(sy[finite])), float(np.nanmax(sy[finite]))
    sx_norm = (sx - sx_min) / (sx_max - sx_min + 1e-12)
    sy_norm = (sy - sy_min) / (sy_max - sy_min + 1e-12)

    front_arr = np.asarray(front, dtype=int)
    dist = np.sqrt((1.0 - sx_norm[front_arr]) ** 2 + (1.0 - sy_norm[front_arr]) ** 2)
    finite_dist = np.isfinite(dist)
    if not np.any(finite_dist):
        return None
    valid_front = front_arr[finite_dist]
    valid_dist = dist[finite_dist]
    return int(valid_front[int(np.argmin(valid_dist))])


def failure_regime_index(
    xs: np.ndarray,
    ys: np.ndarray,
    lower_x_better: bool,
    lower_y_better: bool,
) -> Optional[int]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return None
    finite_idx = np.where(finite)[0]
    score_x = -xs if lower_x_better else xs
    score_y = -ys if lower_y_better else ys
    xvals = score_x[finite_idx]
    yvals = score_y[finite_idx]
    x_min, x_max = float(np.min(xvals)), float(np.max(xvals))
    y_min, y_max = float(np.min(yvals)), float(np.max(yvals))
    x_norm = (xvals - x_min) / (x_max - x_min + 1e-12)
    y_norm = (yvals - y_min) / (y_max - y_min + 1e-12)
    score = x_norm + y_norm
    return int(finite_idx[int(np.argmin(score))])


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
            args.y_metric = "merge_safe_M_clear_m_mean"
        if args.lower_x_better is None:
            args.lower_x_better = metric_lower_better(args.x_metric)
        if args.lower_y_better is None:
            args.lower_y_better = metric_lower_better(args.y_metric)


def plot_errorbars(
    ax: plt.Axes,
    rows: Sequence[Dict[str, Any]],
    xs: np.ndarray,
    ys: np.ndarray,
    args: argparse.Namespace,
    indices: Optional[Sequence[int]] = None,
) -> None:
    if indices is None:
        idx = np.arange(len(rows), dtype=int)
    else:
        idx = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    if idx.size == 0:
        return

    x_std_col = std_column(args.x_metric)
    y_std_col = std_column(args.y_metric)
    xerr = None
    yerr = None
    if x_std_col:
        vals = np.asarray([as_float(row.get(x_std_col)) for row in rows], dtype=float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        xerr = vals[idx]
    if y_std_col:
        vals = np.asarray([as_float(row.get(y_std_col)) for row in rows], dtype=float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        yerr = vals[idx]
    ax.errorbar(
        xs[idx],
        ys[idx],
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor=args.errorbar_color,
        elinewidth=args.errorbar_linewidth,
        capsize=args.errorbar_capsize,
        alpha=args.errorbar_alpha,
        zorder=2,
    )


def selected_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    return [i for i, row in enumerate(rows) if is_baseline(row)]


def add_operating_regions(ax: plt.Axes, args: argparse.Namespace) -> None:
    if not args.show_regions or args.pareto_mode != "composite":
        return
    ax.axhspan(
        args.high_safety_threshold,
        1.05,
        color=args.high_safety_color,
        alpha=args.region_alpha,
        linewidth=0,
        zorder=0,
    )
    ax.axvspan(
        args.high_imitation_threshold,
        1.05,
        color=args.high_imitation_color,
        alpha=args.region_alpha,
        linewidth=0,
        zorder=0,
    )
    ax.text(
        0.02,
        args.high_safety_threshold + 0.012,
        "High safety region",
        fontsize=args.region_label_fontsize,
        color="0.4",
        zorder=1,
    )
    ax.text(
        args.high_imitation_threshold + 0.01,
        0.12,
        "High imitation region",
        fontsize=args.region_label_fontsize,
        color="0.4",
        rotation=90,
        va="bottom",
        zorder=1,
    )


def add_marginal_histograms(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, args: argparse.Namespace) -> None:
    if not args.show_marginals:
        return
    finite_x = xs[np.isfinite(xs)]
    finite_y = ys[np.isfinite(ys)]
    if finite_x.size:
        ax_top = ax.inset_axes([0.0, 1.03, 1.0, 0.18], transform=ax.transAxes)
        ax_top.hist(finite_x, bins=args.marginal_bins, color="0.25", alpha=args.marginal_alpha)
        ax_top.set_xlim(ax.get_xlim())
        ax_top.axis("off")
    if finite_y.size:
        ax_right = ax.inset_axes([1.03, 0.0, 0.18, 1.0], transform=ax.transAxes)
        ax_right.hist(
            finite_y,
            bins=args.marginal_bins,
            orientation="horizontal",
            color="0.25",
            alpha=args.marginal_alpha,
        )
        ax_right.set_ylim(ax.get_ylim())
        ax_right.axis("off")


def set_paper_axes_style(ax: plt.Axes, args: argparse.Namespace) -> None:
    ax.grid(True, linewidth=0.45, alpha=args.grid_alpha)
    for spine in ax.spines.values():
        spine.set_linewidth(args.spine_linewidth)
    if args.pareto_mode == "composite":
        ax.set_xlim(args.xlim)
        ax.set_ylim(args.ylim)


def apply_style_defaults(args: argparse.Namespace) -> None:
    if args.color_cmap:
        args.cmap = args.color_cmap
    if args.use_existing_scores:
        args.use_agg_composite_scores = True

    if args.show_regions is None:
        args.show_regions = args.style in {"paper", "advanced"}
    if args.show_marginals is None:
        args.show_marginals = args.style == "advanced"
    if args.smooth_front is not None:
        args.show_fitted_front = args.smooth_front


def plot_pareto(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    setup_defaults(args)
    color_col = metric_or_mean_column(args.color_by, rows)
    size_col = metric_or_mean_column(args.size_by, rows)
    xs = np.asarray([as_float(row.get(args.x_metric)) for row in rows], dtype=float)
    ys = np.asarray([as_float(row.get(args.y_metric)) for row in rows], dtype=float)
    size_values, point_sizes = scaled_marker_sizes(rows, size_col, args)
    front = pareto_front_indices(xs, ys, args.lower_x_better, args.lower_y_better)
    selected_idx = selected_indices(rows) if args.mark_baseline else []
    pareto_point_idx = (
        closest_to_ideal_pareto_index(front, xs, ys, args.lower_x_better, args.lower_y_better)
        if args.mark_pareto_point
        else None
    )
    failure_idx = (
        failure_regime_index(xs, ys, args.lower_x_better, args.lower_y_better)
        if args.mark_failure_sample
        else None
    )

    fig, ax = plt.subplots(figsize=args.figsize, constrained_layout=True)
    add_operating_regions(ax, args)

    if args.show_errorbars and args.errorbar_mode != "none":
        if args.errorbar_mode == "all":
            plot_errorbars(ax, rows, xs, ys, args)
        else:
            plot_errorbars(ax, rows, xs, ys, args, indices=list(front) + selected_idx)

    color_values = [row.get(color_col, "") for row in rows]
    color_numeric = np.asarray([as_float(value) for value in color_values], dtype=float)
    use_numeric_color = np.any(np.isfinite(color_numeric)) and all(
        np.isfinite(v) or str(raw).strip() in {"", "nan", "None"}
        for v, raw in zip(color_numeric, color_values)
    )
    color_norm = None

    if use_numeric_color:
        finite_color = np.where(np.isfinite(color_numeric), color_numeric, np.nan)
        sc = ax.scatter(
            xs,
            ys,
            c=finite_color,
            cmap=args.cmap,
            marker="o",
            s=point_sizes,
            edgecolors=args.point_edgecolor,
            linewidths=args.point_linewidth,
            alpha=args.point_alpha,
            zorder=3,
        )
        color_norm = sc.norm
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        if args.colorbar_label_bottom:
            cbar.set_label("")
            cbar.ax.set_xlabel(colorbar_label(color_col), labelpad=args.colorbar_labelpad)
            cbar.ax.xaxis.set_label_position("bottom")
        else:
            cbar.set_label(metric_label(color_col))
    else:
        categories = sorted(set(str(v) for v in color_values))
        cmap = plt.get_cmap("tab10")
        cat_color = {cat: cmap(i % 10) for i, cat in enumerate(categories)}
        for cat in categories:
            idx = [i for i, row in enumerate(rows) if str(row.get(color_col, "")) == cat]
            if not idx:
                continue
            ax.scatter(
                xs[idx],
                ys[idx],
                c=[cat_color[cat]],
                marker="o",
                s=point_sizes[idx],
                edgecolors=args.point_edgecolor,
                linewidths=args.point_linewidth,
                alpha=args.point_alpha,
                label=cat if args.category_legend else "_nolegend_",
                zorder=3,
            )

    if front:
        if args.show_empirical_front:
            ax.plot(
                xs[front],
                ys[front],
                color=args.empirical_front_color,
                linewidth=args.empirical_front_linewidth,
                linestyle=args.empirical_front_linestyle,
                alpha=args.empirical_front_alpha,
                zorder=4,
            )
        if args.show_fitted_front:
            curve_x, curve_y = fitted_pareto_curve(xs[front], ys[front], args)
            ax.plot(
                curve_x,
                curve_y,
                color=args.front_color,
                linewidth=args.front_linewidth,
                linestyle=args.front_linestyle,
                alpha=args.front_alpha,
                zorder=4.5,
            )
        if args.show_front_points and use_numeric_color:
            ax.scatter(
                xs[front],
                ys[front],
                c=finite_color[front],
                cmap=args.cmap,
                norm=color_norm,
                marker="o",
                s=point_sizes[front],
                edgecolors="black",
                linewidths=args.front_point_linewidth,
                alpha=1.0,
                zorder=5,
            )
        elif args.show_front_points:
            ax.scatter(
                xs[front],
                ys[front],
                facecolors="white",
                edgecolors="black",
                marker="o",
                s=point_sizes[front],
                linewidths=args.front_point_linewidth,
                alpha=1.0,
                zorder=5,
            )
        if args.annotate_front:
            for idx in front:
                ax.annotate(
                    param_annotation(rows[idx]),
                    (xs[idx], ys[idx]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

    if pareto_point_idx is not None:
        ax.scatter(
            [xs[pareto_point_idx]],
            [ys[pareto_point_idx]],
            marker=args.pareto_point_marker,
            s=args.pareto_point_size,
            facecolors=args.pareto_point_facecolor,
            edgecolors=args.pareto_point_edgecolor,
            linewidths=args.pareto_point_linewidth,
            label=args.pareto_point_label,
            zorder=7,
        )
        if args.annotate_pareto_point:
            ax.annotate(
                args.pareto_point_annotation,
                (xs[pareto_point_idx], ys[pareto_point_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=args.annotation_fontsize,
            )

    if failure_idx is not None:
        ax.scatter(
            [xs[failure_idx]],
            [ys[failure_idx]],
            marker=args.failure_marker,
            s=args.failure_size,
            facecolors="none",
            edgecolors="black",
            linewidths=args.failure_linewidth,
            label=args.failure_label,
            zorder=7,
        )
        if args.annotate_failure_sample:
            ax.annotate(
                args.failure_annotation,
                (xs[failure_idx], ys[failure_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=args.annotation_fontsize,
            )

    if args.mark_baseline:
        if selected_idx:
            ax.scatter(
                xs[selected_idx],
                ys[selected_idx],
                marker=args.selected_marker,
                s=args.selected_size,
                c=args.selected_color,
                edgecolors="black",
                linewidths=args.selected_linewidth,
                label="Selected operating point",
                zorder=8,
            )
            if args.annotate_selected:
                for idx in selected_idx:
                    ax.annotate(
                        "selected",
                        (xs[idx], ys[idx]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=args.annotation_fontsize,
                    )

    ax.set_xlabel(metric_label(args.x_metric))
    ax.set_ylabel(metric_label(args.y_metric))
    set_paper_axes_style(ax, args)
    add_marginal_histograms(ax, xs, ys, args)
    add_parameter_note(ax, args)

    legend_handles = []
    if front and args.show_empirical_front:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=args.empirical_front_color,
                linewidth=args.empirical_front_linewidth,
                linestyle=args.empirical_front_linestyle,
                alpha=args.empirical_front_alpha,
                label=args.empirical_front_label,
            )
        )
    if front and args.show_fitted_front:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=args.front_color,
                linewidth=args.front_linewidth,
                linestyle=args.front_linestyle,
                alpha=args.front_alpha,
                label=args.front_label,
            )
        )
    if pareto_point_idx is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=args.pareto_point_marker,
                color=args.pareto_point_edgecolor,
                markerfacecolor=args.pareto_point_facecolor,
                linestyle="None",
                markersize=9,
                markeredgewidth=args.pareto_point_legend_linewidth,
                label=args.pareto_point_label,
            )
        )
    if failure_idx is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=args.failure_marker,
                color="black",
                markerfacecolor="none",
                linestyle="None",
                markersize=7,
                markeredgewidth=args.failure_linewidth,
                label=args.failure_label,
            )
        )
    if args.mark_baseline and selected_idx:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=args.selected_marker,
                color="black",
                markerfacecolor=args.selected_color,
                linestyle="None",
                markersize=args.selected_legend_markersize,
                markeredgewidth=args.selected_linewidth,
                label="Selected operating point",
            )
        )
    if legend_handles:
        legend_kwargs = {
            "handles": legend_handles,
            "frameon": args.legend_frame,
            "fontsize": args.legend_fontsize,
            "loc": args.pareto_legend_loc,
        }
        if args.pareto_legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(args.pareto_legend_bbox)
        pareto_legend = ax.legend(**legend_kwargs)
        if args.legend_frame:
            style_dotted_legend(pareto_legend, args)
        ax.add_artist(pareto_legend)

    if args.size_legend and args.size_legend_style == "ramp":
        add_size_ramp(ax, size_values, args)
    elif args.size_legend and args.size_legend_style == "bubbles":
        handles = size_legend_handles(size_values, args)
        if handles:
            ax.legend(
                handles=handles,
                title=metric_label(size_col),
                frameon=False,
                fontsize=args.legend_fontsize,
                title_fontsize=args.legend_fontsize,
                loc=args.size_legend_loc,
                labelspacing=1.2,
                borderpad=0.2,
            )
    elif args.category_legend:
        ax.legend(frameon=False, fontsize=args.legend_fontsize, loc="best")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_name = f"pareto_tradeoff_{args.pareto_mode}"
    for fmt in args.formats:
        path = args.outdir / f"{out_name}.{fmt}"
        fig.savefig(path, dpi=args.dpi)
        print(f"[pareto] wrote: {path}")
    plt.close(fig)


def plot_pareto_3d(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    setup_defaults(args)
    color_col = metric_or_mean_column(args.color_by, rows)
    size_col = metric_or_mean_column(args.size_by, rows)
    z_col = metric_or_mean_column(args.z_param, rows)

    xs = np.asarray([as_float(row.get(args.x_metric)) for row in rows], dtype=float)
    ys = np.asarray([as_float(row.get(args.y_metric)) for row in rows], dtype=float)
    zs = np.asarray([as_float(row.get(z_col)) for row in rows], dtype=float)
    colors_numeric = np.asarray([as_float(row.get(color_col)) for row in rows], dtype=float)
    phi_rr = np.asarray([as_float(row.get("phi_rr")) for row in rows], dtype=float)
    size_values, point_sizes = scaled_marker_sizes(rows, size_col, args)

    finite_color = colors_numeric[np.isfinite(colors_numeric)]
    if finite_color.size:
        norm = plt.Normalize(vmin=float(np.nanmin(finite_color)), vmax=float(np.nanmax(finite_color)))
    else:
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(args.cmap)

    fig = plt.figure(figsize=args.figsize)
    ax = fig.add_subplot(111, projection="3d")

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    phi_values = sorted(set(round(v, 12) for v in phi_rr if math.isfinite(v)))
    if not phi_values:
        phi_values = [math.nan]

    for marker_idx, phi_value in enumerate(phi_values):
        if math.isfinite(phi_value):
            idx = np.where(np.isclose(phi_rr, phi_value, atol=1e-12))[0]
            label = rf"$\phi_s^{{rr}}={phi_value:.3f}$"
        else:
            idx = np.arange(len(rows))
            label = r"$\phi_s^{rr}$"
        if idx.size == 0:
            continue
        ax.scatter(
            xs[idx],
            ys[idx],
            zs[idx],
            c=colors_numeric[idx],
            cmap=cmap,
            norm=norm,
            s=point_sizes[idx],
            marker=marker_cycle[marker_idx % len(marker_cycle)],
            edgecolors=args.point_edgecolor,
            linewidths=args.point_linewidth,
            alpha=args.point_alpha,
            label=label,
            depthshade=False,
        )

    front = pareto_front_indices(xs, ys, args.lower_x_better, args.lower_y_better)
    if front:
        ax.plot(xs[front], ys[front], zs[front], color="black", linewidth=args.front_linewidth)

    selected_idx = selected_indices(rows) if args.mark_baseline else []
    pareto_point_idx = (
        closest_to_ideal_pareto_index(front, xs, ys, args.lower_x_better, args.lower_y_better)
        if args.mark_pareto_point
        else None
    )
    if selected_idx:
        ax.scatter(
            xs[selected_idx],
            ys[selected_idx],
            zs[selected_idx],
            marker=args.selected_marker,
            s=args.selected_size,
            c=args.selected_color,
            edgecolors="black",
            linewidths=args.selected_linewidth,
            label="Selected operating point",
            depthshade=False,
        )
    if pareto_point_idx is not None:
        ax.scatter(
            [xs[pareto_point_idx]],
            [ys[pareto_point_idx]],
            [zs[pareto_point_idx]],
            marker=args.pareto_point_marker,
            s=args.pareto_point_size,
            facecolors=args.pareto_point_facecolor,
            edgecolors=args.pareto_point_edgecolor,
            linewidths=args.pareto_point_linewidth,
            label=args.pareto_point_label,
            depthshade=False,
        )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.10, shrink=0.72)
    cbar.set_label(metric_label(color_col))

    ax.set_xlabel(metric_label(args.x_metric), labelpad=8)
    ax.set_ylabel(metric_label(args.y_metric), labelpad=8)
    ax.set_zlabel(metric_label(z_col), labelpad=8)
    ax.view_init(elev=args.view_elev, azim=args.view_azim)
    ax.grid(True, alpha=args.grid_alpha)
    ax.legend(frameon=False, fontsize=args.legend_fontsize, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_name = f"pareto_tradeoff_{args.pareto_mode}_3d"
    for fmt in args.formats:
        path = args.outdir / f"{out_name}.{fmt}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"[pareto] wrote: {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BDCC Pareto trade-off scatter.")
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="Sweep root containing real_merge_phi_grid, real_merge_gamma_grid, and/or real_merge_pareto_samples.",
    )
    parser.add_argument(
        "--summary-csv",
        action="append",
        default=None,
        type=Path,
        help="Specific sweep_summary_agg.csv input. Can still be repeated; --sweep-root is preferred.",
    )
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"])
    parser.add_argument("--pareto-mode", choices=["raw", "composite"], default="composite")
    parser.add_argument("--x-metric", default=None)
    parser.add_argument("--y-metric", default=None)
    parser.add_argument("--color-by", default="phi_hr")
    parser.add_argument("--size-by", default="gamma_rr")
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--color-cmap", default=None, help="Deprecated alias for --cmap.")
    parser.add_argument("--size-min", type=float, default=35.0)
    parser.add_argument("--size-max", type=float, default=170.0)
    parser.add_argument("--point-size", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--point-size-min", dest="size_min", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--point-size-max", dest="size_max", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--size-exp", type=float, default=1.2)
    parser.add_argument("--gamma-size-refs", type=float, nargs="*", default=None)
    parser.add_argument("--marker", default="o", help=argparse.SUPPRESS)
    parser.add_argument("--point-edgecolor", default="white")
    parser.add_argument("--point-linewidth", type=float, default=0.4)
    parser.add_argument("--point-alpha", type=float, default=0.76)
    parser.add_argument("--show-empirical-front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--empirical-front-label", default="Empirical Pareto front")
    parser.add_argument("--empirical-front-color", default="0.25")
    parser.add_argument("--empirical-front-linewidth", type=float, default=1.6)
    parser.add_argument("--empirical-front-linestyle", default="--")
    parser.add_argument("--empirical-front-alpha", type=float, default=0.75)
    parser.add_argument("--show-fitted-front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front-color", default="black")
    parser.add_argument("--front-linewidth", type=float, default=1.4)
    parser.add_argument("--front-linestyle", default="-")
    parser.add_argument("--front-alpha", type=float, default=1.0)
    parser.add_argument("--front-point-linewidth", type=float, default=1.2)
    parser.add_argument("--show-front-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--front-fit-method", choices=["poly", "catmull"], default="poly")
    parser.add_argument("--front-fit-degree", type=int, default=3)
    parser.add_argument("--front-fit-samples", type=int, default=240)
    parser.add_argument("--clip-fitted-front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-front", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--front-smooth-samples", type=int, default=24)
    parser.add_argument("--front-label", default="Fitted Pareto trend")
    parser.add_argument("--selected-marker", default="D")
    parser.add_argument("--selected-color", default="white")
    parser.add_argument("--selected-size", type=float, default=125.0)
    parser.add_argument("--selected-linewidth", type=float, default=1.0)
    parser.add_argument("--selected-legend-markersize", type=float, default=6.0)
    parser.add_argument("--mark-pareto-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pareto-point-marker", default="*")
    parser.add_argument("--pareto-point-facecolor", default="white")
    parser.add_argument("--pareto-point-edgecolor", default="black")
    parser.add_argument("--pareto-point-size", type=float, default=310.0)
    parser.add_argument("--pareto-point-linewidth", type=float, default=0.85)
    parser.add_argument("--pareto-point-legend-linewidth", type=float, default=0.75)
    parser.add_argument("--pareto-point-label", default="Pareto knee point")
    parser.add_argument("--annotate-pareto-point", action="store_true")
    parser.add_argument("--pareto-point-annotation", default="closest-to-ideal")
    parser.add_argument("--annotate-selected", action="store_true")
    parser.add_argument("--annotation-fontsize", type=float, default=8.0)
    parser.add_argument("--size-legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--size-legend-style", choices=["ramp", "bubbles"], default="ramp")
    parser.add_argument("--size-legend-loc", default="lower left")
    parser.add_argument("--size-ramp-box", type=float, nargs=4, default=[0.05, 0.055, 0.34, 0.13])
    parser.add_argument("--size-ramp-scale", type=float, default=0.72)
    parser.add_argument("--size-ramp-face-alpha", type=float, default=0.86)
    parser.add_argument("--size-ramp-frame-linewidth", type=float, default=0.55)
    parser.add_argument("--pareto-legend-loc", default="center left")
    parser.add_argument("--pareto-legend-bbox", type=float, nargs=2, default=[0.02, 0.43])
    parser.add_argument("--legend-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--legend-frame-alpha", type=float, default=0.82)
    parser.add_argument("--legend-frame-linewidth", type=float, default=0.55)
    parser.add_argument("--category-legend", action="store_true")
    parser.add_argument("--mark-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-errorbars", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--errorbar-mode", choices=["all", "pareto", "none"], default="all")
    parser.add_argument("--errorbar-color", default="0.65")
    parser.add_argument("--errorbar-alpha", type=float, default=0.45)
    parser.add_argument("--errorbar-linewidth", type=float, default=0.8)
    parser.add_argument("--errorbar-capsize", type=float, default=2.0)
    parser.add_argument("--annotate-front", action="store_true")
    parser.add_argument("--lower-x-better", type=str_to_bool, default=None)
    parser.add_argument("--lower-y-better", type=str_to_bool, default=None)
    parser.add_argument("--show-regions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-safety-threshold", type=float, default=0.70)
    parser.add_argument("--high-imitation-threshold", type=float, default=0.70)
    parser.add_argument("--high-safety-color", default="#7fc97f")
    parser.add_argument("--high-imitation-color", default="#80b1d3")
    parser.add_argument("--region-alpha", type=float, default=0.12)
    parser.add_argument("--region-label-fontsize", type=float, default=10.0)
    parser.add_argument("--show-marginals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--marginal-alpha", type=float, default=0.16)
    parser.add_argument("--marginal-bins", type=int, default=14)
    parser.add_argument("--style", choices=["paper", "advanced", "clean"], default="paper")
    parser.add_argument("--parameter-note", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parameter-note-xy", type=float, nargs=2, default=[0.97, 0.98])
    parser.add_argument("--parameter-note-ha", choices=["left", "center", "right"], default="right")
    parser.add_argument("--parameter-note-fontsize", type=float, default=9.0)
    parser.add_argument("--mark-failure-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--failure-marker", default="o")
    parser.add_argument("--failure-size", type=float, default=160.0)
    parser.add_argument("--failure-linewidth", type=float, default=1.0)
    parser.add_argument("--failure-label", default="Failure-regime point")
    parser.add_argument("--annotate-failure-sample", action="store_true")
    parser.add_argument("--failure-annotation", default="failure-regime sample")
    parser.add_argument("--plot-3d", action="store_true")
    parser.add_argument("--z-param", choices=["phi_rr", "phi_hr", "gamma_rr", "gamma_hr"], default="gamma_hr")
    parser.add_argument("--view-elev", type=float, default=22.0)
    parser.add_argument("--view-azim", type=float, default=-58.0)
    parser.add_argument("--safety-weights", type=float, nargs=3, default=[0.45, 0.45, 0.10])
    parser.add_argument("--imitation-weights", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--score-normalization", choices=["minmax"], default="minmax")
    parser.add_argument("--score-rmse-metric", default="rmse_q0_deg")
    parser.add_argument("--score-ndtw-metric", default="safe_ndtw_link_deg_mean")
    parser.add_argument(
        "--use-agg-composite-scores",
        action="store_true",
        help="Use precomputed S_safe/S_imit from sweep_summary_agg.csv. "
        "Default composite mode recomputes scores globally from per-run CSVs.",
    )
    parser.add_argument("--use-existing-scores", action="store_true", help="Alias for --use-agg-composite-scores.")
    parser.add_argument("--legend-fontsize", type=float, default=8.0)
    parser.add_argument("--legend-marker-linewidth", type=float, default=0.7)
    parser.add_argument("--colorbar-label-bottom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--colorbar-labelpad", type=float, default=8.0)
    parser.add_argument("--grid-alpha", type=float, default=0.22)
    parser.add_argument("--spine-linewidth", type=float, default=1.0)
    parser.add_argument("--xlim", type=float, nargs=2, default=[0.0, 1.05])
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.0, 1.05])
    parser.add_argument("--figsize", type=float, nargs=2, default=[6.6, 5.1])
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_style_defaults(args)
    resolve_summary_csvs(args)
    if args.labels is not None and len(args.labels) != len(args.summary_csv):
        raise ValueError("--labels length must match the number of --summary-csv entries")

    if args.pareto_mode == "composite" and not args.use_agg_composite_scores:
        rows = load_global_composite_rows(args)
    else:
        rows: List[Dict[str, Any]] = []
        for idx, path in enumerate(args.summary_csv):
            label = args.labels[idx] if args.labels else path.expanduser().parent.name
            rows.extend(read_summary(path.expanduser(), label))

    if not rows:
        raise RuntimeError("No rows found in summary CSV inputs")
    plot_pareto(args, rows)
    if args.plot_3d:
        plot_pareto_3d(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
