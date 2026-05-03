#!/usr/bin/env python3
"""Render thesis-friendly training curves from <output_dir>/training_metrics.jsonl
(written by scripts/_train_with_step_save.py).

Outputs (saved next to the input file by default):
  - loss_curve.png      : training loss vs global_step (raw + EMA-smoothed)
  - lr_schedule.png     : learning_rate vs global_step (linear y-axis)
  - grad_norm.png       : grad_norm vs global_step (log y-axis if range warrants)

If --eval-results <csv> is provided, additionally renders
  - eval_metric_vs_step.png

The CSV is expected to have columns ``step,metric`` (any single metric column
name works; the first non-step column is plotted).

Usage:
  python3 scripts/plot_training_curves.py \
      --metrics /mnt/.../direct_5k/training_metrics.jsonl

  python3 scripts/plot_training_curves.py \
      --metrics /mnt/.../direct_5k/training_metrics.jsonl \
      --out-dir /tmp/figures \
      --eval-results /mnt/.../direct_5k_eval/mmbench_acc_per_step.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[int] = []
    ys: list[float] = []
    for r in rows:
        v = r.get(key)
        s = r.get("global_step")
        if isinstance(v, (int, float)) and isinstance(s, (int, float)):
            xs.append(int(s))
            ys.append(float(v))
    return np.asarray(xs), np.asarray(ys, dtype=float)


def _summarise_numeric_fields(rows: list[dict]) -> list[str]:
    """Return sorted unique keys whose values include at least one numeric entry."""
    keys: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    return sorted(keys)


def _read_tb_scalars(tb_dir: Path, candidate_tags: list[str]) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Try to read a scalar series from TensorBoard event files in ``tb_dir``.

    Returns (steps, values, matched_tag). All empty / None if not found or
    the ``tensorboard`` package is unavailable.
    """
    if not tb_dir.is_dir():
        return np.empty(0), np.empty(0), None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print(
            "[plot] tensorboard package not importable; "
            "cannot fall back to TB events for missing scalars.",
            flush=True,
        )
        return np.empty(0), np.empty(0), None

    event_files = sorted(tb_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        return np.empty(0), np.empty(0), None

    available_tags: set[str] = set()
    for ef in event_files:
        try:
            acc = EventAccumulator(str(ef.parent), size_guidance={"scalars": 0})
            acc.Reload()
            available_tags.update(acc.Tags().get("scalars", []))
        except Exception:
            continue

    matched: Optional[str] = None
    for tag in candidate_tags:
        if tag in available_tags:
            matched = tag
            break
    if matched is None:
        lowered = {t.lower(): t for t in available_tags}
        for tag in candidate_tags:
            if tag.lower() in lowered:
                matched = lowered[tag.lower()]
                break
    if matched is None:
        return np.empty(0), np.empty(0), None

    steps: list[int] = []
    vals: list[float] = []
    seen: set[int] = set()
    for ef in event_files:
        try:
            acc = EventAccumulator(str(ef.parent), size_guidance={"scalars": 0})
            acc.Reload()
            if matched not in acc.Tags().get("scalars", []):
                continue
            for entry in acc.Scalars(matched):
                if entry.step in seen:
                    continue
                seen.add(entry.step)
                steps.append(int(entry.step))
                vals.append(float(entry.value))
        except Exception:
            continue
    if not steps:
        return np.empty(0), np.empty(0), matched
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(vals, dtype=float)[order], matched


def _setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="both", labelsize=10)


def _auto_zoom_range(smoothed: np.ndarray) -> tuple[float, float]:
    """Auto-derive a ymin, ymax for the convergence-tail zoom panel.

    Heuristic: take the last 50% of the smoothed series, then add a small
    margin around its [min, max] so the band has visual headroom.
    """
    if smoothed.size == 0:
        return 0.0, 1.0
    tail = smoothed[smoothed.size // 2 :]
    lo, hi = float(np.nanmin(tail)), float(np.nanmax(tail))
    span = max(hi - lo, 1e-3)
    return max(lo - 0.2 * span, 0.0), hi + 0.2 * span


def _parse_zoom_range(spec: Optional[str]) -> Optional[tuple[float, float]]:
    if not spec:
        return None
    try:
        a, b = spec.split(",")
        ymin, ymax = float(a), float(b)
        if ymax <= ymin:
            raise ValueError("ymax must be > ymin")
        return ymin, ymax
    except Exception as e:
        raise SystemExit(f"--loss-zoom-range must be 'ymin,ymax', got {spec!r} ({e})")


def plot_loss(
    rows: list[dict],
    out_path: Path,
    style: str = "clean",
    zoom_range: Optional[tuple[float, float]] = None,
) -> None:
    """Plot training loss in one of several styles.

    style:
      - "clean":    single smoothed line, linear y-axis. Default; thesis-friendly.
      - "with_raw": same as clean but also overlays the raw curve in faint gray.
      - "log":      single smoothed line, log y-axis with plain decimal ticks.
      - "dual":     two side-by-side panels (linear | log).
      - "zoom":     single panel, y-axis clipped to ``zoom_range`` (auto if not
                    given). Best for showing late-stage convergence detail.
      - "split":    two stacked panels: top = full range, bottom = zoom into
                    ``zoom_range`` (auto if not given). Recommended for thesis
                    when you want both the global drop and the convergence
                    detail in a single figure.
    """
    x, y = _series(rows, "loss")
    if x.size == 0:
        print("[plot] no 'loss' field found, skipping loss_curve.png")
        return

    smoothed = _ema(y, alpha=0.05)
    min_idx = int(np.argmin(smoothed))

    if zoom_range is None and style in ("zoom", "split"):
        zoom_range = _auto_zoom_range(smoothed)

    def _draw(
        ax,
        *,
        log_y: bool = False,
        with_raw: bool = False,
        ylim: Optional[tuple[float, float]] = None,
        annotate_min: bool = True,
        legend: bool = True,
    ) -> None:
        if with_raw:
            ax.plot(x, y, color="#a9b4c8", linewidth=0.9, alpha=0.55, label="raw")
            ax.plot(x, smoothed, color="#1f4e8c", linewidth=2.0, label="smoothed")
            if legend:
                ax.legend(fontsize=10, loc="upper right")
        else:
            ax.plot(x, smoothed, color="#1f4e8c", linewidth=2.0)
        if annotate_min:
            mx, my = x[min_idx], smoothed[min_idx]
            in_view = ylim is None or (ylim[0] <= my <= ylim[1])
            if in_view:
                ax.scatter([mx], [my], color="#c7522a", zorder=5, s=36)
                ax.annotate(
                    f"min loss = {my:.4f} @ step {mx}",
                    xy=(mx, my),
                    xytext=(10, 12),
                    textcoords="offset points",
                    fontsize=9,
                    color="#5a2515",
                )
        if log_y:
            ax.set_yscale("log")
            from matplotlib.ticker import ScalarFormatter
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            fmt.set_useOffset(False)
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(fmt)
            ax.tick_params(axis="y", which="minor", labelsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)

    used = style
    if style == "dual":
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=True)
        _draw(ax_l, log_y=False)
        _setup_axes(ax_l, "Training Loss (linear)", "global step", "loss")
        _draw(ax_r, log_y=True)
        _setup_axes(ax_r, "Training Loss (log)", "global step", "loss")
    elif style == "split":
        fig, (ax_t, ax_b) = plt.subplots(
            2, 1, figsize=(7.6, 6.0), sharex=True,
            gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.18},
        )
        _draw(ax_t, log_y=False, annotate_min=False, legend=False)
        _setup_axes(ax_t, "Training Loss (full range)", "", "loss")
        _draw(ax_b, log_y=False, ylim=zoom_range, annotate_min=True, legend=False)
        zr = zoom_range or (float(np.nanmin(smoothed)), float(np.nanmax(smoothed)))
        _setup_axes(
            ax_b,
            f"Zoom: {zr[0]:.3f}–{zr[1]:.3f}",
            "global step",
            "loss",
        )
    elif style == "zoom":
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _draw(ax, log_y=False, ylim=zoom_range)
        zr = zoom_range or (float(np.nanmin(smoothed)), float(np.nanmax(smoothed)))
        _setup_axes(
            ax,
            f"Training Loss (zoom {zr[0]:.3f}–{zr[1]:.3f})",
            "global step",
            "loss",
        )
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        log_y = style == "log"
        _draw(ax, log_y=log_y, with_raw=(style == "with_raw"))
        _setup_axes(ax, "Training Loss", "global step", "loss")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    extra = f", zoom={zoom_range}" if zoom_range and style in ("zoom", "split") else ""
    print(f"[plot] wrote {out_path} (style: {used}{extra})")


def plot_lr(rows: list[dict], out_path: Path) -> None:
    x, y = _series(rows, "learning_rate")
    if x.size == 0:
        print("[plot] no 'learning_rate' field found, skipping lr_schedule.png")
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(x, y, color="#107a4d", linewidth=2.0)
    _setup_axes(ax, "Learning Rate Schedule", "global step", "learning rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_grad_norm(rows: list[dict], out_path: Path, tb_dir: Optional[Path]) -> None:
    """Plot gradient norm.

    transformers 4.37 (pinned by the LLaVA fork) does not write ``grad_norm``
    into the HF Trainer log dict on the DeepSpeed-ZeRO3 path, so the
    ``training_metrics.jsonl`` written by ``StepSaveAndMetricsCallback``
    will lack this field. As a fallback, look up the same scalar from the
    TensorBoard event files (DeepSpeed/HF write it there under various tag
    names depending on version).
    """
    x, y = _series(rows, "grad_norm")
    source = "training_metrics.jsonl"
    matched_tag: Optional[str] = None
    if x.size == 0 and tb_dir is not None:
        candidates = [
            "train/grad_norm",
            "grad_norm",
            "Train/grad_norm",
            "deepspeed/grad_norm",
            "step_grad_norm",
            "total_grad_norm",
        ]
        x, y, matched_tag = _read_tb_scalars(tb_dir, candidates)
        if x.size > 0:
            source = f"TensorBoard ({tb_dir}, tag='{matched_tag}')"

    if x.size == 0:
        print(
            "[plot] no 'grad_norm' field found in jsonl and no matching scalar in TB; "
            "skipping grad_norm.png. "
            "(transformers 4.37 + DeepSpeed-ZeRO3 does not log grad_norm into the "
            "HF Trainer log dict; this is expected.)"
        )
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(x, y, color="#9c2a6b", linewidth=1.3, alpha=0.85)
    if (np.nanmax(y) / max(float(np.nanmin(y)), 1e-9)) > 50:
        ax.set_yscale("log")
        ylabel = "grad_norm (log scale)"
    else:
        ylabel = "grad_norm"
    _setup_axes(ax, "Gradient Norm", "global step", ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path} (source: {source})")


def plot_eval_vs_step(eval_csv: Path, out_path: Path) -> None:
    import csv

    xs: list[int] = []
    ys: list[float] = []
    metric_name: Optional[str] = None
    with eval_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            print(f"[plot] {eval_csv} has no header, skipping")
            return
        non_step = [c for c in reader.fieldnames if c.lower() not in ("step", "global_step")]
        if not non_step:
            print(f"[plot] {eval_csv} has no metric column besides 'step'")
            return
        metric_name = non_step[0]
        for row in reader:
            try:
                step_str = row.get("step") or row.get("global_step")
                xs.append(int(float(step_str)))
                ys.append(float(row[metric_name]))
            except (ValueError, TypeError):
                continue
    if not xs:
        print(f"[plot] {eval_csv} produced no usable rows")
        return

    order = np.argsort(xs)
    xs_arr = np.asarray(xs)[order]
    ys_arr = np.asarray(ys)[order]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(xs_arr, ys_arr, marker="o", color="#c7522a", linewidth=2.0)
    best_i = int(np.argmax(ys_arr))
    ax.scatter([xs_arr[best_i]], [ys_arr[best_i]], color="black", zorder=5, s=60)
    ax.annotate(
        f"best @ step {xs_arr[best_i]}: {ys_arr[best_i]:.4f}",
        xy=(xs_arr[best_i], ys_arr[best_i]),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=10,
    )
    _setup_axes(ax, "Eval Metric vs Checkpoint Step", "global step", metric_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", required=True, help="Path to training_metrics.jsonl")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: same dir as --metrics)")
    parser.add_argument(
        "--tb-dir",
        default=None,
        help="TensorBoard event dir (used as fallback for grad_norm). "
        "Default: <metrics-dir>/tb if it exists.",
    )
    parser.add_argument("--eval-results", default=None, help="Optional CSV: columns step,<metric>")
    parser.add_argument(
        "--loss-style",
        default="split",
        choices=["clean", "with_raw", "log", "dual", "zoom", "split"],
        help="loss plot style. 'split' (default): top=full range, bottom=zoom; "
        "best for thesis when you want to show both the early drop and the "
        "convergence-tail detail. 'clean': single smoothed line, linear y. "
        "'with_raw': also overlays raw. 'log': log y with decimal ticks. "
        "'dual': linear+log side by side. 'zoom': single panel clipped to "
        "--loss-zoom-range.",
    )
    parser.add_argument(
        "--loss-zoom-range",
        default=None,
        help="ymin,ymax for the zoom panel (used by 'zoom' and 'split' styles), "
        "e.g. '0.01,0.04'. If not set, auto-derived from the smoothed tail.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    if not metrics_path.is_file():
        raise SystemExit(f"metrics file not found: {metrics_path}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tb_dir:
        tb_dir = Path(args.tb_dir).expanduser().resolve()
    else:
        guess = metrics_path.parent / "tb"
        tb_dir = guess if guess.is_dir() else None

    rows = _read_jsonl(metrics_path)
    print(f"[plot] loaded {len(rows)} rows from {metrics_path}")
    avail = _summarise_numeric_fields(rows)
    if avail:
        print(f"[plot] numeric fields available in jsonl: {', '.join(avail)}")
    if tb_dir:
        print(f"[plot] tb-dir for fallback scalars: {tb_dir}")
    else:
        print("[plot] no tb-dir resolved; grad_norm fallback disabled")

    plot_loss(
        rows,
        out_dir / "loss_curve.png",
        style=args.loss_style,
        zoom_range=_parse_zoom_range(args.loss_zoom_range),
    )
    plot_lr(rows, out_dir / "lr_schedule.png")
    plot_grad_norm(rows, out_dir / "grad_norm.png", tb_dir)

    if args.eval_results:
        eval_csv = Path(args.eval_results).expanduser().resolve()
        if eval_csv.is_file():
            plot_eval_vs_step(eval_csv, out_dir / "eval_metric_vs_step.png")
        else:
            print(f"[plot] eval-results file not found: {eval_csv}")


if __name__ == "__main__":
    main()
