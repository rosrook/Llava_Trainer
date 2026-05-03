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


def _setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="both", labelsize=10)


def plot_loss(rows: list[dict], out_path: Path) -> None:
    x, y = _series(rows, "loss")
    if x.size == 0:
        print("[plot] no 'loss' field found, skipping loss_curve.png")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, y, color="#9aa9c4", linewidth=1.0, alpha=0.7, label="train loss (raw)")
    ax.plot(x, _ema(y, alpha=0.08), color="#1f4e8c", linewidth=2.0, label="EMA(α=0.08)")
    _setup_axes(ax, "Training Loss", "global step", "loss")
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


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


def plot_grad_norm(rows: list[dict], out_path: Path) -> None:
    x, y = _series(rows, "grad_norm")
    if x.size == 0:
        print("[plot] no 'grad_norm' field found, skipping grad_norm.png")
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(x, y, color="#9c2a6b", linewidth=1.4, alpha=0.85)
    if (y.max() / max(y.min(), 1e-9)) > 50:
        ax.set_yscale("log")
        ylabel = "grad_norm (log scale)"
    else:
        ylabel = "grad_norm"
    _setup_axes(ax, "Gradient Norm", "global step", ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


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
    parser.add_argument("--eval-results", default=None, help="Optional CSV: columns step,<metric>")
    args = parser.parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    if not metrics_path.is_file():
        raise SystemExit(f"metrics file not found: {metrics_path}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else metrics_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(metrics_path)
    print(f"[plot] loaded {len(rows)} rows from {metrics_path}")
    plot_loss(rows, out_dir / "loss_curve.png")
    plot_lr(rows, out_dir / "lr_schedule.png")
    plot_grad_norm(rows, out_dir / "grad_norm.png")

    if args.eval_results:
        eval_csv = Path(args.eval_results).expanduser().resolve()
        if eval_csv.is_file():
            plot_eval_vs_step(eval_csv, out_dir / "eval_metric_vs_step.png")
        else:
            print(f"[plot] eval-results file not found: {eval_csv}")


if __name__ == "__main__":
    main()
