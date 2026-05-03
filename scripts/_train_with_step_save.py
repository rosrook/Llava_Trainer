#!/usr/bin/env python3
"""LLaVA LoRA training entry that augments ``LLaVATrainer`` with three things
the upstream + in-tree trainer modification do NOT provide:

1. ``checkpoint-XXX/config.json`` so each step is fully self-contained for
   ``LlavaLlamaForCausalLM.from_pretrained()`` style export. (The in-tree
   ``LLaVATrainer._save_checkpoint`` already writes ``non_lora_trainables.bin``
   for the LoRA path, so we no longer duplicate that work here.)
2. ``checkpoint-XXX/step_loss.json`` recording the smoothed training loss at
   save time, used downstream for ranking candidates.
3. ``<output_dir>/training_metrics.jsonl`` (per-log metrics) and
   ``<output_dir>/best_step_candidates.json`` (post-train ranking by
   smoothed training loss). The lowest-loss checkpoint is also copied to
   ``<output_dir>/best_by_train_loss/``. Train-loss is a weak proxy for
   downstream task performance: treat the ranking as a heuristic starting
   point and verify the top-K candidates with a real eval (e.g. MMBench
   via ``scripts/common_scripts/eval_mmbench_v15_7b_step_checkpoint.sh``).

Then runs the original ``llava.train.train.train`` with FlashAttention-2,
identical to upstream ``llava/train/train_mem.py``.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from llava.train import train as llava_train_module
from llava.train.llava_trainer import LLaVATrainer


def _is_main_process(args) -> bool:
    return int(getattr(args, "local_rank", -1)) in (-1, 0) and \
        int(os.environ.get("RANK", "0")) == 0


class StepSaveAndMetricsCallback(TrainerCallback):
    """Augment per-checkpoint output with config.json + step_loss.json,
    record per-log training metrics, and rank steps at the end."""

    def __init__(self) -> None:
        self._loss_history: list[tuple[int, float]] = []
        self._metrics_path: Optional[Path] = None
        self._t0: float = time.time()

    # --- helpers ---------------------------------------------------------
    def _ensure_metrics_handle(self, output_dir: str) -> Path:
        if self._metrics_path is None:
            p = Path(output_dir) / "training_metrics.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_path = p
        return self._metrics_path

    def _ckpt_dir(self, args, state) -> Path:
        return Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{int(state.global_step)}"

    # --- callbacks -------------------------------------------------------
    def on_save(self, args, state, control, model=None, **kwargs):
        """Dump config.json + step_loss.json into checkpoint-XXX.
        non_lora_trainables.bin is already written by LLaVATrainer._save_checkpoint
        for the LoRA path, so we do not duplicate that work here."""
        if model is None or not _is_main_process(args):
            return

        ckpt_dir = self._ckpt_dir(args, state)
        if not ckpt_dir.is_dir():
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            model.config.save_pretrained(str(ckpt_dir))
        except Exception as e:
            print(f"[StepSaveCallback] failed to save config.json: {e}", flush=True)

        try:
            smoothed = self._smoothed_loss()
            with (ckpt_dir / "step_loss.json").open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "global_step": int(state.global_step),
                        "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                        "loss_smoothed": smoothed,
                        "elapsed_sec": round(time.time() - self._t0, 2),
                    },
                    fh,
                    ensure_ascii=False,
                )
        except Exception as e:
            print(f"[StepSaveCallback] failed to save step_loss.json: {e}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _is_main_process(args) or not logs:
            return
        loss = logs.get("loss")
        if isinstance(loss, (int, float)) and not math.isnan(float(loss)) and not math.isinf(float(loss)):
            self._loss_history.append((int(state.global_step), float(loss)))

        path = self._ensure_metrics_handle(args.output_dir)
        try:
            payload = {
                "ts": round(time.time() - self._t0, 2),
                "global_step": int(state.global_step),
                "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            }
            for k, v in logs.items():
                if isinstance(v, (int, float, str, bool)):
                    payload[k] = v
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[StepSaveCallback] metrics write warn: {e}", flush=True)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not _is_main_process(args):
            return
        out_dir = Path(args.output_dir)
        candidates = []
        for d in sorted(out_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*")):
            stamp = d / "step_loss.json"
            if not stamp.is_file():
                continue
            try:
                with stamp.open("r", encoding="utf-8") as fh:
                    info = json.load(fh)
                candidates.append(
                    {
                        "step": int(info.get("global_step", -1)),
                        "epoch": float(info.get("epoch", 0.0)),
                        "loss_smoothed": info.get("loss_smoothed"),
                        "path": str(d),
                    }
                )
            except Exception:
                continue

        ranked = sorted(
            (c for c in candidates if isinstance(c.get("loss_smoothed"), (int, float))),
            key=lambda c: float(c["loss_smoothed"]),
        )
        summary = {
            "note": (
                "Heuristic ranking by smoothed training loss (window=10 log "
                "events). Train loss is NOT a reliable proxy for downstream "
                "performance; verify the top candidates with a real eval "
                "(e.g. MMBench via scripts/common_scripts/"
                "eval_mmbench_v15_7b_step_checkpoint.sh)."
            ),
            "best_by_train_loss": ranked[0] if ranked else None,
            "ranking": ranked,
            "all_candidates": candidates,
        }
        try:
            with (out_dir / "best_step_candidates.json").open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            print(f"[StepSaveCallback] wrote best_step_candidates.json", flush=True)
        except Exception as e:
            print(f"[StepSaveCallback] best_step_candidates write warn: {e}", flush=True)

        if ranked:
            best = ranked[0]
            best_dir = out_dir / "best_by_train_loss"
            try:
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(best["path"], best_dir)
                print(
                    f"[StepSaveCallback] copied best-by-train-loss step={best['step']} "
                    f"loss={best['loss_smoothed']:.4f} -> {best_dir}",
                    flush=True,
                )
            except Exception as e:
                print(f"[StepSaveCallback] best copy warn: {e}", flush=True)

    # --- internals -------------------------------------------------------
    def _smoothed_loss(self, window: int = 10) -> Optional[float]:
        if not self._loss_history:
            return None
        recent = self._loss_history[-window:]
        return sum(v for _, v in recent) / len(recent)


# --------------------------------------------------------------------------
# Patch LLaVATrainer.__init__ to auto-register the callback on every rank.
# --------------------------------------------------------------------------
_orig_llava_trainer_init = LLaVATrainer.__init__


def _patched_init(self, *args, **kwargs):  # type: ignore[no-redef]
    _orig_llava_trainer_init(self, *args, **kwargs)
    self.add_callback(StepSaveAndMetricsCallback())


LLaVATrainer.__init__ = _patched_init  # type: ignore[assignment]


if __name__ == "__main__":
    llava_train_module.train(attn_implementation="flash_attention_2")
