#!/usr/bin/env python3
"""LLaVA LoRA training entry that fixes "intermediate checkpoint cannot be
loaded standalone" by writing ``non_lora_trainables.bin`` + ``config.json``
into every ``checkpoint-XXX/`` directory at save time.

Stock LLaVA only writes ``non_lora_trainables.bin`` once, at end-of-training,
to the top-level ``output_dir`` (see ``llava/train/train.py:safe_save_model``
and the LoRA branch right above ``__main__``).  As a result, intermediate
checkpoint directories miss the non-LoRA trainables (mm_projector,
embed_tokens when use_im_start_end, ...) and ``LlavaLlamaForCausalLM`` cannot
reconstruct the full model from a single ``checkpoint-XXX/`` alone.

This wrapper:
  * Patches ``LLaVATrainer.__init__`` to register a ``TrainerCallback`` that
    on every ``on_save`` event collects non-LoRA trainables (ZeRO-3 aware
    via LLaVA's ``maybe_zero_3``) and dumps them along with ``config.json``
    into the just-created ``checkpoint-XXX/`` folder.
  * Records per-log training metrics into
    ``<output_dir>/training_metrics.jsonl``.
  * On train-end ranks all surviving ``checkpoint-*`` directories by
    smoothed training loss (window=10 logs) and writes
    ``<output_dir>/best_step_candidates.json``. Train-loss alone is a weak
    proxy for "best", so this is a HEURISTIC starting point: confirm with a
    real downstream eval (e.g. MMBench) before picking.

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

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from llava.train import train as llava_train_module
from llava.train.llava_trainer import LLaVATrainer


def _is_main_process(args) -> bool:
    return int(getattr(args, "local_rank", -1)) in (-1, 0) and \
        int(os.environ.get("RANK", "0")) == 0


class StepSaveAndMetricsCallback(TrainerCallback):
    """Make every checkpoint-XXX self-contained and track per-step metrics."""

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
        """Dump non_lora_trainables.bin + config.json into checkpoint-XXX."""
        if model is None:
            return
        if not bool(getattr(args, "lora_enable", False)):
            return

        ckpt_dir = self._ckpt_dir(args, state)
        # Trainer creates the dir before calling on_save, but we double-check.
        if not ckpt_dir.is_dir() and _is_main_process(args):
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ZeRO-3 aware gather; cpu().clone() inside maybe_zero_3.
        non_lora_state = llava_train_module.get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )

        if _is_main_process(args):
            try:
                torch.save(non_lora_state, str(ckpt_dir / "non_lora_trainables.bin"))
            except Exception as e:
                print(f"[StepSaveCallback] failed to save non_lora_trainables: {e}", flush=True)
            try:
                model.config.save_pretrained(str(ckpt_dir))
            except Exception as e:
                print(f"[StepSaveCallback] failed to save config.json: {e}", flush=True)

            # Stamp the smoothed loss for later best-step ranking.
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
