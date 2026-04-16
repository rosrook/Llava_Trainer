#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def _load_non_lora_trainables(model_dir: Path, shared_dir: Path | None) -> dict:
    candidates = [model_dir / "non_lora_trainables.bin"]
    if shared_dir is not None:
        candidates.append(shared_dir / "non_lora_trainables.bin")

    for path in candidates:
        if path.exists():
            state = torch.load(path, map_location="cpu")
            state = {(k[11:] if k.startswith("base_model.") else k): v for k, v in state.items()}
            if any(k.startswith("model.model.") for k in state):
                state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
            return state
    raise FileNotFoundError(
        "Could not find non_lora_trainables.bin in model_dir or shared_dir. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def export_checkpoint(model_path: Path, model_base: str, save_model_path: Path, shared_dir: Path | None) -> None:
    cfg = AutoConfig.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    print("Loading base LLaVA model...", flush=True)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_base,
        low_cpu_mem_usage=True,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading non-LoRA trainables...", flush=True)
    non_lora_trainables = _load_non_lora_trainables(model_path, shared_dir)
    missing, unexpected = model.load_state_dict(non_lora_trainables, strict=False)
    print(f"Loaded non-LoRA weights. missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(model, str(model_path))

    print("Merging LoRA weights...", flush=True)
    model = model.merge_and_unload()

    print(f"Saving merged HF model to: {save_model_path}", flush=True)
    save_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_model_path), safe_serialization=True)
    tokenizer.save_pretrained(str(save_model_path))

    # Try to save image processor if vision tower can be materialized.
    try:
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            vision_tower.load_model(device_map="cpu")
        if vision_tower is not None and getattr(vision_tower, "image_processor", None) is not None:
            vision_tower.image_processor.save_pretrained(str(save_model_path))
    except Exception as e:
        print(f"Warning: failed to save image processor: {e}", flush=True)

    print("Done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a LLaVA LoRA checkpoint directory to merged HF format "
        "without requiring mm_projector.bin."
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint/adapter directory containing config + adapter files.")
    parser.add_argument("--model-base", required=True, help="Base model path or HF repo id, e.g. liuhaotian/llava-v1.5-7b")
    parser.add_argument("--save-model-path", required=True, help="Output merged HF directory.")
    parser.add_argument(
        "--shared-dir",
        default="",
        help="Optional outer directory containing shared files like non_lora_trainables.bin.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    save_model_path = Path(args.save_model_path).expanduser().resolve()
    shared_dir = Path(args.shared_dir).expanduser().resolve() if args.shared_dir else None

    export_checkpoint(model_path=model_path, model_base=args.model_base, save_model_path=save_model_path, shared_dir=shared_dir)


if __name__ == "__main__":
    main()
