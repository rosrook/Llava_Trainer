#!/usr/bin/env python3
import argparse
import os
import torch
import transformers
from peft import PeftModel

from llava.model import LlavaLlamaForCausalLM, LlavaMptForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a LoRA step checkpoint (checkpoint-XXX) to standalone Hugging Face model."
    )
    parser.add_argument("--base_model", required=True, help="Base model path/id, e.g. liuhaotian/llava-v1.5-7b")
    parser.add_argument("--checkpoint_dir", required=True, help="LoRA step checkpoint dir, e.g. .../checkpoint-600")
    parser.add_argument("--output_dir", required=True, help="Output dir for standalone HF model")
    parser.add_argument(
        "--non_lora_path",
        default="",
        help="Path to non_lora_trainables.bin. Default: checkpoint_dir/non_lora_trainables.bin if exists.",
    )
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", help="Load base model with bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    os.makedirs(args.output_dir, exist_ok=True)

    if "mpt" in args.base_model.lower():
        model = LlavaMptForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    non_lora_path = args.non_lora_path.strip() or os.path.join(args.checkpoint_dir, "non_lora_trainables.bin")
    if os.path.isfile(non_lora_path):
        state = torch.load(non_lora_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded non-LoRA trainables: {non_lora_path}")
    else:
        print(f"non_lora_trainables.bin not found, skip: {non_lora_path}")

    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Standalone HF model exported to: {args.output_dir}")


if __name__ == "__main__":
    main()

