import argparse
import os

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    os.makedirs(args.save_model_path, exist_ok=True)

    # Save a standalone HF directory (config + weights + tokenizer + processor).
    # Using safe_serialization when supported (safetensors).
    model.save_pretrained(args.save_model_path, safe_serialization=True)
    tokenizer.save_pretrained(args.save_model_path)
    if image_processor is not None:
        image_processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
