#!/usr/bin/env python3
"""Convert a base64-image VQA JSON/JSONL dataset to LLaVA SFT format.

Input records may be a JSON array, a single JSON object with a list field, or
JSONL. The expected fields for the current CAST/VQA-style dataset are:

  image_base64: JPEG/PNG bytes encoded as base64, optionally with data URI prefix
  full_question: multiple-choice question with options already appended
  question: fallback if full_question is absent
  answer / correct_option: target option letter

Output:
  <output_dir>/vqa.json   - LLaVA JSON array
  <output_dir>/vqa.jsonl  - same records as JSONL, useful for inspection
  <output_dir>/images/*.jpg
  <output_dir>/conversion_report.json
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input JSON array / JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--require-validation-passed", action="store_true", default=True)
    parser.add_argument("--no-require-validation-passed", dest="require_validation_passed", action="store_false")
    parser.add_argument("--min-validation-score", type=float, default=None)
    parser.add_argument("--answer-mode", choices=["letter", "letter_text", "explanation"], default="letter")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files/images")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("data", "items", "records", "examples", "annotations"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            return [obj]
    except json.JSONDecodeError:
        pass

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def strip_data_uri(s: str) -> str:
    if "," in s and s.lstrip().lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s


def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(strip_data_uri(b64), validate=False)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img.convert("RGB")


def clean_id_part(x: Any) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:80] or "na"


def get_question(row: dict[str, Any]) -> str | None:
    q = row.get("full_question") or row.get("question")
    if not isinstance(q, str):
        return None
    q = q.strip()
    return q or None


def get_answer(row: dict[str, Any], mode: str) -> str | None:
    letter = row.get("answer") or row.get("correct_option")
    if isinstance(letter, str):
        letter = letter.strip()
    else:
        letter = None

    if mode == "letter":
        return letter or None

    options = row.get("options")
    option_text = None
    if isinstance(options, dict) and letter in options:
        option_text = str(options[letter]).strip()

    if mode == "letter_text":
        if letter and option_text:
            return f"{letter}. {option_text}"
        return letter or option_text

    explanation = row.get("explanation")
    if mode == "explanation" and isinstance(explanation, str) and explanation.strip():
        if letter:
            return f"{letter}\n{explanation.strip()}"
        return explanation.strip()

    return letter or option_text


def make_sample_id(row: dict[str, Any], row_idx: int) -> str:
    parts = [
        row.get("id", "noid"),
        row.get("sample_index", "nosample"),
        row_idx,
    ]
    return "refined_vqa_1_9_" + "_".join(clean_id_part(x) for x in parts)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    img_dir = out_dir / "images"

    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir is not empty: {out_dir} (use --overwrite to reuse)")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)

    skip = Counter()
    converted: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for idx in order:
        row = records[idx]

        if args.require_validation_passed and row.get("validation_passed") is False:
            skip["validation_failed"] += 1
            continue
        if args.min_validation_score is not None:
            score = row.get("validation_score")
            if not isinstance(score, (int, float)) or float(score) < args.min_validation_score:
                skip["low_validation_score"] += 1
                continue

        b64 = row.get("image_base64")
        if not isinstance(b64, str) or not b64.strip():
            skip["missing_image_base64"] += 1
            continue

        question = get_question(row)
        if question is None:
            skip["missing_question"] += 1
            continue

        answer = get_answer(row, args.answer_mode)
        if answer is None or not str(answer).strip():
            skip["missing_answer"] += 1
            continue

        sample_id = make_sample_id(row, idx)
        image_name = f"{sample_id}.{args.image_format}"
        image_path = img_dir / image_name
        try:
            image = decode_image(b64)
            if args.image_format == "jpg":
                image.save(image_path, format="JPEG", quality=args.jpeg_quality)
            else:
                image.save(image_path, format="PNG")
        except (binascii.Error, OSError, ValueError) as exc:
            skip[f"bad_image:{type(exc).__name__}"] += 1
            continue

        llava_row = {
            "id": sample_id,
            "image": image_name,
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": str(answer).strip()},
            ],
        }
        converted.append(llava_row)

        if len(examples) < 5:
            examples.append({
                "id": sample_id,
                "question": question[:500],
                "answer": str(answer).strip(),
                "image": image_name,
                "source_id": row.get("id"),
                "pipeline_name": row.get("pipeline_name"),
                "validation_score": row.get("validation_score"),
            })

        if len(converted) >= args.sample_size:
            break

    if len(converted) < args.sample_size:
        raise SystemExit(
            f"only converted {len(converted)} samples, requested {args.sample_size}; skips={dict(skip)}"
        )

    json_path = out_dir / "vqa.json"
    jsonl_path = out_dir / "vqa.jsonl"
    json_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in converted:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "total_records": len(records),
        "sample_size": args.sample_size,
        "converted": len(converted),
        "seed": args.seed,
        "answer_mode": args.answer_mode,
        "require_validation_passed": args.require_validation_passed,
        "min_validation_score": args.min_validation_score,
        "skips": dict(skip),
        "examples": examples,
    }
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[convert] input_records={len(records)}")
    print(f"[convert] converted={len(converted)}")
    print(f"[convert] skips={dict(skip)}")
    print(f"[convert] wrote {json_path}")
    print(f"[convert] wrote {jsonl_path}")
    print(f"[convert] images={img_dir}")
    print("[convert] sample examples:")
    print(json.dumps(examples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
