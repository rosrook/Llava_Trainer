#!/usr/bin/env python3
"""Convert a CAST_VQA SFTPacker(sharegpt, image_mode=path_only) dump into the
LLaVA SFT layout that ``scripts/preflight_lora_dataset.py`` and
``scripts/_train_with_step_save.py`` expect.

Input  : sft_sharegpt.jsonl  (one JSON object per line)
   Each row was produced by cast_vqa.sft.formatter.format_sharegpt and
   stream2.py's SFTPacker(image_mode="path_only"), so it looks like:
     {
       "id": "<uuid>",
       "image": "<absolute server path to source image>",
       "conversations": [
         {"from": "human", "value": "<image>\n<MCQ question + options>"},
         {"from": "gpt",   "value": "<letter>. <option text>"}
       ],
       "metadata": {...}
     }

Output : <output-dir>/{vqa.jsonl, vqa.json, images/, conversion_report.json}
   * vqa.jsonl/vqa.json: rows with ``image`` rewritten to a basename
     relative to <output-dir>/images.
   * images/: every referenced source image is symlinked (default) or
     copied here. Filename collisions are disambiguated with an md5 prefix
     of the original absolute path so that two distinct sources never
     overwrite each other.
   * conversion_report.json: counts + first few examples for inspection.

The training launcher (finetune_lora_cast_vqa_*_safe_2gpu.sh) runs this
script before calling DeepSpeed; you can also run it standalone:

    python3 scripts/prepare_cast_vqa_sft.py \\
        --input  /path/to/sft_sharegpt.jsonl \\
        --output-dir /path/to/cast_vqa_xxx \\
        --image-mode symlink

Why a separate script (rather than feeding sft_sharegpt.jsonl directly)?
  preflight_lora_dataset.py rejects rows whose ``image`` field is absolute
  (``_has_traversal`` treats is_absolute() as a path-escape).  CAST_VQA
  emits absolute paths because its packer ran in image_mode="path_only".
  This script bridges the two conventions without modifying either side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="sft_sharegpt.jsonl path")
    p.add_argument("--output-dir", required=True, help="Target dir for vqa.jsonl + images/")
    p.add_argument(
        "--image-mode", choices=["symlink", "copy"], default="symlink",
        help="How to materialise images under <output-dir>/images "
             "(default: symlink, fastest and uses no disk).",
    )
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="Optional cap on number of converted rows (0 = no cap).",
    )
    p.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle rows before applying --max-samples. Without this flag "
             "the script takes the first N rows in file order; CAST_VQA "
             "writes records in curriculum order (tier 0 -> 1 -> 2), so "
             "head-N would bias the training set towards easy samples.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for --shuffle (default 42, matches direct_5k recipe).",
    )
    p.add_argument(
        "--keep-metadata", action="store_true",
        help="Keep the ``metadata`` field on each row (default: drop it; "
             "LLaVA's loader ignores it but it bloats vqa.json).",
    )
    p.add_argument(
        "--allow-missing-images", action="store_true",
        help="Skip rows whose source image does not exist instead of failing.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Allow output dir to already contain vqa.jsonl / images/.",
    )
    return p.parse_args()


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"[prepare] JSONL parse error at line {line_no}: {exc}"
                )
            if isinstance(obj, dict):
                yield obj


def stable_basename(src: Path, taken: dict[str, Path]) -> str:
    """Return a basename for ``src`` under images/ that does not collide.

    Two distinct absolute paths with the same basename get disambiguated
    by an 8-char md5 prefix of the full path.
    """
    name = src.name or "image"
    existing = taken.get(name)
    if existing is None:
        taken[name] = src.resolve()
        return name
    if existing == src.resolve():
        return name  # same file referenced twice
    digest = hashlib.md5(str(src.resolve()).encode("utf-8")).hexdigest()[:8]
    suffix = src.suffix or ".jpg"
    new_name = f"{src.stem}_{digest}{suffix}"
    taken[new_name] = src.resolve()
    return new_name


def materialise(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    src_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    img_dir = out_dir / "images"

    if not src_path.is_file():
        raise SystemExit(f"[prepare] input not found: {src_path}")

    vqa_jsonl = out_dir / "vqa.jsonl"
    vqa_json = out_dir / "vqa.json"
    if (vqa_jsonl.exists() or vqa_json.exists()) and not args.overwrite:
        raise SystemExit(
            f"[prepare] {out_dir} already has vqa.jsonl / vqa.json; "
            f"pass --overwrite to reuse."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    converted: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    skip = Counter()
    taken: dict[str, Path] = {}

    if args.shuffle:
        all_rows = list(iter_rows(src_path))
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)
        row_iter: Iterable[dict[str, Any]] = all_rows
        print(
            f"[prepare] loaded {len(all_rows)} rows; shuffled with seed={args.seed} "
            f"(cap={args.max_samples or 'none'})"
        )
    else:
        row_iter = iter_rows(src_path)
        if args.max_samples:
            print(
                f"[prepare] WARNING: --max-samples without --shuffle takes the first "
                f"{args.max_samples} rows in file order. CAST_VQA writes curriculum "
                f"order (easy first), so consider passing --shuffle."
            )

    for idx, row in enumerate(row_iter):
        if args.max_samples and len(converted) >= args.max_samples:
            break

        raw_image = row.get("image")
        if not isinstance(raw_image, str) or not raw_image.strip():
            skip["missing_image_field"] += 1
            continue

        src_img = Path(raw_image).expanduser()
        if not src_img.is_absolute():
            src_img = (src_path.parent / src_img).resolve()
        if not src_img.is_file():
            skip["source_image_missing"] += 1
            if not args.allow_missing_images:
                if skip["source_image_missing"] <= 5:
                    print(f"[prepare] WARN missing image (row {idx}): {src_img}")
                continue
            continue

        conv = row.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            skip["bad_conversations"] += 1
            continue
        bad_turn = False
        for turn in conv[:2]:
            if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
                bad_turn = True
                break
        if bad_turn:
            skip["bad_turn"] += 1
            continue

        basename = stable_basename(src_img, taken)
        materialise(src_img, img_dir / basename, args.image_mode)

        new_row = {
            "id": str(row.get("id") or f"cast_vqa_{idx}"),
            "image": basename,
            "conversations": conv,
        }
        if args.keep_metadata and "metadata" in row:
            new_row["metadata"] = row["metadata"]
        converted.append(new_row)

        if len(examples) < 5:
            examples.append({
                "id": new_row["id"],
                "image": basename,
                "human": conv[0].get("value", "")[:300],
                "gpt": conv[1].get("value", "")[:200],
            })

    if not converted:
        raise SystemExit(
            f"[prepare] zero rows converted from {src_path}; skips={dict(skip)}"
        )

    with vqa_jsonl.open("w", encoding="utf-8") as fh:
        for r in converted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with vqa_json.open("w", encoding="utf-8") as fh:
        json.dump(converted, fh, ensure_ascii=False)

    report = {
        "input": str(src_path),
        "output_dir": str(out_dir),
        "image_mode": args.image_mode,
        "max_samples": args.max_samples or None,
        "shuffle": args.shuffle,
        "seed": args.seed if args.shuffle else None,
        "converted": len(converted),
        "unique_images": len({p.resolve() for p in taken.values()}),
        "skips": dict(skip),
        "examples": examples,
    }
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"[prepare] input          : {src_path}")
    print(f"[prepare] output_dir     : {out_dir}")
    print(f"[prepare] vqa.jsonl rows : {len(converted)}")
    print(f"[prepare] images dir     : {img_dir}  ({report['unique_images']} unique source images)")
    print(f"[prepare] image_mode     : {args.image_mode}")
    print(f"[prepare] skips          : {dict(skip)}")


if __name__ == "__main__":
    main()
