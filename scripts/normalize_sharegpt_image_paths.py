#!/usr/bin/env python3
"""Normalize the ``image`` field of a LLaVA SFT JSONL.

LLaVA's LazySupervisedDataset and ``preflight_lora_dataset.py`` both expect
the per-row ``image`` field to be a relative path under ``image_folder``.
Some upstream pipelines (e.g. cast_vqa's ``sft_with_cot`` exporter) emit
ABSOLUTE paths into a content-addressable cache, which the preflight rejects
with ``path_escape``.

This script writes a sibling JSONL where every row's ``image`` field is
reduced to its basename (``Path(image).name``). It also verifies that each
basename exists under ``--image-folder``.

Idempotent: if ``--output`` already exists and is newer than ``--input``,
the script reuses it (still re-runs the existence audit).

Usage:
    python3 scripts/normalize_sharegpt_image_paths.py \\
        --input  /home/.../sft_with_cot_sharegpt.jsonl \\
        --output /tmp/sft_with_cot_sharegpt.normalized.jsonl \\
        --image-folder /mnt/.../direct_5k/images

On success prints ``USE_DATA_PATH=<output>`` on the last line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def iter_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[normalize] JSON parse error at line {ln_no}: {e}")


def normalize(input_path: Path, output_path: Path) -> tuple[int, int]:
    written = 0
    rewritten = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in iter_rows(input_path):
            img = row.get("image")
            if isinstance(img, str) and img:
                base = Path(img).name
                if base != img:
                    rewritten += 1
                row["image"] = base
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written, rewritten


def audit(output_path: Path, image_folder: Path) -> tuple[int, list[str]]:
    folder = image_folder.resolve()
    missing: list[str] = []
    total = 0
    for row in iter_rows(output_path):
        img = row.get("image")
        if not isinstance(img, str) or not img:
            continue
        total += 1
        if not (folder / img).exists():
            if len(missing) < 20:
                missing.append(img)
    return total, missing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Source JSONL with possibly absolute image paths.")
    p.add_argument("--output", required=True, help="Destination JSONL (normalized).")
    p.add_argument("--image-folder", required=True, help="Folder under which normalized basenames must resolve.")
    p.add_argument("--force", action="store_true", help="Always rewrite even if output is newer than input.")
    args = p.parse_args()

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()
    img_dir = Path(args.image_folder).expanduser().resolve()

    if not src.is_file():
        raise SystemExit(f"[normalize] input not found: {src}")
    if not img_dir.is_dir():
        raise SystemExit(f"[normalize] image folder not found: {img_dir}")

    if dst.exists() and not args.force and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"[normalize] reuse existing output: {dst}")
    else:
        n_written, n_rewritten = normalize(src, dst)
        print(f"[normalize] wrote {n_written} rows -> {dst}  ({n_rewritten} image fields rewritten)")

    total, missing = audit(dst, img_dir)
    if missing:
        print("[normalize] missing image files (first 20):")
        for m in missing:
            print(f"  - {m}")
        # The exit code is non-zero so the launcher aborts before training.
        raise SystemExit(
            f"[normalize] FAILED: {len(missing)} of {total} images missing under {img_dir}"
        )

    print(f"[normalize] OK: all {total} images resolve under {img_dir}")
    print(f"USE_DATA_PATH={dst}")


if __name__ == "__main__":
    main()
