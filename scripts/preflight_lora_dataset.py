#!/usr/bin/env python3
"""Preflight + format normalizer for LLaVA LoRA SFT data.

LLaVA's LazySupervisedDataset uses ``json.load(open(data_path))`` so it
requires a JSON ARRAY file. This script:

1. Detects whether ``--data`` is a JSON array (``[...]``) or JSON Lines.
2. If JSONL, converts it to a sibling ``<name>.json`` (re-converts only when
   the source is newer than the sibling) and prints the path to use.
3. Validates each row has ``image`` (str) and ``conversations`` (list of
   >= 2 dicts with ``from`` / ``value``).
4. Verifies every referenced image file exists under ``--images``.

Usage:
    python3 scripts/preflight_lora_dataset.py \
        --data  /path/to/vqa.jsonl \
        --images /path/to/images

On success the script prints ``USE_DATA_PATH=<resolved-json-path>`` on its
last line, which the launcher can capture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def detect_format(path: Path) -> str:
    """Return 'json_array' or 'jsonl' by peeking the first non-whitespace char."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped.startswith("["):
                return "json_array"
            if stripped.startswith("{"):
                return "jsonl"
            raise ValueError(
                f"Unrecognized leading char {stripped[:1]!r} in {path}"
            )
    raise ValueError(f"Empty data file: {path}")


def jsonl_to_array(src: Path, dst: Path) -> int:
    rows = []
    with src.open("r", encoding="utf-8") as fh:
        for ln_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"[preflight] JSONL parse error at line {ln_no}: {e}"
                )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    return len(rows)


def maybe_convert(src: Path) -> Path:
    """Return path to a JSON-array file. Convert from JSONL if needed."""
    fmt = detect_format(src)
    if fmt == "json_array":
        return src
    sibling = src.with_suffix(".json")
    if sibling.exists() and sibling.stat().st_mtime >= src.stat().st_mtime:
        print(f"[preflight] reuse cached array: {sibling}")
        return sibling
    print(f"[preflight] converting JSONL -> JSON array: {src} -> {sibling}")
    n = jsonl_to_array(src, sibling)
    print(f"[preflight] wrote {n} rows to {sibling}")
    return sibling


def load_array(path: Path) -> list:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(
            f"[preflight] expected JSON array at {path}, got {type(data).__name__}"
        )
    return data


def _has_traversal(rel: str) -> bool:
    """Lexical (NOT symlink-following) check that ``rel`` stays inside the
    image folder. Rejects absolute paths and any ``..`` segment after
    normalization. Follows symlinks is the responsibility of ``exists()``,
    NOT this security check, because content-addressable image stores
    routinely use symlinks that point outside the visible folder yet are
    legitimate.
    """
    rel_path = Path(rel)
    if rel_path.is_absolute():
        return True
    parts = Path(os.path.normpath(rel)).parts
    return any(p == ".." for p in parts)


def validate(rows: list, image_folder: Path, sample_limit: int) -> None:
    if not rows:
        raise SystemExit("[preflight] dataset is empty")

    n_total = len(rows)
    n_check = min(n_total, sample_limit)
    bad: list[tuple[int, str]] = []
    missing_files = 0

    image_folder_resolved = image_folder.resolve()

    for i in range(n_check):
        row = rows[i]
        if not isinstance(row, dict):
            bad.append((i, "row_not_dict"))
            continue

        rel = str(row.get("image", "") or "").strip()
        if not rel:
            bad.append((i, "missing_image_field"))
        else:
            if _has_traversal(rel):
                bad.append((i, f"path_escape:{rel}"))
            else:
                # exists() follows symlinks; that is exactly what we want.
                file_path = image_folder_resolved / rel
                if not file_path.exists():
                    missing_files += 1
                    if len(bad) < 20:
                        bad.append((i, f"missing_file:{rel}"))

        conv = row.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            bad.append((i, "bad_conversations"))
            continue
        for j, turn in enumerate(conv[:2]):
            if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
                bad.append((i, f"bad_turn[{j}]"))
                break

    if bad:
        print("[preflight] sample failures (first 20):")
        for idx, reason in bad[:20]:
            print(f"  row {idx}: {reason}")
        raise SystemExit(
            f"[preflight] FAILED: {len(bad)} issues out of {n_check} sampled rows; "
            f"missing_files={missing_files}"
        )

    print(
        f"[preflight] OK: total_rows={n_total} sampled={n_check} "
        f"image_folder={image_folder_resolved}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="LLaVA LoRA dataset preflight (JSON/JSONL aware)."
    )
    p.add_argument("--data", required=True, help="Path to vqa data file (.json or .jsonl)")
    p.add_argument("--images", required=True, help="Image folder root.")
    p.add_argument(
        "--sample-limit",
        type=int,
        default=int(os.environ.get("PREFLIGHT_SAMPLE_LIMIT", "0")) or 10**9,
        help="Validate at most this many rows (default: all).",
    )
    args = p.parse_args()

    data_src = Path(args.data).expanduser().resolve()
    images = Path(args.images).expanduser().resolve()

    if not data_src.exists():
        raise SystemExit(f"[preflight] data file not found: {data_src}")
    if not images.is_dir():
        raise SystemExit(f"[preflight] images folder not found: {images}")

    data_resolved = maybe_convert(data_src)
    rows = load_array(data_resolved)
    validate(rows, images, args.sample_limit)

    print(f"USE_DATA_PATH={data_resolved}")


if __name__ == "__main__":
    main()
