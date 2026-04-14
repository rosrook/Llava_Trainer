#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Set


def load_target_ids(trace_glob: str, step: int) -> Set[str]:
    ids: Set[str] = set()
    files = sorted(glob.glob(trace_glob))
    if not files:
        raise FileNotFoundError(f"No trace files matched: {trace_glob}")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                if int(row.get("global_step", -1)) != step:
                    continue
                for sample in row.get("samples", []) or []:
                    sid = str(sample.get("id") or "").strip()
                    if sid:
                        ids.add(sid)
    if not ids:
        raise RuntimeError(f"No samples found for step={step} in trace files.")
    return ids


def build_subset(data_path: Path, target_ids: Set[str]) -> List[Dict]:
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Input dataset must be a JSON list.")

    subset = [row for row in data if isinstance(row, dict) and str(row.get("id") or "").strip() in target_ids]
    if not subset:
        raise RuntimeError("No matching rows found in dataset for extracted ids.")
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small probe subset from sample trace by step.")
    parser.add_argument("--trace-glob", required=True, help="Glob for sample trace files, e.g. .../sample_trace.rank*.jsonl")
    parser.add_argument("--step", type=int, required=True, help="Global step to extract.")
    parser.add_argument("--data-path", required=True, help="Path to full vqa_data.json")
    parser.add_argument("--output-json", required=True, help="Output subset json path")
    parser.add_argument("--output-ids", default="", help="Optional output txt for extracted ids")
    args = parser.parse_args()

    target_ids = load_target_ids(trace_glob=args.trace_glob, step=int(args.step))
    subset = build_subset(Path(args.data_path), target_ids)

    out_json = Path(args.output_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    if args.output_ids:
        out_ids = Path(args.output_ids).expanduser().resolve()
        out_ids.parent.mkdir(parents=True, exist_ok=True)
        with out_ids.open("w", encoding="utf-8") as f:
            for sid in sorted(target_ids):
                f.write(sid + "\n")

    print(f"[probe-subset] step={args.step}")
    print(f"[probe-subset] ids={len(target_ids)}")
    print(f"[probe-subset] rows={len(subset)}")
    print(f"[probe-subset] output_json={out_json}")
    if args.output_ids:
        print(f"[probe-subset] output_ids={Path(args.output_ids).expanduser().resolve()}")


if __name__ == "__main__":
    main()
