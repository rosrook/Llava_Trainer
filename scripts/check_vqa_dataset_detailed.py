#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
WHITESPACE_ONLY_RE = re.compile(r"^\s*$")


@dataclass
class Issue:
    idx: int
    code: str
    severity: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _resolve_image_path(image_folder: Path, image_rel: str) -> Path:
    return (image_folder / image_rel).resolve()


def _is_path_within(base: Path, target: Path) -> bool:
    try:
        target.relative_to(base.resolve())
        return True
    except Exception:
        return False


def _basic_text_stats(values: List[str]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    lens = [len(v) for v in values]
    return {
        "count": len(values),
        "min": min(lens),
        "max": max(lens),
        "mean": round(statistics.mean(lens), 2),
        "median": round(statistics.median(lens), 2),
        "p95": round(sorted(lens)[max(0, int(len(lens) * 0.95) - 1)], 2),
    }


def validate_dataset(
    data: Any,
    image_folder: Path,
    strict_role_order: bool,
    strict_two_turns: bool,
    check_image_decode: bool,
    sample_decode_limit: int,
) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    summary: Dict[str, Any] = {}

    if not isinstance(data, list):
        issues.append(Issue(-1, "dataset_not_list", "error", "Top-level JSON must be a list."))
        return summary, issues

    id_counter = Counter()
    image_rel_counter = Counter()
    role_counter = Counter()
    image_ext_counter = Counter()
    conversation_len_counter = Counter()
    question_texts: List[str] = []
    answer_texts: List[str] = []
    image_size_counter = Counter()
    duplicate_signature = Counter()

    decoded_count = 0
    decode_skipped = 0

    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            issues.append(Issue(idx, "row_not_dict", "error", "Row is not a JSON object."))
            continue

        row_id = _safe_get(row, "id", "")
        image_rel = str(_safe_get(row, "image", "") or "").strip()
        conv = _safe_get(row, "conversations", None)

        if not row_id:
            issues.append(Issue(idx, "missing_id", "warning", "Row id is missing or empty."))
        else:
            id_counter[str(row_id)] += 1

        if not image_rel:
            issues.append(Issue(idx, "missing_image", "error", "Image field is missing or empty."))
        else:
            image_rel_counter[image_rel] += 1
            ext = Path(image_rel).suffix.lower() or "<none>"
            image_ext_counter[ext] += 1

            abs_path = _resolve_image_path(image_folder, image_rel)
            if not _is_path_within(image_folder, abs_path):
                issues.append(
                    Issue(
                        idx,
                        "image_path_traversal",
                        "error",
                        "Image path escapes image_folder.",
                        {"image": image_rel, "resolved": str(abs_path)},
                    )
                )
            elif not abs_path.exists():
                issues.append(
                    Issue(
                        idx,
                        "image_not_found",
                        "error",
                        "Image file does not exist.",
                        {"image": image_rel, "resolved": str(abs_path)},
                    )
                )
            elif not abs_path.is_file():
                issues.append(
                    Issue(
                        idx,
                        "image_not_file",
                        "error",
                        "Image path exists but is not a file.",
                        {"image": image_rel, "resolved": str(abs_path)},
                    )
                )
            else:
                # Light-weight decode check to catch corrupted images.
                if check_image_decode and Image is not None:
                    if sample_decode_limit <= 0 or decoded_count < sample_decode_limit:
                        try:
                            with Image.open(abs_path) as im:
                                im.verify()
                            with Image.open(abs_path) as im2:
                                w, h = im2.size
                                mode = im2.mode
                            image_size_counter[f"{w}x{h}"] += 1
                            if w <= 0 or h <= 0:
                                issues.append(
                                    Issue(idx, "image_invalid_size", "error", "Image width/height is invalid.", {"size": [w, h]})
                                )
                            if mode not in {"RGB", "RGBA", "L", "P", "CMYK"}:
                                issues.append(
                                    Issue(
                                        idx,
                                        "image_unusual_mode",
                                        "warning",
                                        "Image mode is unusual.",
                                        {"mode": mode, "image": image_rel},
                                    )
                                )
                        except Exception as e:
                            issues.append(
                                Issue(
                                    idx,
                                    "image_decode_failed",
                                    "error",
                                    "Image decode/verify failed.",
                                    {"image": image_rel, "error": str(e)},
                                )
                            )
                        decoded_count += 1
                    else:
                        decode_skipped += 1

        if not isinstance(conv, list):
            issues.append(Issue(idx, "conversations_not_list", "error", "conversations is not a list."))
            continue

        conversation_len_counter[len(conv)] += 1
        if strict_two_turns and len(conv) != 2:
            issues.append(
                Issue(
                    idx,
                    "conversations_len_not_2",
                    "warning",
                    "conversations length is not 2 in strict mode.",
                    {"len": len(conv)},
                )
            )
        if len(conv) < 2:
            issues.append(Issue(idx, "conversations_too_short", "error", "conversations must contain at least 2 turns."))
            continue

        for turn_i, turn in enumerate(conv):
            if not isinstance(turn, dict):
                issues.append(
                    Issue(
                        idx,
                        "turn_not_dict",
                        "error",
                        "Conversation turn is not an object.",
                        {"turn_index": turn_i},
                    )
                )
                continue
            frm = str(_safe_get(turn, "from", "") or "").strip()
            val = _safe_get(turn, "value", None)
            role_counter[frm or "<empty>"] += 1

            if not frm:
                issues.append(Issue(idx, "turn_missing_from", "error", "Turn is missing `from`.", {"turn_index": turn_i}))
            if not isinstance(val, str):
                issues.append(
                    Issue(
                        idx,
                        "turn_value_not_string",
                        "error",
                        "Turn value is not a string.",
                        {"turn_index": turn_i, "type": type(val).__name__},
                    )
                )
                continue
            if WHITESPACE_ONLY_RE.match(val or ""):
                issues.append(Issue(idx, "turn_value_empty", "error", "Turn value is empty/whitespace.", {"turn_index": turn_i}))
            if CONTROL_CHARS_RE.search(val):
                issues.append(
                    Issue(
                        idx,
                        "turn_contains_control_chars",
                        "warning",
                        "Turn value contains control characters.",
                        {"turn_index": turn_i},
                    )
                )
            if len(val) > 10000:
                issues.append(
                    Issue(
                        idx,
                        "turn_value_very_long",
                        "warning",
                        "Turn value is unusually long (>10k chars).",
                        {"turn_index": turn_i, "length": len(val)},
                    )
                )

        first_from = str(_safe_get(conv[0], "from", "") or "").strip()
        second_from = str(_safe_get(conv[1], "from", "") or "").strip()
        first_val = str(_safe_get(conv[0], "value", "") or "")
        second_val = str(_safe_get(conv[1], "value", "") or "")

        if strict_role_order and (first_from != "human" or second_from != "gpt"):
            issues.append(
                Issue(
                    idx,
                    "role_order_unexpected",
                    "warning",
                    "Expected first two turns to be human -> gpt in strict mode.",
                    {"first_from": first_from, "second_from": second_from},
                )
            )

        if "<image>" not in first_val:
            issues.append(
                Issue(
                    idx,
                    "question_missing_image_token",
                    "warning",
                    "First turn does not contain <image> token.",
                )
            )

        question_texts.append(first_val)
        answer_texts.append(second_val)

        sig = _sha1_text(f"{image_rel}||{first_val.strip()}||{second_val.strip()}")
        duplicate_signature[sig] += 1

    # Duplicate checks
    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}
    if duplicate_ids:
        issues.append(
            Issue(
                -1,
                "duplicate_ids",
                "warning",
                f"Found duplicated ids: {len(duplicate_ids)}.",
                {"top_examples": list(sorted(duplicate_ids.items(), key=lambda x: -x[1]))[:20]},
            )
        )
    exact_dups = sum(1 for v in duplicate_signature.values() if v > 1)
    if exact_dups > 0:
        issues.append(
            Issue(
                -1,
                "duplicate_triplets",
                "warning",
                "Found duplicated (image, question, answer) triplets.",
                {"duplicate_triplet_groups": exact_dups},
            )
        )

    severity_counter = Counter(i.severity for i in issues)
    code_counter = Counter(i.code for i in issues)

    summary.update(
        {
            "num_rows": len(data),
            "num_issues": len(issues),
            "issues_by_severity": dict(severity_counter),
            "issues_by_code_top50": dict(code_counter.most_common(50)),
            "role_distribution": dict(role_counter),
            "conversation_length_distribution": dict(conversation_len_counter),
            "image_extension_distribution": dict(image_ext_counter),
            "question_length_stats": _basic_text_stats(question_texts),
            "answer_length_stats": _basic_text_stats(answer_texts),
            "num_unique_ids": len(id_counter),
            "num_duplicate_ids": len(duplicate_ids),
            "num_unique_image_paths": len(image_rel_counter),
            "num_duplicate_triplet_groups": exact_dups,
            "image_decode_checked": decoded_count,
            "image_decode_skipped": decode_skipped,
            "image_size_top20": dict(image_size_counter.most_common(20)),
        }
    )
    return summary, issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detailed validator for LLaVA-style vqa_data.json."
    )
    parser.add_argument("--json-path", required=True, help="Path to vqa_data.json")
    parser.add_argument("--image-folder", required=True, help="Base image folder used by training")
    parser.add_argument(
        "--output-report",
        default="",
        help="Output report JSON path (default: <json>.validation_report.json)",
    )
    parser.add_argument(
        "--output-issues-jsonl",
        default="",
        help="Optional output issues JSONL path (one issue per line).",
    )
    parser.add_argument(
        "--strict-role-order",
        action="store_true",
        default=True,
        help="Require first two turns to be human->gpt (default enabled).",
    )
    parser.add_argument(
        "--no-strict-role-order",
        action="store_true",
        help="Disable strict role order check.",
    )
    parser.add_argument(
        "--strict-two-turns",
        action="store_true",
        default=False,
        help="Warn if conversations length != 2.",
    )
    parser.add_argument(
        "--check-image-decode",
        action="store_true",
        default=True,
        help="Open and verify images with PIL (default enabled when PIL is installed).",
    )
    parser.add_argument(
        "--no-check-image-decode",
        action="store_true",
        help="Disable image decode verification.",
    )
    parser.add_argument(
        "--sample-decode-limit",
        type=int,
        default=0,
        help="Only decode first N images for speed; 0 means check all.",
    )
    parser.add_argument(
        "--max-print-issues",
        type=int,
        default=30,
        help="Max issues to print to stdout.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    image_folder = Path(args.image_folder).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"json-path not found: {json_path}")
    if not image_folder.exists():
        raise FileNotFoundError(f"image-folder not found: {image_folder}")

    strict_role_order = bool(args.strict_role_order) and (not bool(args.no_strict_role_order))
    check_image_decode = bool(args.check_image_decode) and (not bool(args.no_check_image_decode))
    if Image is None:
        check_image_decode = False

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary, issues = validate_dataset(
        data=data,
        image_folder=image_folder,
        strict_role_order=strict_role_order,
        strict_two_turns=bool(args.strict_two_turns),
        check_image_decode=check_image_decode,
        sample_decode_limit=int(args.sample_decode_limit),
    )

    report_path = (
        Path(args.output_report).expanduser().resolve()
        if args.output_report
        else json_path.with_suffix(".validation_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_obj = {
        "json_path": str(json_path),
        "image_folder": str(image_folder),
        "config": {
            "strict_role_order": strict_role_order,
            "strict_two_turns": bool(args.strict_two_turns),
            "check_image_decode": check_image_decode,
            "sample_decode_limit": int(args.sample_decode_limit),
        },
        "summary": summary,
    }
    with report_path.open("w", encoding="utf-8") as wf:
        json.dump(report_obj, wf, ensure_ascii=False, indent=2)

    if args.output_issues_jsonl:
        issues_path = Path(args.output_issues_jsonl).expanduser().resolve()
        issues_path.parent.mkdir(parents=True, exist_ok=True)
        with issues_path.open("w", encoding="utf-8") as wf:
            for i in issues:
                wf.write(
                    json.dumps(
                        {
                            "idx": i.idx,
                            "code": i.code,
                            "severity": i.severity,
                            "message": i.message,
                            "extra": i.extra,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        issues_path = None

    print(f"[check] dataset: {json_path}")
    print(f"[check] image_folder: {image_folder}")
    print(f"[check] report: {report_path}")
    if issues_path:
        print(f"[check] issues_jsonl: {issues_path}")
    print(f"[check] num_rows={summary.get('num_rows')} num_issues={summary.get('num_issues')}")
    print(f"[check] issues_by_severity={summary.get('issues_by_severity')}")
    print(f"[check] role_distribution={summary.get('role_distribution')}")
    print(f"[check] image_decode_checked={summary.get('image_decode_checked')} skipped={summary.get('image_decode_skipped')}")

    max_print = max(0, int(args.max_print_issues))
    for i, issue in enumerate(issues[:max_print], start=1):
        print(
            f"[issue {i}] idx={issue.idx} severity={issue.severity} code={issue.code} "
            f"msg={issue.message} extra={issue.extra}"
        )
    if len(issues) > max_print:
        print(f"[check] ... {len(issues) - max_print} more issues not printed.")


if __name__ == "__main__":
    main()
