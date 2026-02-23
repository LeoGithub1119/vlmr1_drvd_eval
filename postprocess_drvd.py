#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process DrVD model outputs to match DrVD-Bench metric scripts.

- For single-choice tasks: model_response must be "A"/"B"/"C"/"D"
- For joint-choice tasks: model_response must be "A,B,C,D"
- For report generation: keep full text
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
CHOICE_DOT_RE = re.compile(r"\b([ABCD])\s*[\.\)]", re.IGNORECASE)
JOINT_RE = re.compile(r"([ABCD])", re.IGNORECASE)

CAND_KEYS = [
    "model_response",
    "model_answer",
    "pred",
    "prediction",
    "response",
    "output",
    "text",
    "decoded",
    "generation",
    "answer_pred",
]

def _get_pred_text(item: Dict[str, Any]) -> str:
    for k in CAND_KEYS:
        v = item.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 有些人會把 raw 放在 nested
    for k in ["raw", "raw_response", "model_output", "completion"]:
        v = item.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _normalize_single(pred_text: str) -> Optional[str]:
    if not pred_text:
        return None

    # 先抓 "A." / "B)" 這種最常見
    m = CHOICE_DOT_RE.search(pred_text)
    if m:
        return m.group(1).upper()

    # 再抓獨立字母
    m = CHOICE_RE.search(pred_text)
    if m:
        return m.group(1).upper()

    # 再退一步：有時候模型回 "C. CT" 沒空白
    m = re.search(r"^([ABCD])", pred_text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None

def _normalize_joint(pred_text: str) -> Optional[str]:
    if not pred_text:
        return None

    # 抓出所有 A/B/C/D，保持順序
    letters = [x.upper() for x in JOINT_RE.findall(pred_text)]
    if len(letters) < 4:
        # 有些輸出是 "A, B, C, D" 或 "A B C D"；上面也會抓到
        return None

    # 只取前 4 個，避免模型多講廢話
    letters = letters[:4]
    return ",".join(letters)

def _task_kind(task_name: str) -> str:
    # 你的 driver 用的 task 名稱：independent_qa / joint_qa / visual_evidence_qa / report_generation
    t = task_name.lower()
    if "joint" in t:
        return "joint"
    if "report" in t:
        return "report"
    return "single"

def process_file(in_path: Path, out_path: Path, task: str, keep_raw: bool = True) -> Tuple[int, int]:
    kind = _task_kind(task)
    ok = 0
    bad = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            pred_text = _get_pred_text(item)

            if kind == "single":
                norm = _normalize_single(pred_text)
                if norm is None:
                    bad += 1
                    # 仍然寫出，讓你 debug
                    item["model_response"] = ""
                    item["_postprocess_error"] = "cannot_parse_single_choice"
                else:
                    ok += 1
                    item["model_response"] = norm

            elif kind == "joint":
                norm = _normalize_joint(pred_text)
                if norm is None:
                    bad += 1
                    item["model_response"] = ""
                    item["_postprocess_error"] = "cannot_parse_joint_choice"
                else:
                    ok += 1
                    item["model_response"] = norm

            else:  # report
                # 直接保留完整文字（metrics 會再抽 key info）
                if pred_text:
                    ok += 1
                    item["model_response"] = pred_text
                else:
                    bad += 1
                    item["model_response"] = ""
                    item["_postprocess_error"] = "empty_report"

            if keep_raw:
                item["_raw_pred_text"] = pred_text

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    return ok, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="independent_qa / joint_qa / visual_evidence_qa / report_generation")
    ap.add_argument("--in", dest="in_path", required=True, help="input jsonl (your driver output)")
    ap.add_argument("--out", dest="out_path", required=True, help="output jsonl (metric-ready)")
    ap.add_argument("--no-raw", action="store_true", help="do not keep _raw_pred_text")
    args = ap.parse_args()

    ok, bad = process_file(Path(args.in_path), Path(args.out_path), args.task, keep_raw=(not args.no_raw))
    print(f"[POST] task={args.task} ok={ok} bad={bad} out={args.out_path}")

if __name__ == "__main__":
    main()