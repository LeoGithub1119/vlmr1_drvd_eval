#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM


# =========================
# Regex / parsing helpers
# =========================
LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)


def normalize_image_paths(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def normalize_options(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    opts = sample.get("options", None)

    if isinstance(opts, dict):
        keys = sorted(opts.keys(), key=lambda k: str(k))
        lines, letters = [], []
        for k in keys:
            kk = str(k).strip()
            letters.append(kk.upper())
            lines.append(f"{kk}. {opts[k]}")
        return "\n".join(lines), letters

    if isinstance(opts, list):
        lines, letters = [], []
        for idx, item in enumerate(opts):
            s = str(item).strip()
            m = re.match(r"^\s*([A-H])[\.\)]\s*(.*)$", s, flags=re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                text = m.group(2).strip()
                letters.append(letter)
                lines.append(f"{letter}. {text}")
            else:
                letter = chr(ord("A") + idx)
                letters.append(letter)
                lines.append(f"{letter}. {s}")
        return "\n".join(lines), letters

    return "", []


def extract_single_letter(gen_text: str, valid_letters: Optional[List[str]] = None) -> Optional[str]:
    if not gen_text:
        return None
    m = LETTER_RE.search(gen_text)
    if not m:
        return None
    letter = m.group(1).upper()
    if valid_letters:
        valid = {v.upper() for v in valid_letters}
        if letter not in valid:
            return None
    return letter


def extract_joint_letters(gen_text: str, k: int = 4) -> Optional[str]:
    """
    joint_qa 需要輸出 "A,B,C,D" 這種四段答案（允許 A-H）
    """
    if not gen_text:
        return None
    letters = [m.group(1).upper() for m in LETTER_RE.finditer(gen_text)]
    if len(letters) < k:
        return None
    return ",".join(letters[:k])


# =========================
# Image resizing (area cap)
# =========================
def safe_open_image_with_area_cap(path: str, max_pixels: int) -> Tuple[Image.Image, Dict[str, Any]]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    area = w * h

    meta = {
        "orig_w": w,
        "orig_h": h,
        "orig_area": area,
        "resized": False,
        "new_w": w,
        "new_h": h,
        "new_area": area,
        "max_pixels_cap": max_pixels,
    }

    if area <= max_pixels:
        return img, meta

    scale = math.sqrt(max_pixels / float(area))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    while new_w * new_h > max_pixels and new_w > 1 and new_h > 1:
        new_w = max(1, new_w - 1)
        new_h = max(1, new_h - 1)

    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    meta.update({"resized": True, "new_w": new_w, "new_h": new_h, "new_area": new_w * new_h})
    return img, meta


def is_oom_error(e: Exception) -> bool:
    msg = repr(e)
    return (
        isinstance(e, torch.OutOfMemoryError)
        or "CUDA out of memory" in msg
        or "OutOfMemoryError" in msg
        or "CUBLAS_STATUS_ALLOC_FAILED" in msg
        or "CUDNN_STATUS_ALLOC_FAILED" in msg
    )


# =========================
# Prompt builder
# =========================
def build_prompt(sample: Dict[str, Any], task: str) -> Tuple[str, List[str]]:
    """
    - joint_qa: 直接用 sample["joint_prompt"]（資料集已經幫你組好四段問答）
    - 其他：沿用單題格式
    """
    if task == "joint_qa":
        jp = sample.get("joint_prompt", "")
        if not jp:
            raise KeyError("joint_qa sample missing `joint_prompt`")

        # 重要：joint 必須強制輸出格式，避免模型講太多話害你 parse 失敗
        jp2 = jp.rstrip() + "\n\nReturn ONLY four letters separated by commas, e.g., A,B,C,D.\n"
        return jp2, list("ABCDEFGH")
    else:
        question = sample.get("question") or sample.get("query") or sample.get("prompt") or ""
        options_text, letters = normalize_options(sample)

        modality = sample.get("modality", "")
        level = sample.get("level", "")

        header = "You are a medical expert.\n"
        if modality or level:
            header += f"Modality: {modality}\nLevel: {level}\n"
        header += "\n"

        if options_text:
            prompt = (
                header
                + f"Question:\n{question}\n\n"
                + f"Options:\n{options_text}\n\n"
                + "Please answer with ONLY the option letter (e.g., A, B, C, D, E, F, G, H)."
            )
        else:
            prompt = header + f"Question:\n{question}\n\n" + "Please answer with the best choice letter."

        return prompt, letters


# =========================
# LLaVA-OneVision generate
# =========================
def llava_ov_generate(processor, model, prompt: str, images: List[Image.Image], max_new_tokens: int) -> str:
    """
    LLaVA-OneVision-1.5 官方 HF 推理方式：apply_chat_template + process_vision_info。:contentReference[oaicite:2]{index=2}

    注意：
    - 要 trust_remote_code=True 才會有對應的 processor/model 行為
    - process_vision_info 在官方示例是從 qwen_vl_utils import process_vision_info :contentReference[oaicite:3]{index=3}
    """
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `qwen_vl_utils`. "
            "In LLaVA-OneVision-1.5 official HF quick start, they import `process_vision_info` from qwen_vl_utils. "
            "Please ensure it is installed/available in your environment."
        ) from e

    messages = [
    {
        "role": "user",
        "content": [{"type": "image", "image": img} for img in images]
                 + [{"type": "text", "text": prompt}],
    }
]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    # 只 decode 新生成 token（避免把 prompt/options 的 A. B. C. 汙染答案抽取）
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    gen_text = processor.decode(gen_ids, skip_special_tokens=True)
    return gen_text.strip()


# =========================
# Task mapping
# =========================
TASK_TO_JSONL = {
    "independent_qa": "independent_qa.jsonl",
    "joint_qa": "joint_qa.jsonl",
    "visual_evidence_qa": "visual_evidence_qa.jsonl",
    "report_generation": "report_generation.jsonl",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_TO_JSONL.keys()))
    ap.add_argument("--model-path", required=True, help="HF repo id or local directory")
    ap.add_argument("--drvd-repo", required=True, help="Path to DrVD-Bench-repo (contains jsonl + images/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-tag", default="model", help="Used in output filename (e.g., qwen3b, ov15_4b)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max-new-tokens", type=int, default=128)  # joint 建議 128
    ap.add_argument("--max-pixels", type=int, default=1536 * 1536)
    ap.add_argument("--retry", default="1536,1024,768,512", help="retry sizes, e.g. 1536,1024,768,512")
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Required for LLaVA-OneVision-1.5 HF inference",
    )
    args = ap.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "local")

    jsonl_path = os.path.join(args.drvd_repo, TASK_TO_JSONL[args.task])
    image_repo_root = args.drvd_repo

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, f"{args.model_tag}_drvd_{args.task}_{job_id}.jsonl")
    out_err = os.path.join(args.out_dir, f"{args.model_tag}_drvd_{args.task}_{job_id}.errors.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"[FATAL] JSONL not found: {jsonl_path}", file=sys.stderr)
        return 2

    retry_schedule: List[int] = []
    for s in args.retry.split(","):
        s = s.strip()
        if not s:
            continue
        retry_schedule.append(int(s) * int(s))
    if not retry_schedule:
        retry_schedule = [args.max_pixels]

    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    print(f"[INFO] task={args.task} samples={len(rows)} jsonl={jsonl_path}")
    print("[INFO] Loading model/processor...")

    # LLaVA-OneVision-1.5 official HF loading uses trust_remote_code=True :contentReference[oaicite:4]{index=4}
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    ok = err = oom_recovered = resized_any = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout, open(out_err, "w", encoding="utf-8") as ferr:
        for i, sample in enumerate(tqdm(rows, total=len(rows))):
            try:
                img_paths = normalize_image_paths(sample.get("image_paths") or sample.get("image_path"))
                if not img_paths:
                    raise ValueError("No image_paths in sample")

                abs_paths = [os.path.join(image_repo_root, p) for p in img_paths]
                for p in abs_paths:
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Missing image file: {p}")

                prompt, valid_letters = build_prompt(sample, args.task)

                gen_text = None
                used_max_pixels = args.max_pixels
                resize_metas: List[Dict[str, Any]] = []
                did_resize = False

                for mp in retry_schedule:
                    used_max_pixels = mp
                    try:
                        images, resize_metas = [], []
                        did_resize = False
                        for p in abs_paths:
                            img, meta = safe_open_image_with_area_cap(p, max_pixels=mp)
                            images.append(img)
                            resize_metas.append(meta)
                            did_resize = did_resize or meta["resized"]

                        gen_text = llava_ov_generate(
                            processor, model, prompt, images, max_new_tokens=args.max_new_tokens
                        )
                        break
                    except Exception as e:
                        if is_oom_error(e):
                            torch.cuda.empty_cache()
                            continue
                        raise

                if gen_text is None:
                    raise torch.OutOfMemoryError(f"OOM even after retries: {retry_schedule}")

                if used_max_pixels != args.max_pixels:
                    oom_recovered += 1
                if did_resize:
                    resized_any += 1

                # --- Task-specific parsing ---
                if args.task == "joint_qa":
                    pred = extract_joint_letters(gen_text, k=4)  # "A,B,C,D"
                else:
                    pred = extract_single_letter(gen_text, valid_letters if valid_letters else None)

                out_obj = dict(sample)
                out_obj["model_response"] = pred
                out_obj["used_max_pixels"] = used_max_pixels
                out_obj["resized"] = did_resize
                out_obj["image_resize_meta"] = resize_metas
                if args.save_raw:
                    out_obj["raw_model_output"] = gen_text  # 只存生成結果（乾淨）

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                ok += 1

                if (i + 1) % 200 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                err += 1
                ferr.write(
                    json.dumps(
                        {
                            "index": i,
                            "error": repr(e),
                            "unique_id": sample.get("unique_id"),
                            "image_paths": sample.get("image_paths"),
                            "sample_keys": list(sample.keys()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"[DONE] wrote: {out_jsonl}")
    print(f"[DONE] errors: {out_err}")
    print(f"[STATS] ok={ok} err={err} oom_recovered={oom_recovered} resized_any={resized_any}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())