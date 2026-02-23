#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# =========================
# Paths (edit if needed)
# =========================
MODEL_PATH = "/work/foobarbaz911/vlmr1/models/VLM-R1-Qwen2.5VL-3B-Math-0305"

# DrVD repo root (contains independent_qa.jsonl + images/)
DRVDR_REPO = "/work/foobarbaz911/vlmr1/datasets/DrVD-Bench-repo"

JSONL_PATH = os.path.join(DRVDR_REPO, "independent_qa.jsonl")
IMAGE_REPO_ROOT = DRVDR_REPO  # sample["image_paths"] is relative to this root

OUT_DIR = "/work/foobarbaz911/vlmr1/outputs"
OUT_JSONL = os.path.join(OUT_DIR, f"qwen_drvd_independent_{JOB_ID}.jsonl")
OUT_ERR = os.path.join(OUT_DIR, f"qwen_drvd_independent_{JOB_ID}.errors.jsonl")

# =========================
# Inference config
# =========================
DEVICE = "cuda"
DTYPE = torch.float16
MAX_NEW_TOKENS = 32

# Optional: truncate dataset for smoke test
LIMIT: Optional[int] = None  # e.g., 200 for quick run

# Save full raw model output for debugging
SAVE_RAW_OUTPUT = True

# =========================
# Vision size control (guaranteed)
# =========================
# 1536*1536 = 2,359,296 (same as common max_pixels=2359296 you saw before)
MAX_PIXELS_DEFAULT = 1536 * 1536

# If OOM, retry with smaller max_pixels (area cap)
MAX_PIXELS_RETRY_SCHEDULE = [
    1536 * 1536,
    1024 * 1024,
    768 * 768,
    512 * 512,
]

# =========================
# Utilities
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

    # dict: {"A": "...", "B": "..."} etc.
    if isinstance(opts, dict):
        keys = sorted(opts.keys(), key=lambda k: str(k))
        lines = []
        letters = []
        for k in keys:
            kk = str(k).strip()
            letters.append(kk.upper())
            lines.append(f"{kk}. {opts[k]}")
        return "\n".join(lines), letters

    # list: ["A. xxx", "B. yyy", ...] or just ["xxx","yyy",...]
    if isinstance(opts, list):
        lines = []
        letters = []
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


def extract_letter(text: str, valid_letters: Optional[List[str]] = None) -> Optional[str]:
    if not text:
        return None
    m = LETTER_RE.search(text)
    if not m:
        return None
    letter = m.group(1).upper()
    if valid_letters and letter not in [v.upper() for v in valid_letters]:
        return None
    return letter


def safe_open_image_with_area_cap(path: str, max_pixels: int) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Guaranteed: cap image area <= max_pixels by resizing while preserving aspect ratio.
    Returns (image, meta) where meta records resizing info.
    """
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

    # Ensure new area is not above cap due to rounding
    # If still above, reduce by 1 step
    while new_w * new_h > max_pixels and new_w > 1 and new_h > 1:
        new_w = max(1, new_w - 1)
        new_h = max(1, new_h - 1)

    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    meta.update(
        {
            "resized": True,
            "new_w": new_w,
            "new_h": new_h,
            "new_area": new_w * new_h,
        }
    )
    return img, meta


def build_prompt(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
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
            + "Please answer with ONLY the option letter (e.g., A, B, C, D)."
        )
    else:
        prompt = header + f"Question:\n{question}\n\n" + "Please answer with the best choice letter."

    return prompt, letters


def to_cuda(batch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in batch.items():
        if hasattr(v, "to"):
            batch[k] = v.to(DEVICE)
    return batch


def is_oom_error(e: Exception) -> bool:
    msg = repr(e)
    return (
        isinstance(e, torch.OutOfMemoryError)
        or "CUDA out of memory" in msg
        or "OutOfMemoryError" in msg
        or "CUBLAS_STATUS_ALLOC_FAILED" in msg
        or "cuDNN error: CUDNN_STATUS_ALLOC_FAILED" in msg
    )


def qwen_vl_generate(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    prompt: str,
    images: List[Image.Image],
) -> str:
    """
    Qwen2.5-VL: build chat template with correct number of image placeholders,
    then pass images=... to processor.
    """
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
    )
    inputs = to_cuda(inputs)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    out = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(JSONL_PATH):
        print(f"[FATAL] JSONL not found: {JSONL_PATH}", file=sys.stderr)
        return 2

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if LIMIT is not None:
        rows = rows[:LIMIT]

    print(f"[INFO] Loaded {len(rows)} samples from {JSONL_PATH}")
    print("[INFO] Loading model...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # SDPA is torch built-in (low risk). FlashAttention requires extra install/build.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    num_ok = 0
    num_err = 0
    num_oom_recovered = 0
    num_resized = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as fout, open(OUT_ERR, "w", encoding="utf-8") as ferr:
        for i, sample in enumerate(tqdm(rows, total=len(rows))):
            try:
                img_paths = normalize_image_paths(sample.get("image_paths") or sample.get("image_path"))
                if not img_paths:
                    raise ValueError("No image_paths in sample")

                abs_paths = [os.path.join(IMAGE_REPO_ROOT, p) for p in img_paths]
                for p in abs_paths:
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Missing image file: {p}")

                prompt, valid_letters = build_prompt(sample)

                raw_out = None
                used_max_pixels = MAX_PIXELS_DEFAULT
                resize_metas: List[Dict[str, Any]] = []
                resized_any = False

                # Try a schedule to auto-recover from OOM on huge images
                for mp in MAX_PIXELS_RETRY_SCHEDULE:
                    used_max_pixels = mp
                    try:
                        # IMPORTANT: guaranteed resize here
                        images = []
                        resize_metas = []
                        resized_any = False
                        for p in abs_paths:
                            img, meta = safe_open_image_with_area_cap(p, max_pixels=mp)
                            images.append(img)
                            resize_metas.append(meta)
                            resized_any = resized_any or meta["resized"]

                        raw_out = qwen_vl_generate(processor, model, prompt, images)
                        break
                    except Exception as e:
                        if is_oom_error(e):
                            torch.cuda.empty_cache()
                            continue
                        raise

                if raw_out is None:
                    raise torch.OutOfMemoryError(f"OOM even after retries: {MAX_PIXELS_RETRY_SCHEDULE}")

                # If we had to reduce mp below default, count as recovered
                if used_max_pixels != MAX_PIXELS_DEFAULT:
                    num_oom_recovered += 1

                if resized_any:
                    num_resized += 1

                letter = extract_letter(raw_out, valid_letters if valid_letters else None)

                out_obj = dict(sample)
                out_obj["model_response"] = letter
                out_obj["used_max_pixels"] = used_max_pixels
                out_obj["resized"] = resized_any
                out_obj["image_resize_meta"] = resize_metas
                if SAVE_RAW_OUTPUT:
                    out_obj["raw_model_output"] = raw_out

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                num_ok += 1

            except Exception as e:
                num_err += 1
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
                continue

    print(f"[DONE] wrote: {OUT_JSONL}")
    print(f"[DONE] errors: {OUT_ERR}")
    print(f"[STATS] ok={num_ok} err={num_err} oom_recovered={num_oom_recovered} resized_any={num_resized}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())