#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# =========================
# Paths (edit if needed)
# =========================
MODEL_PATH = "/work/foobarbaz911/vlmr1/models/VLM-R1-Qwen2.5VL-3B-Math-0305"

# DrVD repo root (contains independent_qa.jsonl + images/)
DRVDR_REPO = "/work/foobarbaz911/vlmr1/datasets/DrVD-Bench-repo"

JSONL_PATH = os.path.join(DRVDR_REPO, "independent_qa.jsonl")
IMAGE_REPO_ROOT = DRVDR_REPO  # IMPORTANT: sample["image_paths"] usually starts with "images/..."

OUT_DIR = "/work/foobarbaz911/vlmr1/outputs"
OUT_JSONL = os.path.join(OUT_DIR, "qwen_drvd_independent.jsonl")
OUT_ERR = os.path.join(OUT_DIR, "qwen_drvd_independent.errors.jsonl")

# =========================
# Inference config
# =========================
DEVICE = "cuda"
DTYPE = torch.float16
MAX_NEW_TOKENS = 32

# Optional: truncate dataset for smoke test
LIMIT: Optional[int] = None  # e.g., 20 to test quickly

# Save full raw model output for debugging
SAVE_RAW_OUTPUT = True

# =========================
# Utilities
# =========================

LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)


def normalize_image_paths(x: Any) -> List[str]:
    """Return list of relative image paths."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        # ensure strings
        return [str(i) for i in x]
    # fallback
    return [str(x)]


def normalize_options(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Return:
      options_text: string for prompt
      option_letters: list of letters existing (A..)
    Handles:
      - sample["options"] as list like ["A. ...", "B. ..."] or ["...","..."]
      - sample["options"] as dict {"A": "...", "B": "..."}
    """
    opts = sample.get("options", None)

    # dict form
    if isinstance(opts, dict):
        # sort by key
        keys = sorted(opts.keys(), key=lambda k: str(k))
        lines = []
        letters = []
        for k in keys:
            kk = str(k).strip()
            letters.append(kk.upper())
            lines.append(f"{kk}. {opts[k]}")
        return "\n".join(lines), letters

    # list form
    if isinstance(opts, list):
        lines = []
        letters = []
        for idx, item in enumerate(opts):
            s = str(item).strip()
            # if already like "A. xxx"
            m = re.match(r"^\s*([A-H])[\.\)]\s*(.*)$", s, flags=re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                text = m.group(2).strip()
                letters.append(letter)
                lines.append(f"{letter}. {text}")
            else:
                # assign letters by position
                letter = chr(ord("A") + idx)
                letters.append(letter)
                lines.append(f"{letter}. {s}")
        return "\n".join(lines), letters

    # fallback: no options
    return "", []


def extract_letter(text: str, valid_letters: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract first A-H letter from text.
    If valid_letters provided, enforce membership.
    """
    if not text:
        return None

    # prefer patterns like "Answer: A" / "A." / "(A)"
    # generic A-H word boundary
    m = LETTER_RE.search(text)
    if not m:
        return None
    letter = m.group(1).upper()
    if valid_letters and letter not in [v.upper() for v in valid_letters]:
        return None
    return letter


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def build_prompt(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Build MCQ prompt.
    Returns (prompt_text, option_letters).
    """
    question = sample.get("question") or sample.get("query") or sample.get("prompt") or ""
    options_text, letters = normalize_options(sample)

    # DrVD has medical modality/task fields; include if present
    modality = sample.get("modality", "")
    task = sample.get("task", "")

    header = "You are a medical expert.\n"
    if modality or task:
        header += f"Modality: {modality}\nTask: {task}\n"
    header += "\n"

    if options_text:
        prompt = (
            header
            + f"Question:\n{question}\n\n"
            + f"Options:\n{options_text}\n\n"
            + "Please answer with ONLY the option letter (e.g., A, B, C, ...)."
        )
    else:
        # If no options provided (shouldn't happen for choice task)
        prompt = (
            header
            + f"Question:\n{question}\n\n"
            + "Please answer with the best choice letter."
        )

    return prompt, letters


def to_cuda(batch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in batch.items():
        if hasattr(v, "to"):
            batch[k] = v.to(DEVICE)
    return batch


def qwen_vl_generate(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    prompt: str,
    images: List[Image.Image],
) -> str:
    """
    Qwen2.5-VL requires image tokens in text; use apply_chat_template.
    """
    content = []
    for _ in images:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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


# =========================
# Main
# =========================
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()

    # Run
    num_ok = 0
    num_err = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as fout, open(OUT_ERR, "w", encoding="utf-8") as ferr:
        for i, sample in enumerate(tqdm(rows, total=len(rows))):
            try:
                img_paths = normalize_image_paths(sample.get("image_paths") or sample.get("image_path"))
                if not img_paths:
                    raise ValueError("No image_paths in sample")

                # Resolve to absolute paths
                abs_paths = [os.path.join(IMAGE_REPO_ROOT, p) for p in img_paths]
                for p in abs_paths:
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Missing image file: {p}")

                images = [safe_open_image(p) for p in abs_paths]

                prompt, valid_letters = build_prompt(sample)

                raw_out = qwen_vl_generate(processor, model, prompt, images)

                letter = extract_letter(raw_out, valid_letters if valid_letters else None)

                out_obj = dict(sample)
                out_obj["model_response"] = letter
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
                            "sample_keys": list(sample.keys()),
                            "image_paths": sample.get("image_paths"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # continue processing
                continue

    print(f"[DONE] wrote: {OUT_JSONL}")
    print(f"[DONE] errors: {OUT_ERR}")
    print(f"[STATS] ok={num_ok} err={num_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())