#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMAD benchmark driver for Qwen3-VL (Transformers)
- Question-level evaluation (MMAD "39,672 questions / 8,366 images" protocol)
- Supports few-shot templates (random/similar) and K templates
- Chunking for parallel Slurm runs
- Robust OOM retry with progressive image area caps
- Writes JSONL predictions + JSONL errors
- Prints overall accuracy + per-task accuracy
- Optional: resume (skip already-evaluated (image_path, turn_idx))
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# Qwen3-VL class (present in recent transformers)
try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None


# =========================
# Regex / parsing helpers
# =========================
LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)


def extract_single_letter(gen_text: str, valid_letters: Optional[List[str]] = None) -> Optional[str]:
    """
    Extract the first A-H letter from model output.
    If valid_letters is provided, only accept those letters.
    """
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
# Image resize (area cap)
# =========================
def resize_to_max_pixels(img: Image.Image, max_pixels: int) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Resize image if w*h > max_pixels, keeping aspect ratio.
    Returns (img2, meta).
    """
    w, h = img.size
    orig_area = w * h
    meta = {
        "orig_w": w,
        "orig_h": h,
        "orig_area": orig_area,
        "resized": False,
        "new_w": w,
        "new_h": h,
        "new_area": orig_area,
        "max_pixels_cap": max_pixels,
    }
    if orig_area <= max_pixels:
        return img, meta

    scale = math.sqrt(max_pixels / float(orig_area))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img2 = img.resize((new_w, new_h), resample=Image.BICUBIC)
    meta.update(
        {
            "resized": True,
            "new_w": new_w,
            "new_h": new_h,
            "new_area": new_w * new_h,
        }
    )
    return img2, meta


# =========================
# Qwen3-VL generate
# =========================
def qwen3_vl_generate(processor, model, prompt: str, images: List[Image.Image], max_new_tokens: int) -> str:
    """
    Qwen3-VL official-style HF inference:
      - messages + apply_chat_template
      - qwen_vl_utils.process_vision_info
    """
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `qwen_vl_utils` (Qwen3-VL 官方示例會用到 process_vision_info). "
            "請先安裝/提供該套件。"
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
    )

    # move to cuda
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    # 只 decode 新生成 token，避免 prompt/options 的字母污染抽取
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    gen_text = processor.decode(gen_ids, skip_special_tokens=True)
    return gen_text.strip()


# =========================
# MMAD prompt / sample helpers
# =========================
def choose_templates(sample: Dict[str, Any], which: str, k: int) -> List[str]:
    """
    Select K templates from either:
      - similar_templates
      - random_templates
    """
    if which == "similar":
        cand = sample.get("similar_templates") or []
    else:
        cand = sample.get("random_templates") or []
    cand = [str(x) for x in cand]
    return cand[:k]


def build_mmad_prompt(conversation_item: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Build a MCQ prompt and valid option letters.
    MMAD options is dict like {"A": "...", "B": "..."}.
    """
    q = conversation_item["Question"].strip()
    options: Dict[str, str] = conversation_item.get("Options") or {}
    keys = sorted(options.keys())
    valid_letters = [k.upper() for k in keys]

    lines = [q, "", "Options:"]
    for k in keys:
        lines.append(f"{k}. {options[k]}")
    lines.append("")
    lines.append("Please answer with ONLY the option letter (e.g., A, B, C, D).")
    return "\n".join(lines), valid_letters


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def flatten_mmad(mmad_json: Dict[str, Any], templates_k: int, templates_which: str) -> List[Dict[str, Any]]:
    """
    Flatten MMAD into question-level samples.
    Each image has a conversation list (multi-turn). We output one record per turn.
    """
    out = []
    for image_key, sample in mmad_json.items():
        # IMPORTANT: use key as canonical image path (same as your LLaVA script)
        image_path = image_key

        conv = sample.get("conversation") or []
        tpaths = choose_templates(sample, templates_which, templates_k)

        for turn_idx, turn in enumerate(conv):
            prompt, valid = build_mmad_prompt(turn)
            gt = (turn.get("Answer") or "").strip()
            gt = gt.upper() if gt else ""

            out.append(
                {
                    "image_key": image_key,
                    "image_path": image_path,
                    "template_paths": tpaths,
                    "turn_idx": turn_idx,
                    "question": turn.get("Question"),
                    "options": turn.get("Options"),
                    "gt": gt,
                    "type": turn.get("type"),
                    "prompt": prompt,
                    "valid_letters": valid,
                }
            )
    return out


def compute_accuracy(records: List[Dict[str, Any]]) -> Tuple[float, int, int, Dict[str, Tuple[int, int]]]:
    """
    Compute overall accuracy and per-type accuracy.
    """
    correct = 0
    total = 0
    per = defaultdict(lambda: [0, 0])  # type -> [correct, total]
    for r in records:
        if r.get("gt") is None:
            continue
        pred = r.get("model_response")
        gt = r.get("gt")
        if not gt:
            continue
        total += 1
        if pred == gt:
            correct += 1
            per[r.get("type", "NA")][0] += 1
        per[r.get("type", "NA")][1] += 1

    acc = (correct / total) if total else 0.0
    per_type = {k: (v[0], v[1]) for k, v in per.items()}
    return acc, correct, total, per_type


def load_done_keys(out_jsonl: str) -> Set[Tuple[str, int]]:
    """
    For resume: load (image_path, turn_idx) already evaluated.
    """
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                done.add((j.get("image_path"), int(j.get("turn_idx", -1))))
            except Exception:
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmad-dir", required=True, help="MMAD dataset root dir (contains mmad.json, DS-MVTec, MVTec-AD, etc.)")
    ap.add_argument("--mmad-json", default="mmad.json", help="mmad.json filename under mmad-dir")
    ap.add_argument("--out-dir", required=True, help="output folder")
    ap.add_argument("--model-path", required=True, help="local path or HF repo id of Qwen3-VL")
    ap.add_argument("--model-tag", required=True, help="tag prefix in output filename")
    ap.add_argument("--max-new-tokens", type=int, default=16)

    ap.add_argument("--templates-k", type=int, default=1, help="number of templates to include (few-shot)")
    ap.add_argument("--templates-which", choices=["random", "similar"], default="random")

    ap.add_argument("--chunk-idx", type=int, default=0, help="chunk index (0-based)")
    ap.add_argument("--chunk-num", type=int, default=1, help="number of chunks")

    ap.add_argument("--resume", action="store_true", help="resume: skip already predicted samples in out jsonl")

    ap.add_argument("--max-pixels", type=int, default=1536 * 1536, help="initial max pixel area cap")
    ap.add_argument("--oom-retry", type=int, default=3, help="oom retry times with decreasing pixel cap")

    # NEW: limit question-level samples (must apply BEFORE chunk)
    ap.add_argument("--max-samples", type=int, default=0, help="0=no limit. Limit number of flattened question samples BEFORE chunking.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # output names (keep your style)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    out_jsonl = os.path.join(args.out_dir, f"{args.model_tag}_mmad_{job_id}_c{args.chunk_idx}of{args.chunk_num}.jsonl")
    err_jsonl = os.path.join(args.out_dir, f"{args.model_tag}_mmad_{job_id}_c{args.chunk_idx}of{args.chunk_num}.errors.jsonl")
    summary_json = os.path.join(args.out_dir, f"{args.model_tag}_summary.json")

    # load mmad.json
    mmad_path = os.path.join(args.mmad_dir, args.mmad_json)
    with open(mmad_path, "r", encoding="utf-8") as f:
        mmad = json.load(f)

    flat = flatten_mmad(mmad, templates_k=args.templates_k, templates_which=args.templates_which)
    n_before = len(flat)

    # ---- APPLY LIMIT BEFORE CHUNK ----
    if args.max_samples and args.max_samples > 0:
        flat = flat[: args.max_samples]
    n_after = len(flat)

    # chunk
    if args.chunk_num < 1:
        args.chunk_num = 1
    chunk_size = math.ceil(n_after / args.chunk_num) if n_after else 0
    s = args.chunk_idx * chunk_size if chunk_size else 0
    e = min(n_after, (args.chunk_idx + 1) * chunk_size) if chunk_size else 0
    flat = flat[s:e]

    print(
        f"[INFO] total_flat_before={n_before} total_flat_after_limit={n_after} "
        f"chunk={args.chunk_idx}/{args.chunk_num} range=[{s},{e}) samples={len(flat)} mmad_json={mmad_path} "
        f"max_samples={args.max_samples}"
    )

    # quick sanity check for paths
    if flat:
        p0_rel = flat[0]["image_path"]
        p0_abs = os.path.join(args.mmad_dir, p0_rel)
        print(f"[CHECK] first_image_rel={p0_rel} exists={os.path.exists(p0_abs)} abs={p0_abs}")

    # resume keys
    done_keys = load_done_keys(out_jsonl) if args.resume else set()
    if args.resume:
        print(f"[INFO] resume enabled: loaded done_keys={len(done_keys)} from {out_jsonl}")

    # load model/processor
    print("[INFO] Loading model/processor (Qwen3-VL)...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    if Qwen3VLForConditionalGeneration is None:
        raise RuntimeError(
            "Qwen3VLForConditionalGeneration not found in your transformers. "
            "Qwen3-VL 官方說明需要 transformers >= 4.57.0（很多人用 git main）。"
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",
    ).to("cuda").eval()

    ok_records: List[Dict[str, Any]] = []
    ok = 0
    err = 0
    resized_any = 0
    oom_recovered = 0

    # open writers
    out_f = open(out_jsonl, "a", encoding="utf-8")
    err_f = open(err_jsonl, "a", encoding="utf-8")

    try:
        for idx, item in enumerate(tqdm(flat, total=len(flat))):
            img_rel = item["image_path"]
            turn_idx = int(item["turn_idx"])
            if args.resume and (img_rel, turn_idx) in done_keys:
                continue

            # Build absolute paths
            img_abs = os.path.join(args.mmad_dir, img_rel)
            template_abs = [os.path.join(args.mmad_dir, p) for p in (item.get("template_paths") or [])]

            try:
                # Load images (templates + query image)
                imgs: List[Image.Image] = []
                resize_meta = []

                max_pixels = args.max_pixels
                used_max_pixels = max_pixels

                def _load_with_cap(pth: str, cap: int) -> Tuple[Image.Image, Dict[str, Any]]:
                    img0 = load_image(pth)
                    img1, meta = resize_to_max_pixels(img0, cap)
                    meta["path"] = pth
                    return img1, meta

                # OOM retry loop: progressively reduce pixel cap
                last_exc = None
                gen_text = ""
                for attempt in range(args.oom_retry + 1):
                    try:
                        imgs = []
                        resize_meta = []

                        # templates first
                        for tp in template_abs:
                            im, meta = _load_with_cap(tp, max_pixels)
                            imgs.append(im)
                            resize_meta.append(meta)

                        # query image last
                        imq, metaq = _load_with_cap(img_abs, max_pixels)
                        imgs.append(imq)
                        resize_meta.append(metaq)

                        if any(m.get("resized") for m in resize_meta):
                            resized_any += 1

                        # generate
                        gen_text = qwen3_vl_generate(
                            processor=processor,
                            model=model,
                            prompt=item["prompt"],
                            images=imgs,
                            max_new_tokens=args.max_new_tokens,
                        )
                        last_exc = None
                        used_max_pixels = max_pixels
                        break

                    except Exception as e:
                        last_exc = e
                        if is_oom_error(e) and attempt < args.oom_retry:
                            max_pixels = int(max_pixels * 0.7)
                            max_pixels = max(max_pixels, 512 * 512)
                            torch.cuda.empty_cache()
                            oom_recovered += 1
                            continue
                        else:
                            raise

                if last_exc is not None:
                    raise last_exc

                valid_letters = item.get("valid_letters") or []
                pred = extract_single_letter(gen_text, valid_letters=valid_letters)

                rec = {
                    "image_path": img_rel,
                    "template_paths": item.get("template_paths") or [],
                    "turn_idx": turn_idx,
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "gt": item.get("gt"),
                    "type": item.get("type"),
                    "model_response": pred,
                    "raw_model_output": gen_text,
                    "used_max_pixels": used_max_pixels,
                    "resized": any(m.get("resized") for m in resize_meta),
                    "image_resize_meta": resize_meta,
                }

                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()

                ok_records.append(rec)
                ok += 1

            except Exception as e:
                err += 1
                err_rec = {
                    "index": idx,
                    "error": repr(e),
                    "image_path": item.get("image_path"),
                    "turn_idx": item.get("turn_idx"),
                    "sample_keys": list(item.keys()),
                }
                err_f.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                err_f.flush()

    finally:
        out_f.close()
        err_f.close()

    acc, correct, total, per_type = compute_accuracy(ok_records)
    print(f"[STATS] ok={ok} err={err} oom_recovered={oom_recovered} resized_any={resized_any}")
    print(f"[METRIC] accuracy_on_gt={acc:.4f} ({correct}/{total})")

    # write summary for convenience
    summary = {
        "overall_accuracy": acc,
        "correct": correct,
        "total": total,
        "per_type": {k: {"acc": (c / t if t else 0.0), "correct": c, "total": t} for k, (c, t) in per_type.items()},
        "out_jsonl": out_jsonl,
        "err_jsonl": err_jsonl,
        "model_path": args.model_path,
        "model_tag": args.model_tag,
        "chunk_idx": args.chunk_idx,
        "chunk_num": args.chunk_num,
        "mmad_json": mmad_path,
        "templates_k": args.templates_k,
        "templates_which": args.templates_which,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "oom_retry": args.oom_retry,
        "max_samples": args.max_samples,
        "total_flat_before": n_before,
        "total_flat_after_limit": n_after,
        "chunk_range": [s, e],
        "samples_in_this_run": len(flat),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote: {out_jsonl}")
    print(f"[DONE] errors: {err_jsonl}")
    print(f"[DONE] summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())