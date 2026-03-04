#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMAD benchmark driver for LLaVA-OneVision (HF)
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
from transformers import AutoProcessor, AutoModelForCausalLM


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
# Image resizing (area cap)
# =========================
def safe_open_image_with_area_cap(path: str, max_pixels: int) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Open image and (optionally) downscale so that w*h <= max_pixels.
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
        "path": path,
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


# =========================
# LLaVA-OneVision HF generate
# =========================
def llava_ov_generate(processor, model, prompt: str, images: List[Image.Image], max_new_tokens: int) -> str:
    """
    HF LLaVA-OneVision inference (requires qwen_vl_utils.process_vision_info).
    Decodes ONLY newly generated tokens to avoid option-letter contamination.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `qwen_vl_utils` (needed for LLaVA-OneVision HF). "
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
    conversation_item typically:
      {
        "Question": "...",
        "Options": {"A": "...", ...} or list,
        "Answer": "A"
      }
    """
    q = (
        conversation_item.get("Question")
        or conversation_item.get("question")
        or conversation_item.get("query")
        or ""
    )
    opts = conversation_item.get("Options") or conversation_item.get("options") or {}

    lines: List[str] = []
    letters: List[str] = []

    if isinstance(opts, dict):
        # Prefer alphabetical order A,B,C...
        keys = sorted(opts.keys(), key=lambda x: str(x))
        for k in keys:
            kk = str(k).strip()
            m = re.match(r"^\s*([A-H])", kk, flags=re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
            else:
                letter = chr(ord("A") + len(letters))
            letters.append(letter)
            lines.append(f"{letter}. {str(opts[k]).strip()}")

    elif isinstance(opts, list):
        for i, item in enumerate(opts):
            s = str(item).strip()
            m = re.match(r"^\s*([A-H])[\.\)]\s*(.*)$", s, flags=re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                text = m.group(2).strip()
            else:
                letter = chr(ord("A") + i)
                text = s
            letters.append(letter)
            lines.append(f"{letter}. {text}")

    else:
        # Fallback, though MMAD should have options
        letters = list("ABCDEFGH")

    options_text = "\n".join(lines).strip()

    prompt = (
        "You are an industrial anomaly detection expert.\n"
        "You will be given one or more TEMPLATE images and one TEST image.\n"
        "Compare them carefully and answer the question.\n\n"
        f"Question:\n{q}\n\n"
        f"Options:\n{options_text}\n\n"
        "Please answer with ONLY the option letter (e.g., A, B, C, D)."
    )
    return prompt, letters


def get_gt_letter(conversation_item: Dict[str, Any]) -> Optional[str]:
    ans = (
        conversation_item.get("Answer")
        or conversation_item.get("answer")
        or conversation_item.get("gt")
        or None
    )
    if ans is None:
        return None
    s = str(ans).strip()
    m = re.match(r"^\s*([A-H])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def normalize_question_text(conversation_item: Dict[str, Any]) -> str:
    return (
        conversation_item.get("Question")
        or conversation_item.get("question")
        or conversation_item.get("query")
        or ""
    )


# =========================
# Resume support
# =========================
def load_done_keys(out_jsonl: str) -> Set[Tuple[str, int]]:
    """
    Load already evaluated (image_path, turn_idx) pairs from an existing JSONL.
    """
    done: Set[Tuple[str, int]] = set()
    if not out_jsonl or not os.path.exists(out_jsonl):
        return done
    try:
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ip = obj.get("image_path")
                    ti = obj.get("turn_idx")
                    if isinstance(ip, str) and isinstance(ti, int):
                        done.add((ip, ti))
                except Exception:
                    continue
    except Exception:
        pass
    return done


# =========================
# Main
# =========================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF repo id or local directory (LLaVA-OneVision HF)")
    ap.add_argument("--mmad-root", required=True, help="MMAD directory containing mmad.json and image folders")
    ap.add_argument("--mmad-json", default="mmad.json", help="mmad.json filename (default: mmad.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-tag", default="model", help="Used in output filename (e.g., llavaov15_8b)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (image-level limit before expanding questions)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=1536 * 1536)
    ap.add_argument(
        "--retry",
        default="1536,1024,768,512",
        help="retry square side list (comma-separated), e.g. 1536,1024,768,512",
    )
    ap.add_argument("--save-raw", action="store_true", help="Store raw model output text in JSONL")

    # few-shot templates
    ap.add_argument("--template-type", choices=["random", "similar"], default="random")
    ap.add_argument("--num-templates", type=int, default=1)

    # chunking
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--chunk-idx", type=int, default=0)

    # resume
    ap.add_argument("--resume", action="store_true", help="Append + skip already done (image_path, turn_idx)")

    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Often required for LLaVA-OneVision HF inference",
    )
    args = ap.parse_args()

    if args.chunk_idx < 0 or args.chunk_idx >= args.num_chunks:
        print(f"[FATAL] invalid chunk-idx {args.chunk_idx} for num-chunks {args.num_chunks}", file=sys.stderr)
        return 2

    job_id = os.environ.get("SLURM_JOB_ID", "local")

    mmad_json_path = os.path.join(args.mmad_root, args.mmad_json)
    if not os.path.exists(mmad_json_path):
        print(f"[FATAL] MMAD json not found: {mmad_json_path}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, f"{args.model_tag}_mmad_{job_id}_c{args.chunk_idx}of{args.num_chunks}.jsonl")
    out_err = os.path.join(args.out_dir, f"{args.model_tag}_mmad_{job_id}_c{args.chunk_idx}of{args.num_chunks}.errors.jsonl")

    # retry schedule as area caps
    retry_schedule: List[int] = []
    for s in args.retry.split(","):
        s = s.strip()
        if not s:
            continue
        side = int(s)
        retry_schedule.append(side * side)
    if not retry_schedule:
        retry_schedule = [args.max_pixels]

    print(f"[INFO] Loading MMAD: {mmad_json_path}")
    with open(mmad_json_path, "r", encoding="utf-8") as f:
        mmad = json.load(f)

    keys = list(mmad.keys())

    # chunking at image level (then expand questions per image)
    keys = [k for idx, k in enumerate(keys) if (idx % args.num_chunks) == args.chunk_idx]
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    print(
        f"[INFO] images={len(keys)} template_type={args.template_type} num_templates={args.num_templates} "
        f"chunk={args.chunk_idx}/{args.num_chunks} retry={retry_schedule}"
    )

    # resume
    done_keys: Set[Tuple[str, int]] = set()
    if args.resume:
        done_keys = load_done_keys(out_jsonl)
        print(f"[INFO] resume enabled: loaded done_keys={len(done_keys)} from {out_jsonl}")

    print("[INFO] Loading model/processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # counters
    ok = err = oom_recovered = resized_any = 0

    # question-level metrics
    n_has_gt = 0
    n_correct = 0
    type_total = defaultdict(int)
    type_correct = defaultdict(int)

    # open output file (append if resume)
    fout_mode = "a" if args.resume else "w"
    ferr_mode = "a" if args.resume else "w"

    with open(out_jsonl, fout_mode, encoding="utf-8") as fout, open(out_err, ferr_mode, encoding="utf-8") as ferr:
        for i, image_rel in enumerate(tqdm(keys, total=len(keys))):
            sample = mmad.get(image_rel, {})
            try:
                # 1) build image list = templates + query
                template_rels = choose_templates(sample, which=args.template_type, k=args.num_templates)
                image_paths = template_rels + [image_rel]
                abs_paths = [os.path.join(args.mmad_root, p) for p in image_paths]

                for p in abs_paths:
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Missing image file: {p}")

                # 2) conversation list
                conv = sample.get("conversation") or sample.get("conversations") or []
                if not isinstance(conv, list) or len(conv) == 0:
                    raise ValueError("Missing conversation in MMAD sample")

                # 3) expand to question-level (MMAD protocol)
                for turn_idx, conv_item in enumerate(conv):
                    if args.resume and (image_rel, turn_idx) in done_keys:
                        continue

                    prompt, valid_letters = build_mmad_prompt(conv_item)

                    gen_text: Optional[str] = None
                    used_max_pixels = args.max_pixels
                    resize_metas: List[Dict[str, Any]] = []
                    did_resize = False

                    # 4) retry schedule with area cap for OOM resilience
                    for mp in retry_schedule:
                        used_max_pixels = mp
                        try:
                            images: List[Image.Image] = []
                            resize_metas = []
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

                    pred = extract_single_letter(gen_text, valid_letters if valid_letters else None)

                    # metrics
                    gt = get_gt_letter(conv_item)
                    qtype = conv_item.get("type") or "UNKNOWN"

                    if gt is not None:
                        n_has_gt += 1
                        type_total[qtype] += 1
                        if pred == gt:
                            n_correct += 1
                            type_correct[qtype] += 1

                    out_obj = {
                        "image_path": image_rel,
                        "turn_idx": turn_idx,
                        "type": qtype,
                        "template_paths": template_rels,
                        "question": normalize_question_text(conv_item),
                        "options": conv_item.get("Options") or conv_item.get("options"),
                        "gt": gt,
                        "model_response": pred,
                        "used_max_pixels": used_max_pixels,
                        "resized": did_resize,
                        "image_resize_meta": resize_metas,
                    }
                    if args.save_raw:
                        out_obj["raw_model_output"] = gen_text

                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    ok += 1

                # occasional cache clear
                if (i + 1) % 200 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                err += 1
                ferr.write(
                    json.dumps(
                        {
                            "index": i,
                            "error": repr(e),
                            "image_path": image_rel,
                            "template_paths": (sample.get("random_templates") or sample.get("similar_templates")),
                            "sample_keys": list(sample.keys()) if isinstance(sample, dict) else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    overall_acc = (n_correct / n_has_gt) if n_has_gt else 0.0

    print(f"[DONE] wrote: {out_jsonl}")
    print(f"[DONE] errors: {out_err}")
    print(f"[STATS] ok_questions={ok} err_images={err} oom_recovered={oom_recovered} resized_any={resized_any}")
    print(f"[METRIC] overall_accuracy={overall_acc:.4f} ({n_correct}/{n_has_gt})")

    # per-type accuracy
    if type_total:
        print("[METRIC] per_type_accuracy:")
        for t in sorted(type_total.keys()):
            tot = type_total[t]
            cor = type_correct.get(t, 0)
            acc = (cor / tot) if tot else 0.0
            print(f"  - {t}: {acc:.4f} ({cor}/{tot})")

    # Generate and save summary
    summary = {
        "overall_accuracy": overall_acc,
        "correct": n_correct,
        "total": n_has_gt,
        "per_type": {
            t: {
                "acc": type_correct[t] / type_total[t] if type_total[t] else 0.0,
                "correct": type_correct[t],
                "total": type_total[t],
            }
            for t in type_total
        }
    }

    summary_path = os.path.join(args.out_dir, f"{args.model_tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUMMARY] saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())