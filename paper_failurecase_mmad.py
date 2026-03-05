#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_failurecase.py

功能：
- 讀取已跑完的 MMAD result JSONL（含 gt / model_response / image_path / template_paths / question / options / turn_idx）
- 只挑「答錯」題目（pred != gt）
- 依 turn_idx=0..4（Q1~Q5）各挑 per_turn 題
- 對這些題目做「補推論」：要求模型輸出 Reasoning + Final Answer（只做少數題供 paper 用）
- 輸出 JSONL：包含 prompt、raw output、抽出的 final answer 等

支援：
- Qwen3-VL: Qwen3VLForConditionalGeneration + AutoProcessor
- LLaVA-OneVision-1.5: AutoModelForCausalLM (trust_remote_code) + AutoProcessor (trust_remote_code)

注意：
- LLaVA-OneVision 需要較新的 transformers（通常 >=4.55 起跳；你目前貼的 4.36.2 會直接炸）
- 會使用 qwen_vl_utils.process_vision_info；若沒裝則會用 minimal fallback（只支援 images，不支援 videos）
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None


LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)


def extract_letter(text: str, valid_letters: Optional[List[str]] = None) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Final Answer\s*:\s*([A-H])\b", text, flags=re.IGNORECASE)
    if m:
        c = m.group(1).upper()
        if valid_letters and c not in {x.upper() for x in valid_letters}:
            return None
        return c

    m = LETTER_RE.search(text)
    if not m:
        return None
    c = m.group(1).upper()
    if valid_letters and c not in {x.upper() for x in valid_letters}:
        return None
    return c


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def pick_wrong_by_turn(rows: List[Dict[str, Any]], per_turn: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    buckets = defaultdict(list)

    for r in rows:
        gt = (r.get("gt") or "").strip().upper()
        pred = (r.get("model_response") or "").strip().upper()
        if not gt or not pred:
            continue
        if gt == pred:
            continue
        turn = r.get("turn_idx")
        if isinstance(turn, int):
            buckets[turn].append(r)

    picked: List[Dict[str, Any]] = []
    for turn in range(5):  # Q1~Q5
        cand = buckets.get(turn, [])
        if not cand:
            continue
        random.shuffle(cand)
        picked.extend(cand[:per_turn])
    return picked


def safe_open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_options(options: Any) -> Tuple[str, List[str]]:
    """
    Return (options_text, valid_letters).
    """
    opt_lines: List[str] = []
    valid_letters: List[str] = []

    if isinstance(options, dict):
        # keys might be A/B/C...
        keys = list(options.keys())
        # try to sort A,B,C... first
        def _key(k):
            s = str(k).strip().upper()
            return (0, s) if re.fullmatch(r"[A-H]", s) else (1, s)
        keys = sorted(keys, key=_key)

        for k in keys:
            lk = str(k).strip().upper()
            if re.fullmatch(r"[A-H]", lk):
                letter = lk
            else:
                letter = chr(ord("A") + len(valid_letters))
            valid_letters.append(letter)
            opt_lines.append(f"{letter}. {str(options[k]).strip()}")

    elif isinstance(options, list):
        for i, item in enumerate(options):
            s = str(item).strip()
            m = re.match(r"^\s*([A-H])[\.\)]\s*(.*)$", s, flags=re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                txt = m.group(2).strip()
            else:
                letter = chr(ord("A") + i)
                txt = s
            valid_letters.append(letter)
            opt_lines.append(f"{letter}. {txt}")
    else:
        valid_letters = list("ABCD")
        opt_lines = ["A. ...", "B. ...", "C. ...", "D. ..."]

    return "\n".join(opt_lines).strip(), valid_letters


def build_reasoning_prompt(
    question: str,
    options: Any,
    original_pred: Optional[str],
    force_explain_original: bool,
) -> Tuple[str, List[str]]:
    opts_text, valid_letters = normalize_options(options)

    lock_clause = ""
    if force_explain_original and original_pred:
        original_pred = original_pred.strip().upper()
        lock_clause = (
            f"Important: You MUST keep your final choice as '{original_pred}'. "
            f"Do NOT change your decision; only explain why you would choose '{original_pred}'.\n\n"
        )

    prompt = f"""You are an industrial visual inspection expert.

You will be shown one or more TEMPLATE images (normal reference) and one TEST image.
Compare them carefully.

{lock_clause}Please answer in two parts:
1) Reasoning: 3-8 sentences. Mention what the object is, whether there is an anomaly, the anomaly type, and the anomaly location if any.
2) Final Answer: ONLY ONE option letter.

Question:
{question}

Options:
{opts_text}

Output format:
Reasoning: <text>
Final Answer: <ONE letter>
"""
    return prompt, valid_letters


def _get_process_vision_info():
    """
    Prefer qwen_vl_utils.process_vision_info.
    If missing, use a minimal fallback that only collects PIL images.
    """
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
        return process_vision_info
    except Exception:
        def process_vision_info(messages):
            image_inputs = []
            video_inputs = None
            for msg in messages:
                for c in msg.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "image":
                        image_inputs.append(c.get("image"))
            return image_inputs, video_inputs
        return process_vision_info


def generate_with_processor(
    processor,
    model,
    prompt: str,
    images: List[Image.Image],
    max_new_tokens: int,
) -> str:
    process_vision_info = _get_process_vision_info()

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

    # move to cuda if possible
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to("cuda")

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0][prompt_len:]
    gen_text = processor.decode(gen_ids, skip_special_tokens=True)
    return gen_text.strip()


def load_model_and_processor(model_type: str, model_path: str):
    import transformers

    # LLaVA-OneVision custom code typically needs newer transformers than 4.36.x
    if model_type == "llava_ov":
        ver = transformers.__version__
        major_minor = tuple(int(x) for x in ver.split(".")[:2])
        if major_minor < (4, 50):
            raise RuntimeError(
                f"[FATAL] transformers=={ver} is too old for LLaVA-OneVision custom code.\n"
                f"Please upgrade to a newer stable version (e.g. 4.57.x).\n"
                f"Example: uv pip install 'transformers==4.57.3'"
            )

    # Processor (avoid interactive prompt; fix mistral tokenizer regex warning)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
    )

    if model_type == "qwen3vl":
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError(
                "Qwen3VLForConditionalGeneration not available in your transformers install.\n"
                "Use a transformers build that includes Qwen3-VL."
            )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        return model, processor

    # llava_ov
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, choices=["qwen3vl", "llava_ov"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mmad-root", required=True)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--per-turn", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--force-explain-original", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_jsonl)

    print(f"[INFO] Reading input jsonl: {args.input_jsonl}")
    rows = read_jsonl(args.input_jsonl)
    print(f"[INFO] Loaded rows={len(rows)}")

    picked = pick_wrong_by_turn(rows, per_turn=args.per_turn, seed=args.seed)
    print(f"[INFO] Picked wrong cases={len(picked)} (per_turn={args.per_turn}, turns=0..4)")

    if not picked:
        print("[WARN] No wrong cases found. Exit.")
        return

    print("[INFO] Loading model/processor...")
    model, processor = load_model_and_processor(args.model_type, args.model_path)

    out_f = open(args.out_jsonl, "w", encoding="utf-8")
    try:
        for idx, r in enumerate(picked):
            image_rel = r["image_path"]
            template_rels = r.get("template_paths") or []
            question = r.get("question") or ""
            options = r.get("options") or {}
            gt = r.get("gt")
            original_pred = r.get("model_response")
            turn_idx = r.get("turn_idx")
            qtype = r.get("type")

            prompt, valid_letters = build_reasoning_prompt(
                question=question,
                options=options,
                original_pred=original_pred,
                force_explain_original=args.force_explain_original,
            )

            # templates first, query last
            abs_paths = [os.path.join(args.mmad_root, p) for p in (template_rels + [image_rel])]
            for p in abs_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing image: {p}")

            images = [safe_open_image(p) for p in abs_paths]

            gen_text = generate_with_processor(
                processor=processor,
                model=model,
                prompt=prompt,
                images=images,
                max_new_tokens=args.max_new_tokens,
            )

            final_letter = extract_letter(gen_text, valid_letters=valid_letters)

            out_obj = {
                "picked_index": idx,
                "model_type": args.model_type,
                "model_path": args.model_path,
                "mmad_root": args.mmad_root,

                "turn_idx": turn_idx,
                "type": qtype,

                "image_path": image_rel,
                "template_paths": template_rels,

                "question": question,
                "options": options,

                "gt": gt,
                "original_pred": original_pred,

                "force_explain_original": bool(args.force_explain_original),

                "reasoning_prompt": prompt,
                "raw_reasoning_output": gen_text,
                "final_answer_extracted": final_letter,
            }

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"[OK] {idx+1}/{len(picked)} turn={turn_idx} type={qtype} "
                f"orig={original_pred} gt={gt} final={final_letter} img={image_rel}"
            )
    finally:
        out_f.close()

    print(f"[DONE] Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()