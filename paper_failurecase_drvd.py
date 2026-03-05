#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper_failurecase_drvd_qwen_v4.py

- For DrVD Qwen3-VL outputs that ALREADY contain GT fields:
  - visual_evidence_qa / independent_qa: use row["answer"] vs row["model_response"]
  - joint_qa: use row["modality_answer","organ_answer","lesion_answer","diagnosis_answer"]
            vs row["model_response"] (e.g., "C,A,A,B")
- Re-generates "reasoning" while forcing the model to KEEP the original (possibly wrong) choice(s),
  so you can paste "naked exam" failure cases into paper.

Usage example:
python paper_failurecase_drvd_qwen_v4.py \
  --model-path /work/foobarbaz911/vlmr1/models/Qwen3-VL-8B-Instruct \
  --drvd-root /work/foobarbaz911/vlmr1/datasets/DrVD-Bench-repo \
  --inputs \
    /home/foobarbaz911/VLM-R1/fix_0223/qwen3vl_8b_drvd_visual_evidence_qa_131806.jsonl \
    /home/foobarbaz911/VLM-R1/fix_0223/qwen3vl_8b_drvd_joint_qa_131806.jsonl \
    /home/foobarbaz911/VLM-R1/fix_0223/qwen3vl_8b_drvd_independent_qa_131806.jsonl \
  --out-jsonl /home/foobarbaz911/VLM-R1/fix_0223/paper/drvd_qwen_failcases_reasoning_v4.jsonl \
  --k-per-file 1 --seed 7 --max-new-tokens 256
"""

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None


# Prefer leading "A." / "A," / "A)" then fallback
RE_LEAD = re.compile(r"^\s*([A-D])\s*[\.\,\)]\s*", re.IGNORECASE)
RE_ANY = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def read_jsonl(p: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(p: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_image_paths(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        # common separators for image paths
        if ";" in s:
            return [x.strip() for x in s.split(";") if x.strip()]
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return []


def load_images(drvd_root: str, rels: List[str]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for r in rels:
        p = r if os.path.isabs(r) else os.path.join(drvd_root, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def extract_letter(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        x = json.dumps(x, ensure_ascii=False)
    s = str(x).strip()
    if not s:
        return None

    m = RE_LEAD.search(s)
    if m:
        return m.group(1).upper()

    m = re.search(r"Final Answer\s*:\s*([A-D])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"Answer\s*:\s*([A-D])\b", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    ms = list(RE_ANY.finditer(s))
    return ms[-1].group(1).upper() if ms else None


def split_joint_letters(s: Any) -> Optional[List[str]]:
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if len(parts) < 4:
        return None
    out: List[str] = []
    for p in parts[:4]:
        a = extract_letter(p)
        if not a:
            return None
        out.append(a)
    return out


def guess_uid(row: Dict[str, Any]) -> str:
    for k in ["unique_id", "uid", "id", "question_id", "qid", "sample_id"]:
        if row.get(k) is not None:
            return str(row[k])
    return ""


def guess_model_text(row: Dict[str, Any]) -> str:
    for k in ["model_response", "response", "output", "text", "pred_text", "generation", "answer"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(row.get("model_response", "") or "").strip()


def guess_prompt(row: Dict[str, Any]) -> str:
    # DrVD often has joint_prompt OR question/options
    jp = row.get("joint_prompt")
    if isinstance(jp, str) and jp.strip():
        return jp.strip()

    q = row.get("question")
    opts = row.get("options")
    if isinstance(q, str) and q.strip():
        if isinstance(opts, list) and opts:
            return q.strip() + "\nOptions:\n" + "\n".join([str(x) for x in opts])
        return q.strip()

    for k in ["prompt", "input_text", "text"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_reasoning_prompt(base_prompt: str, pred: str) -> str:
    return f"""You are answering a medical visual question based on the provided medical image(s).

Below is the original question (with options). You previously chose option '{pred}'.
Important: You MUST keep your final choice as '{pred}'. Do NOT change your decision; only explain why you would choose '{pred}'.

Original question:
{base_prompt}

Now respond in this exact format:
Reasoning: 3-8 sentences. Briefly describe what the image is, whether there is an abnormality, what abnormality, and where it is located (if applicable).
Final Answer: {pred}
"""


def build_joint_reasoning_prompt(joint_prompt: str, pred_csv: str) -> str:
    # pred_csv example: "C,A,A,B"
    return f"""You are answering a 4-step medical reasoning task based on the provided medical image(s).

Below is the original multi-question prompt. You previously answered:
{pred_csv}

Important: You MUST keep EXACTLY the same final answers ({pred_csv}). Do NOT change any letter. Only explain WHY you chose them.

Original prompt:
{joint_prompt}

Now output in this exact format:

Q1_Description: 1-3 sentences describing what the image is (modality/body region) and key visual cues.
Q1_Why: 1-2 sentences explaining why you chose the Q1 option.

Q2_Abnormality: 1-3 sentences describing whether there is abnormality.
Q2_Why: 1-2 sentences explaining why you chose the Q2 option.

Q3_Lesion: 1-3 sentences describing what lesion or finding is present.
Q3_Why: 1-2 sentences explaining why you chose the Q3 option.

Q4_Location: 1-3 sentences describing where the lesion is located.
Q4_Why: 1-2 sentences explaining why you chose the Q4 option.

Final Answer: {pred_csv}
"""


def get_process_vision_info():
    # qwen-vl-utils provides process_vision_info; fallback for older envs
    try:
        from qwen_vl_utils import process_vision_info
        return process_vision_info
    except Exception:
        def process_vision_info(messages):
            imgs = []
            for msg in messages:
                for c in msg.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "image":
                        imgs.append(c.get("image"))
            return imgs, None
        return process_vision_info


@torch.no_grad()
def qwen_generate(model, processor, prompt: str, images: List[Image.Image], max_new_tokens: int) -> str:
    pvi = get_process_vision_info()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = pvi(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to("cuda")

    # Greedy decode for reproducibility (paper-friendly)
    out_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0][prompt_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def is_wrong_visual_or_indep(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    # GT is inside current row (answer)
    gt = extract_letter(row.get("answer"))
    pred = extract_letter(row.get("model_response")) or extract_letter(row.get("raw_model_output"))
    if gt and pred and gt != pred:
        return gt, pred
    return None


def is_wrong_joint(row: Dict[str, Any]) -> Optional[Tuple[List[str], List[str]]]:
    # GT is inside current row (four answers)
    gt4 = [
        extract_letter(row.get("modality_answer")),
        extract_letter(row.get("organ_answer")),
        extract_letter(row.get("lesion_answer")),
        extract_letter(row.get("diagnosis_answer")),
    ]
    if any(x is None for x in gt4):
        return None

    pred4 = split_joint_letters(row.get("model_response")) or split_joint_letters(row.get("raw_model_output"))
    if not pred4:
        return None

    if gt4 != pred4:
        return gt4, pred4
    return None


def infer_task_from_filename(path: str) -> str:
    base = os.path.basename(path).lower()
    if "joint" in base:
        return "joint_qa"
    if "visual" in base:
        return "visual_evidence_qa"
    return "independent_qa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--drvd-root", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--k-per-file", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    random.seed(args.seed)

    if Qwen3VLForConditionalGeneration is None:
        raise RuntimeError("Qwen3VLForConditionalGeneration not available in this transformers build.")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype="auto",          # (avoid torch_dtype deprecated warning)
        device_map="auto",
    ).eval()

    out_rows: List[Dict[str, Any]] = []

    for in_path in args.inputs:
        task = infer_task_from_filename(in_path)
        rows = read_jsonl(in_path)

        wrong = []
        for r in rows:
            uid = guess_uid(r)
            if not uid:
                continue

            if task == "joint_qa":
                res = is_wrong_joint(r)
                if not res:
                    continue
                gt4, pred4 = res
                wrong.append((uid, r, gt4, pred4))
            else:
                res = is_wrong_visual_or_indep(r)
                if not res:
                    continue
                gt, pred = res
                wrong.append((uid, r, gt, pred))

        random.shuffle(wrong)
        pick = wrong[: args.k_per_file]
        print(f"[INFO] {task} rows={len(rows)} wrong={len(wrong)} picked={len(pick)}")

        for item in pick:
            if task == "joint_qa":
                uid, r, gt4, pred4 = item
                img_rels = parse_image_paths(r.get("image_paths"))
                base_prompt = guess_prompt(r)
                pred_str = ",".join(pred4)
                reasoning_prompt = build_joint_reasoning_prompt(base_prompt, pred_str)
                images = load_images(args.drvd_root, img_rels)
                gen = qwen_generate(model, processor, reasoning_prompt, images, args.max_new_tokens)

                out_rows.append(
                    {
                        "task": task,
                        "source_jsonl": in_path,
                        "unique_id": uid,
                        "image_paths": img_rels,
                        "gt": ",".join(gt4),
                        "original_pred": pred_str,
                        "base_prompt": base_prompt,
                        "reasoning_prompt": reasoning_prompt,
                        "original_model_output": guess_model_text(r),
                        "raw_reasoning_output": gen,
                    }
                )
            else:
                uid, r, gt, pred = item
                img_rels = parse_image_paths(r.get("image_paths"))
                base_prompt = guess_prompt(r)
                reasoning_prompt = build_reasoning_prompt(base_prompt, pred)
                images = load_images(args.drvd_root, img_rels)
                gen = qwen_generate(model, processor, reasoning_prompt, images, args.max_new_tokens)

                out_rows.append(
                    {
                        "task": task,
                        "source_jsonl": in_path,
                        "unique_id": uid,
                        "image_paths": img_rels,
                        "gt": gt,
                        "original_pred": pred,
                        "base_prompt": base_prompt,
                        "reasoning_prompt": reasoning_prompt,
                        "original_model_output": guess_model_text(r),
                        "raw_reasoning_output": gen,
                    }
                )

    write_jsonl(args.out_jsonl, out_rows)
    print(f"[DONE] Wrote: {args.out_jsonl} (n={len(out_rows)})")


if __name__ == "__main__":
    main()