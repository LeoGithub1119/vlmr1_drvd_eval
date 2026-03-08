# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple

from babel.numbers import parse_decimal
from .utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json

from open_r1.vlm_modules import *

from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer

from openai import OpenAI

logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
    monkey_patch_torch_load,
)

monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()

tokenizer = None


def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """

    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={"help": "Choose reward method: 'default', 'mcp', ..."},
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Choose task type: 'default', 'gui', ..."},
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )


def extract_choice(text):
    text = text.upper()
    text = re.sub(r"\s+", " ", text)

    choices = re.findall(r"(?<![A-Z])([A-Z])(?=[\.\,\?\!\:\;]|$)", text)

    if not choices:
        return None

    if len(choices) == 1:
        return choices[0]

    choice_scores = {choice: 0 for choice in choices}

    keywords = [
        "答案", "选择", "正确", "是", "对",
        "answer", "correct", "choose", "select", "right",
        "认为", "应该", "觉得", "think", "believe", "should"
    ]

    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos - 20): min(len(text), pos + 20)]

        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        if pos > len(text) * 0.7:
            choice_scores[choice] += 2

        if pos < len(text) - 1 and text[pos + 1] in "。.!！,，":
            choice_scores[choice] += 1

    return max(choice_scores.items(), key=lambda x: x[1])[0]


def evaluate_answer_similarity(student_answer, ground_truth):
    """Use llm to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a evaluation expert. First, analyze the student's response "
                        "to identify and extract their final answer. Then, compare the extracted "
                        "answer with the correct solution. Output ONLY '1.0' if the extracted answer "
                        "matches the correct solution in meaning, or '0.0' if the student's response "
                        "does not contain a clear or correct answer. No other output is allowed."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Student's response: {student_answer}\n"
                        f"Correct solution: {ground_truth}\n"
                        "Output only 1.0 or 0.0:"
                    ),
                },
            ],
            temperature=0,
        )
        result = response.choices[0].message.content.strip()
        return float(result)

    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        return 1.0 if student_answer == ground_truth else 0.0


def llm_reward(content, sol, **kwargs):
    sol_match = re.search(r"<answer>(.*?)</answer>", sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    content_matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return evaluate_answer_similarity(student_answer, ground_truth)


def mcq_reward(content, sol, **kwargs):
    sol_match = re.search(r"<answer>(.*?)</answer>", sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    has_choices = extract_choice(ground_truth)
    correct_choice = has_choices.upper() if has_choices else sol.strip()

    content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()
    student_choice = extract_choice(student_answer)
    if student_choice:
        reward = 1.0 if student_choice == correct_choice else 0.0
    else:
        reward = 0.0

    return reward


def yes_no_reward(content, sol, **kwargs):
    content = content.lower()
    sol = sol.lower()

    sol_match = re.search(r"<answer>(.*?)</answer>", sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else content.strip()

    ground_yes_no = re.search(r"(yes|no)", ground_truth)
    ground_yes_no = ground_yes_no.group(1) if ground_yes_no else ""
    student_yes_no = re.search(r"(yes|no)", student_answer)
    student_yes_no = student_yes_no.group(1) if student_yes_no else ""

    reward = 1.0 if ground_yes_no == student_yes_no else 0.0
    return reward


def calculate_map(pred_bbox_list, gt_bbox_list, score_type=0):
    gt_json = {"annotations": [], "images": [], "categories": []}
    gt_json["images"] = [{
        "id": 0,
        "width": 2048,
        "height": 2048,
        "file_name": "image_0.jpg",
    }]

    gt_json["categories"] = []

    cats2id = {}
    cat_count = 0
    for idx, gt_bbox in enumerate(gt_bbox_list):
        if gt_bbox["label"] not in cats2id:
            cats2id[gt_bbox["label"]] = cat_count
            gt_json["categories"].append({
                "id": cat_count,
                "name": gt_bbox["label"],
            })
            cat_count += 1

        gt_json["annotations"].append({
            "id": idx + 1,
            "image_id": 0,
            "category_id": cats2id[gt_bbox["label"]],
            "bbox": [
                gt_bbox["bbox_2d"][0],
                gt_bbox["bbox_2d"][1],
                gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0],
                gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1],
            ],
            "area": (
                (gt_bbox["bbox_2d"][2] - gt_bbox["bbox_2d"][0]) *
                (gt_bbox["bbox_2d"][3] - gt_bbox["bbox_2d"][1])
            ),
            "iscrowd": 0,
        })
    coco_gt = COCO(gt_json)

    dt_json = []
    for pred_bbox in pred_bbox_list:
        try:
            dt_json.append({
                "image_id": 0,
                "category_id": cats2id[pred_bbox["label"]],
                "bbox": [
                    pred_bbox["bbox_2d"][0],
                    pred_bbox["bbox_2d"][1],
                    pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0],
                    pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1],
                ],
                "score": 1.0,
                "area": (
                    (pred_bbox["bbox_2d"][2] - pred_bbox["bbox_2d"][0]) *
                    (pred_bbox["bbox_2d"][3] - pred_bbox["bbox_2d"][1])
                ),
            })
        except Exception:
            pass

    if len(dt_json) == 0:
        return 0.0

    coco_dt = coco_gt.loadRes(dt_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[score_type]


def map_reward(content, sol, length_reward=False, score_type=0, **kwargs):
    pattern = r"```json(.*?)```"
    json_match = re.findall(pattern, sol, re.DOTALL)
    bbox_json = json_match[-1].strip() if json_match else None

    gt_bbox_list = []
    if bbox_json:
        bbox_data = json.loads(bbox_json)
        gt_bbox_list = [item for item in bbox_data]

    pred_bbox_list = []
    json_match = re.findall(pattern, content, re.DOTALL)
    if json_match:
        try:
            bbox_data = json.loads(json_match[-1].strip())
            pred_bbox_list = [item for item in bbox_data]
        except Exception:
            pred_bbox_list = []

    if len(pred_bbox_list) > 0 and len(gt_bbox_list) > 0:
        bbox_reward = calculate_map(pred_bbox_list, gt_bbox_list, score_type=score_type)
    elif len(pred_bbox_list) == 0 and len(gt_bbox_list) == 0:
        bbox_reward = 1.0
    else:
        bbox_reward = 0.0

    if length_reward:
        gt_length = len(gt_bbox_list)
        pred_length = len(pred_bbox_list)
        length_score = 1.0 if gt_length >= pred_length else gt_length / pred_length
        return bbox_reward * length_score
    else:
        return bbox_reward


def od_reward(content, sol, score_type=0, **kwargs):
    match_pattern = r"<answer>(.*?)</answer>"

    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None

    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    if student_answer is None:
        return 0.0
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    else:
        return map_reward(student_answer, ground_truth, score_type=score_type)


def odLength_reward(content, sol, **kwargs):
    match_pattern = r"<answer>(.*?)</answer>"

    sol_match = re.search(match_pattern, sol, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else None

    content_match = re.findall(match_pattern, content, re.DOTALL)
    student_answer = content_match[-1].strip() if content_match else None

    if student_answer is None:
        return 0.0
    elif ground_truth == "None" and student_answer == "None":
        return 1.0
    else:
        bbox_reward = map_reward(student_answer, ground_truth, length_reward=True, score_type=0)
        return bbox_reward


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (
        (box1[2] - box1[0]) * (box1[3] - box1[1]) +
        (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    )
    return float(inter) / union


def detection_score(content, sol, iou_threshold=0.5, alpha=0.7, beta=0.0, gamma=0.3):
    pattern = r"```json(.*?)```"
    json_match = re.search(pattern, clean_text(content), re.DOTALL)
    content_bbox_json = json_match.group(1).strip() if json_match else None
    if content_bbox_json:
        try:
            bbox_data = json.loads(content_bbox_json)
            pred_boxes = [item for item in bbox_data]
        except Exception:
            pred_boxes = []
    else:
        pred_boxes = []

    json_match = re.search(pattern, clean_text(sol), re.DOTALL)
    sol_bbox_json = json_match.group(1).strip() if json_match else None
    if sol_bbox_json:
        bbox_data = json.loads(sol_bbox_json)
        gt_boxes = [item for item in bbox_data]
    else:
        gt_boxes = []

    if len(gt_boxes) == 0:
        return 1.0 if not pred_boxes else 0.0

    if len(pred_boxes) == 0:
        return 0.0

    matches = []
    unmatched_preds = list(range(len(pred_boxes)))
    unmatched_gts = list(range(len(gt_boxes)))

    iou_matrix = []
    for pred_box in pred_boxes:
        iou_row = []
        for gt_box in gt_boxes:
            try:
                curr_iou = iou(pred_box["bbox_2d"], gt_box["bbox_2d"])
            except Exception:
                curr_iou = 0.0
            iou_row.append(curr_iou)
        iou_matrix.append(iou_row)

    while unmatched_preds and unmatched_gts:
        max_iou = -1
        max_pred_idx = -1
        max_gt_idx = -1

        for pred_idx in unmatched_preds:
            for gt_idx in unmatched_gts:
                curr_iou = iou_matrix[pred_idx][gt_idx]
                if curr_iou > max_iou:
                    max_iou = curr_iou
                    max_pred_idx = pred_idx
                    max_gt_idx = gt_idx

        if max_iou < iou_threshold:
            break

        try:
            pred_label = pred_boxes[max_pred_idx]["label"].lower()
        except Exception:
            pred_label = ""
        try:
            gt_label = gt_boxes[max_gt_idx]["label"].lower()
        except Exception:
            gt_label = ""

        label_correct = (pred_label == gt_label)

        if label_correct:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": max_iou,
                "label_correct": label_correct,
            })
        else:
            matches.append({
                "pred_idx": max_pred_idx,
                "gt_idx": max_gt_idx,
                "iou": 0,
                "label_correct": label_correct,
            })

        unmatched_preds.remove(max_pred_idx)
        unmatched_gts.remove(max_gt_idx)

    position_score = sum(m["iou"] for m in matches) / len(gt_boxes) if matches else 0.0
    label_score = sum(1.0 for m in matches if m["label_correct"]) / len(gt_boxes) if matches else 0.0

    miss_rate = len(unmatched_gts) / len(gt_boxes)
    false_alarm_rate = len(unmatched_preds) / len(pred_boxes) if pred_boxes else 0.0

    completeness_score = 1.0 - (miss_rate + false_alarm_rate) / 2.0

    final_score = (
        alpha * position_score +
        beta * label_score +
        gamma * completeness_score
    ) / (alpha + beta + gamma)

    return final_score


def cosine_reward(content, tokenizer, acc_reward, **kwargs):
    min_len_value_wrong = 0.0
    max_len_value_wrong = -0.5
    min_len_value_correct = 1.0
    max_len_value_correct = 0.5
    cosine_max_len = 1024

    gen_len = len(tokenizer.encode(content))
    acc_reward = 1.0
    is_correct = acc_reward >= 0.7

    if is_correct:
        min_value = max_len_value_correct
        max_value = min_len_value_correct
    else:
        min_value = min_len_value_wrong
        max_value = max_len_value_wrong

    reward = max_value - (max_value - min_value) * (
        1 - math.cos(gen_len * math.pi / cosine_max_len)
    ) / 2

    return reward


def repetition_reward(content, **kwargs):
    max_penalty = -1.0

    if content == "":
        return 0.0

    pattern = r"```json(.*?)```"
    json_match = re.search(pattern, content, re.DOTALL)

    if json_match:
        bbox_json = json_match.group(1).strip()
    else:
        pattern = r"```(.*?)```"
        json_match = re.search(pattern, content, re.DOTALL)
        bbox_json = json_match.group(1).strip() if json_match else None

        if not bbox_json:
            pattern = r'\[\s*{.*?"bbox_2d".*?"label".*?}\s*\]'
            json_match = re.search(pattern, content, re.DOTALL)
            bbox_json = json_match.group(0) if json_match else None

    if bbox_json:
        try:
            data = json.loads(bbox_json)
        except json.JSONDecodeError:
            try:
                repaired_json = repair_json(bbox_json)
                data = json.loads(repaired_json)
            except Exception:
                data = None

        if data and isinstance(data, list):
            try:
                ngram_size = 1
                items = []
                for item in data:
                    if "bbox_2d" in item and "label" in item:
                        items.append(f"{item['bbox_2d']}_{item['label']}")

                def zipngram_list(text_list, ngram_size):
                    return zip(*[text_list[i:] for i in range(ngram_size)])

                ngrams = set()
                total = 0

                for ng in zipngram_list(items, ngram_size):
                    ngrams.add(ng)
                    total += 1

                if total == 0:
                    return 0.0

                scaling = 1 - len(ngrams) / total
                reward = scaling * max_penalty
                return reward
            except KeyError:
                pass

    ngram_size = 6

    if len(content.split()) < ngram_size:
        return 0.0

    def zipngram_text(text, ngram_size):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    ngrams = set()
    total = 0

    for ng in zipngram_text(content, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    reward = scaling * max_penalty

    return reward


def repetition_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        reward = repetition_reward(content)
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 0.0:
                with open(log_path + "_repetition.txt", "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

    return rewards


def cosine_rewards(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        clean_content = clean_text(content)
        sol = clean_text(sol)
        if sol == "none":
            if clean_content == "none":
                acc_reward = 1.0
            else:
                acc_reward = 0.0
        else:
            acc_reward = detection_score(clean_content, sol)

        reward = cosine_reward(content, tokenizer, acc_reward)
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 1.0:
                with open(log_path + "_cosine.txt", "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

    return rewards


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except Exception:
        return None


def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)


def clean_text(text, exclue_chars=["\n", "\r"]):
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    return text.strip().rstrip(".").lower()


def all_match_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return 1.0 if content == sol else 0.0


def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0

    sol_match = re.search(r"<answer>(.*?)</answer>", sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    content_matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()

    try:
        answer = parse(student_answer)
        if float(verify(answer, parse(ground_truth))) > 0:
            reward = 1.0
    except Exception:
        pass

    if reward == 0.0:
        try:
            has_numbers = bool(re.search(r"\d", ground_truth))
            has_choices = extract_choice(ground_truth)

            if has_numbers:
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            elif has_choices:
                correct_choice = has_choices.upper()
                student_choice = extract_choice(student_answer)
                if student_choice:
                    reward = 1.0 if student_choice == correct_choice else 0.0
            else:
                reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass

    return reward


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol, accu_reward_method in zip(contents, solution, kwargs.get("accu_reward_method")):
        if accu_reward_method == "mcq":
            reward = mcq_reward(content, sol)
        elif accu_reward_method == "yes_no":
            reward = yes_no_reward(content, sol)
        elif accu_reward_method == "llm":
            reward = llm_reward(content, sol)
        elif accu_reward_method == "map":
            reward = map_reward(content, sol)
        elif accu_reward_method == "math":
            reward = math_reward(content, sol)
        elif accu_reward_method == "weighted_sum":
            clean_content = clean_text(content)
            sol = clean_text(sol)
            if sol == "none":
                if clean_content == "none":
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = detection_score(clean_content, sol)
        elif accu_reward_method == "od_ap":
            reward = od_reward(content, sol)
        elif accu_reward_method == "od_ap50":
            reward = od_reward(content, sol, score_type=1)
        elif accu_reward_method == "odLength":
            reward = odLength_reward(content, sol)
        elif accu_reward_method == "all_match":
            reward = all_match_reward(content, sol)
        else:
            reward = default_accuracy_reward(content, sol)

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 1.0:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"accu_reward_method: {accu_reward_method}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": cosine_rewards,
    "repetition": repetition_rewards,
}


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def get_vlm_module(model_name_or_path):
    if "qwen3" in model_name_or_path.lower():
        return Qwen3VLModule
    elif "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "glm" in model_name_or_path.lower():
        return GLMVModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


def main(script_args, training_args, model_args):
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.task_type)

    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in script_args.reward_funcs]
    else:
        reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    from datasets import Dataset

    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), (
            f"Number of reward methods must match number of data files: "
            f"{len(accu_reward_methods)} != {len(data_files)}"
        )

    all_data = []
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                if "image" in item:
                    if isinstance(item["image"], str):
                        item["image_path"] = [os.path.join(image_folder, item["image"])]
                        del item["image"]
                    elif isinstance(item["image"], list):
                        item["image_path"] = [os.path.join(image_folder, image) for image in item["image"]]
                        del item["image"]
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")

                # 重要修正：
                # 不要刪掉 <image>，否則 multi-image prompt 與實際 image inputs 會失配
                human_value = item["conversations"][0]["value"]
                item["problem"] = human_value if isinstance(human_value, str) else str(human_value)

                solution_value = item["conversations"][1]["value"]
                if isinstance(solution_value, str):
                    item["solution"] = solution_value.strip()
                else:
                    item["solution"] = str(solution_value)

                del item["conversations"]
                item["accu_reward_method"] = item.get("accu_reward_method", accu_reward_method)
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if "image_path" in example and example["image_path"] is not None:
            assert all(os.path.exists(p) for p in example["image_path"]), (
                f"Image paths do not exist: {example['image_path']}"
            )

            return {
                "image_path": [p for p in example["image_path"]],
                "problem": example["problem"],
                "solution": example["solution"] if "<answer>" in example["solution"] else f"<answer>{example['solution']}</answer>",
                "accu_reward_method": example["accu_reward_method"],
                "prompt": [{
                    "role": "user",
                    "content": [
                        *({"type": "image", "text": None} for _ in range(len(example["image_path"]))),
                        {"type": "text", "text": question_prompt.format(Question=example["problem"])},
                    ],
                }],
            }
        else:
            return {
                "problem": example["problem"],
                "solution": example["solution"] if "<answer>" in example["solution"] else f"<answer>{example['solution']}</answer>",
                "accu_reward_method": example["accu_reward_method"],
                "prompt": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_prompt.format(Question=example["problem"])},
                    ],
                }],
            }

    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    splits = {"train": dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(test_size=script_args.val_split_ratio)
        splits["train"] = train_val_split["train"]
        splits["validation"] = train_val_split["test"]

    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits["train"],
        eval_dataset=splits.get("validation") if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )
    print("ATTENTION IMPLEMENTATION:", trainer.model.config._attn_implementation)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print("trainable/total:", trainable, "/", total)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)