from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
import re
import os
import json
import numpy as np
from datetime import datetime
import ast
from scipy.optimize import linear_sum_assignment

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "Qwen3-VL" in model_id:
            if Qwen3VLForConditionalGeneration is not None:
                model_cls = Qwen3VLForConditionalGeneration
            else:
                raise ImportError(
                    "Qwen3VLForConditionalGeneration not available. "
                    "Please upgrade transformers >= 4.57.0 with: pip install --upgrade transformers"
                )
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    # def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
    #     # FIXME
    #     # This could only process pure-multimodal or pure-text inputs
    #     additional_output = None
    #     if len(images) > 0:
    #         prompt_inputs = processing_class(
    #             text=prompts_text,
    #             images=images,
    #             return_tensors=return_tensors,
    #             padding=padding,
    #             padding_side=padding_side,
    #             add_special_tokens=add_special_tokens)
    #         additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
    #     else:
    #         prompt_inputs = processing_class(
    #             text=prompts_text,
    #             return_tensors=return_tensors,
    #             padding=padding,
    #             padding_side=padding_side,
    #             add_special_tokens=add_special_tokens)
    #     return prompt_inputs, additional_output
    


    def prepare_model_inputs(
        self, processing_class, prompts_text, images,
        return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    ):
        additional_output = None

        # --------- helper ---------
        def _debug_prompt(i: int):
            p = prompts_text[i] if i < len(prompts_text) else ""
            return p[:220].replace("\n", "\\n")

        # --------- no image case ---------
        if images is None or len(images) == 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
            return prompt_inputs, additional_output

        # --------- normalize images to list-of-lists aligned with prompts ---------
        # case A: already nested (per-sample)
        is_nested = isinstance(images[0], (list, tuple))

        if not is_nested:
            # flat list of images -> split by <image> counts per prompt
            counts = [p.count("<image>") for p in prompts_text]
            if len(counts) == 1:
                # single sample: allow flat -> wrap
                images = [images]
            else:
                grouped = []
                idx = 0
                for i, c in enumerate(counts):
                    if c <= 0:
                        raise ValueError(
                            f"[Qwen2VLModule] prompt[{i}] has 0 <image> tokens but got flat images list. "
                            f"prompt[:220]={_debug_prompt(i)}"
                        )
                    chunk = images[idx: idx + c]
                    if len(chunk) != c:
                        raise ValueError(
                            f"[Qwen2VLModule] not enough images for prompt[{i}]: need {c}, got {len(chunk)}. "
                            f"idx={idx}, total_images={len(images)}. prompt[:220]={_debug_prompt(i)}"
                        )
                    grouped.append(chunk)
                    idx += c
                if idx != len(images):
                    # leftover images that didn't get assigned
                    raise ValueError(
                        f"[Qwen2VLModule] leftover images after grouping: used {idx}, total {len(images)}. "
                        f"counts={counts}"
                    )
                images = grouped
        else:
            # nested images -> validate each prompt has matching number of <image> tokens
            counts = [p.count("<image>") for p in prompts_text]
            if len(images) != len(prompts_text):
                raise ValueError(
                    f"[Qwen2VLModule] nested images batch mismatch: len(images)={len(images)} "
                    f"len(prompts_text)={len(prompts_text)}"
                )
            for i, (c, im_list) in enumerate(zip(counts, images)):
                if c <= 0:
                    raise ValueError(
                        f"[Qwen2VLModule] prompt[{i}] has 0 <image> tokens but received images[{i}] "
                        f"with len={len(im_list)}. prompt[:220]={_debug_prompt(i)}"
                    )
                if len(im_list) != c:
                    raise ValueError(
                        f"[Qwen2VLModule] prompt[{i}] <image> count={c} but images[{i}] len={len(im_list)}. "
                        f"prompt[:220]={_debug_prompt(i)}"
                    )

        # extra guard: ensure no empty list
        for i, im_list in enumerate(images):
            if len(im_list) == 0:
                raise ValueError(
                    f"[Qwen2VLModule] images[{i}] is empty after grouping. prompt[:220]={_debug_prompt(i)}"
                )

        # --------- call processor ---------
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens
        )

        # --------- align image_grid_thw per-sample ---------
        grids = prompt_inputs.get("image_grid_thw", None)
        if grids is None:
            additional_output = [None] * len(prompts_text)
        else:
            if torch.is_tensor(grids):
                # grids shape: (total_images, 3)
                counts = [len(im_list) for im_list in images]
                out = []
                idx = 0
                for c in counts:
                    out.append({"image_grid_thw": grids[idx: idx + c]})
                    idx += c
                additional_output = out
            else:
                additional_output = [{"image_grid_thw": g} for g in grids]

        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "mc":
                return "{Question} First output the thinking process in <think><obs>...</obs><evidence>...</evidence><logic>...</logic></think> tags and then output the final answer in <answer> </answer> tags. Also output defect locations in <location> </location> tags."
            case "odLength":
                return "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\n{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the model output strictly matches the required format."""
        pattern = (
            r"^<think>"
            r"<obs>.*?</obs>"
            r"<evidence>.*?</evidence>"
            r"<logic>.*?</logic>"
            r"</think>"
            r"<answer>"
            r"("
                r"\[\s*\]"
                r"|"
                r"\[\s*"
                    r"(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*,\s*)*"
                    r"(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])"
                r"\s*\]"
            r")"
            r"</answer>$"
        )
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        
        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Format reward (rec) -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def format_reward_mc(completions, sigma=0.00, **kwargs):
        """Verify MC format: <think> with <obs>, <evidence>, <logic> </think> <answer> <location>"""
        rewards = []
        think_pattern = r"<think>(.*?)</think>"
        obs_pattern = r"<obs>(.*?)</obs>"
        evidence_pattern = r"<evidence>(.*?)</evidence>"
        logic_pattern = r"<logic>(.*?)</logic>"
        answer_pattern = r"<answer>(.*?)</answer>"
        location_pattern = r"<location>(.*?)</location>"
        completion_contents = [completion[0]["content"] for completion in completions]

        for content in completion_contents:
            base_reward = 1.0
            think_match = re.search(think_pattern, content, re.DOTALL)
            if not think_match:
                base_reward = 0.0
            else:
                think_block = think_match.group(1)
                if not re.search(obs_pattern, think_block, re.DOTALL):
                    base_reward = 0.0
                elif not re.search(evidence_pattern, think_block, re.DOTALL):
                    base_reward = 0.0
                elif not re.search(logic_pattern, think_block, re.DOTALL):
                    base_reward = 0.0

                if base_reward == 1.0:
                    answer_match = re.search(answer_pattern, content, re.DOTALL)
                    location_match = re.search(location_pattern, content, re.DOTALL)
                    if not answer_match or not location_match:
                        base_reward = 0.0
                    else:
                        think_end = think_match.end()
                        if answer_match.start() < think_end or location_match.start() < think_end:
                            base_reward = 0.0

            noise = np.random.normal(0, sigma)
            reward = max(0.0, min(1.0, base_reward + noise))
            rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "reward_debug.txt")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"----------- {current_time} Format MC reward -----------\n")
                for content, reward in zip(completion_contents, rewards):
                    f.write(f"Content:\n{content}\nReward: {reward}\n\n")
        return rewards

    @staticmethod
    def format_reward(completions, **kwargs):
        """Check if the model output matches required format."""
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward using Hungarian matching."""
        def iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            inter = inter_w * inter_h
            area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
            area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
            union = area1 + area2 - inter
            return 0.0 if union <= 0 else inter / union

        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            return [
                bbox[0] / input_width * image_width,
                bbox[1] / input_height * image_height,
                bbox[2] / input_width * image_width,
                bbox[3] / input_height * image_height
            ]

        def sanitize_bbox(b):
            x1, y1, x2, y2 = b
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

        def compute_hungarian_iou_reward(gt_boxes, pred_boxes):
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                return 1.0
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                return 0.0
            num_gt = len(gt_boxes)
            num_pred = len(pred_boxes)
            cost_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
            for i, gt in enumerate(gt_boxes):
                for j, pred in enumerate(pred_boxes):
                    cost_matrix[i, j] = -iou(gt, pred)
            gt_idx, pred_idx = linear_sum_assignment(cost_matrix)
            matched_ious = [-cost_matrix[i, j] for i, j in zip(gt_idx, pred_idx)]
            if len(matched_ious) < num_gt:
                matched_ious.extend([0.0] * (num_gt - len(matched_ious)))
            return float(np.mean(matched_ious))

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        answer_tag_pattern = r"<answer>(.*?)</answer>"

        for i, (content, sol_str) in enumerate(zip(contents, solution)):
            reward = 0.0
            try:
                image_grid_thw = kwargs.get("image_grid_thw")[i]
                image_path = kwargs.get("image_path")[i][0]
                image = Image.open(image_path)
                image_width, image_height = image.size
                input_height = int(image_grid_thw[1] * 14)
                input_width = int(image_grid_thw[2] * 14)

                sol_match = re.findall(answer_tag_pattern, sol_str, re.DOTALL)
                gt_boxes = json.loads(sol_match[-1].strip()) if sol_match else []

                content_match = re.search(answer_tag_pattern, content, re.DOTALL)
                pred_boxes = json.loads(content_match.group(1).strip()) if content_match else []

                processed_preds = []
                for b in pred_boxes:
                    b = resize_bbox(b, input_height, input_width, image_height, image_width)
                    b = sanitize_bbox(b)
                    processed_preds.append(b)
                gt_boxes = [sanitize_bbox(b) for b in gt_boxes]
                reward = compute_hungarian_iou_reward(gt_boxes, processed_preds)
            except Exception:
                reward = 0.0
            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                problem = kwargs.get("problem")[i] if "problem" in kwargs else None
                image_path = kwargs.get("image_path")[i][0] if "image_path" in kwargs else None
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
        return rewards

    @staticmethod
    def multiple_choice_reward(completions, solution, sigma=0.00, **kwargs):
        """Reward for multiple choice: 1.0 if choice matches, 0.0 otherwise"""
        rewards = []
        answer_pattern = r"<answer>(.*?)</answer>"

        for completion, sol in zip(completions, solution):
            content = completion[0]["content"]
            model_match = re.search(answer_pattern, content, re.DOTALL | re.IGNORECASE)
            if not model_match:
                base_reward = 0.0
                model_choice = None
            else:
                model_raw = model_match.group(1).strip().upper()
                model_choices = re.findall(r"\b[A-D]\b", model_raw)
                model_choice = model_choices[0] if model_choices else None

            sol_choice = None
            try:
                sol_text_match = re.search(answer_pattern, sol, re.DOTALL | re.IGNORECASE)
                sol_text = sol_text_match.group(1).strip() if sol_text_match else sol.strip()
                if sol_text.startswith("["):
                    sol_list = ast.literal_eval(sol_text)
                    sol_choice = str(sol_list[0]).strip().upper() if sol_list else None
                else:
                    sol_choice = str(sol_text).strip().upper()
            except Exception:
                sol_choice = None

            if model_choice and sol_choice and model_choice == sol_choice:
                base_reward = 1.0
            else:
                base_reward = 0.0

            noise = np.random.normal(0, sigma)
            reward = max(0.0, min(1.0, base_reward + noise))
            rewards.append(reward)
        return rewards

    @staticmethod
    def location_reward(completions, solution, sigma=0.00, **kwargs):
        """Calculate location IoU reward"""
        rewards = []
        location_pattern = r"<location>(.*?)</location>"
        answer_pattern = r"<answer>(.*?)</answer>"

        for completion, sol in zip(completions, solution):
            content = completion[0]["content"]
            loc_match = re.search(location_pattern, content, re.DOTALL | re.IGNORECASE)
            model_set = set(c for c in loc_match.group(1).strip() if c in "123456789") if loc_match else None

            truth_set = None
            if model_set is not None:
                try:
                    sol_list = ast.literal_eval(sol) if sol.strip().startswith("[") else [sol]
                    truth_raw = str(sol_list[1]).strip() if len(sol_list) > 1 else str(sol).strip()
                    answer_match = re.search(answer_pattern, truth_raw, re.DOTALL | re.IGNORECASE)
                    truth_raw = answer_match.group(1).strip() if answer_match else truth_raw
                    truth_set = set(c for c in truth_raw if c in "123456789")
                except Exception:
                    truth_set = None

            base_reward = 0.0
            if model_set is not None and truth_set is not None:
                intersection = len(model_set & truth_set)
                union = len(model_set | truth_set)
                base_reward = 1.0 if union == 0 else intersection / union

            noise = np.random.normal(0, sigma)
            reward = max(0.0, min(1.0, base_reward + noise))
            rewards.append(reward)
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        """Select reward function based on task and reward type.
        
        Args:
            func: Type of reward ('accuracy', 'format', 'location')
            task_type: Type of task ('rec' for object detection, 'mc' for multiple choice)
        
        Returns:
            Reward function callable
        """
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case "mc":
                    return Qwen2VLModule.multiple_choice_reward
                case _:
                    raise ValueError(f"Unsupported task_type for accuracy reward: {task_type}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "mc":
                    return Qwen2VLModule.format_reward_mc
                case _:
                    raise ValueError(f"Unsupported task_type for format reward: {task_type}")
        elif func == "location":
            match task_type:
                case "mc":
                    return Qwen2VLModule.location_reward
                case _:
                    raise ValueError(f"Unsupported task_type for location reward: {task_type}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")


class Qwen3VLModule(Qwen2VLModule):
    """Module for Qwen3-VL models with different prompt and image handling."""
    
    def get_vlm_key(self):
        return "qwen3"
    
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        """Qwen3-VL uses Qwen3VLForConditionalGeneration class."""
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration not available. "
                "Please upgrade transformers >= 4.57.0 with: pip install --upgrade transformers"
            )
        return Qwen3VLForConditionalGeneration
    
    def prepare_model_inputs(
        self, processing_class, prompts_text, images,
        return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    ):
        """
        Qwen3-VL format handling.
        Qwen3-VL uses vision tokens like <|vision_start|><|image_pad|><|vision_end|> in the prompt.
        We count these to determine how many images each prompt expects.
        """
        additional_output = None

        # --------- no image case ---------
        if images is None or len(images) == 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
            return prompt_inputs, additional_output

        # --------- count Qwen3-style vision tokens in each prompt ---------
        # Qwen3-VL uses: <|vision_start|><|image_pad|><|vision_end|> for each image
        def count_vision_blocks(p: str) -> int:
            """Count instances of vision blocks in Qwen3 format."""
            return p.count("<|vision_start|>")
        
        counts = [count_vision_blocks(p) for p in prompts_text]
        
        # --------- normalize images to list-of-lists aligned with prompts ---------
        is_nested = isinstance(images[0], (list, tuple))

        if not is_nested:
            # flat list of images -> group by counts
            if len(counts) == 1:
                # single prompt: wrap all images
                images = [images]
            else:
                # multiple prompts: distribute images according to vision token counts
                grouped = []
                idx = 0
                for i, c in enumerate(counts):
                    if c <= 0:
                        # No images needed for this prompt
                        grouped.append([])
                    else:
                        chunk = images[idx: idx + c]
                        if len(chunk) != c:
                            raise ValueError(
                                f"[Qwen3VLModule] not enough images for prompt[{i}]: need {c}, got {len(chunk)}. "
                                f"idx={idx}, total_images={len(images)}"
                            )
                        grouped.append(chunk)
                        idx += c
                if idx != len(images):
                    raise ValueError(
                        f"[Qwen3VLModule] leftover images after grouping: used {idx}, total {len(images)}. "
                        f"vision_block_counts={counts}"
                    )
                images = grouped
        else:
            # nested images -> validate batch size matches
            if len(images) != len(prompts_text):
                raise ValueError(
                    f"[Qwen3VLModule] nested images batch mismatch: len(images)={len(images)} "
                    f"len(prompts_text)={len(prompts_text)}"
                )
            # Validate that each prompt's vision block count matches image list length
            for i, (c, im_list) in enumerate(zip(counts, images)):
                if c > 0 and len(im_list) != c:
                    raise ValueError(
                        f"[Qwen3VLModule] prompt[{i}] vision blocks={c} but images[{i}] len={len(im_list)}"
                    )

        # --------- call processor ---------
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens
        )

        # --------- align image_grid_thw per-sample if present ---------
        grids = prompt_inputs.get("image_grid_thw", None)
        if grids is not None:
            if torch.is_tensor(grids):
                # grids might be flattened; align with image batch structure
                counts_img = [len(im_list) for im_list in images]
                out = []
                idx = 0
                for c in counts_img:
                    if c > 0:
                        out.append({"image_grid_thw": grids[idx: idx + c]})
                    else:
                        out.append(None)
                    idx += c
                additional_output = out
            else:
                # If grids is already a list
                additional_output = [{"image_grid_thw": g} if g is not None else None for g in grids]
        
        return prompt_inputs, additional_output
