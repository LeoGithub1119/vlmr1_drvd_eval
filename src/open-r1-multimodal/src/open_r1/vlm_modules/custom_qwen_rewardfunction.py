from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image

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
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the model output strictly matches the required format."""
        import re
        import os
        from datetime import datetime

        # Strict pattern required by your updated rules
        pattern = (
            r"^<think>"
            r"<obs>.*?</obs>"
            r"<evidence>.*?</evidence>"
            r"<logic>.*?</logic>"
            r"</think>"
            r"<answer>"
            r"("
                r"\[\s*\]"                                         # empty list
                r"|"
                r"\[\s*"                                           # opening bracket for list of boxes
                    r"(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]\s*,\s*)*"  # zero or more boxes with trailing comma
                    r"(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])"          # one final box
                r"\s*\]"                                            # closing bracket
            r")"
            r"</answer>$"
        )

        # Extract the content of each completion
        completion_contents = [completion[0]["content"] for completion in completions]

        # Boolean match results
        matches = [
            re.search(pattern, content, re.DOTALL) is not None
            for content in completion_contents
        ]

        # Optional debug log
        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")

        # Reward: 1 for correct format, 0 for mismatch
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def format_reward_mc(completions, sigma=0.00, **kwargs): 
        """
        Verify the model output format:

        <think>
            <obs>...</obs>
            <evidence>...</evidence>
            <logic>...</logic>
        </think>
        <answer>...</answer>
        <location>...</location>

        Return:
            Smoothed reward: 1.0 + N(0, sigma) or 0.0 + N(0, sigma), clamped to [0, 1]
        """

        import re
        import os
        import numpy as np
        from datetime import datetime

        rewards = []

        # Patterns
        think_pattern = r"<think>(.*?)</think>"
        obs_pattern = r"<obs>(.*?)</obs>"
        evidence_pattern = r"<evidence>(.*?)</evidence>"
        logic_pattern = r"<logic>(.*?)</logic>"
        answer_pattern = r"<answer>(.*?)</answer>"
        location_pattern = r"<location>(.*?)</location>"

        completion_contents = [completion[0]["content"] for completion in completions]

        for content in completion_contents:

            base_reward = 1.0  # assume correct until proven wrong

            # ---------------------------
            # 1. Check <think>...</think>
            # ---------------------------
            think_match = re.search(think_pattern, content, re.DOTALL)
            if not think_match:
                base_reward = 0.0
            else:
                think_block = think_match.group(1)

                # ---------------------------
                # 2. Check required subtags inside <think>
                # ---------------------------
                if not re.search(obs_pattern, think_block, re.DOTALL):
                    base_reward = 0.0
                elif not re.search(evidence_pattern, think_block, re.DOTALL):
                    base_reward = 0.0
                elif not re.search(logic_pattern, think_block, re.DOTALL):
                    base_reward = 0.0

                # ---------------------------
                # 3. Check <answer> and <location> are outside </think>
                # ---------------------------
                if base_reward == 1.0:

                    answer_match = re.search(answer_pattern, content, re.DOTALL)
                    location_match = re.search(location_pattern, content, re.DOTALL)

                    if not answer_match or not location_match:
                        base_reward = 0.0
                    else:
                        think_end = think_match.end()

                        # answer must be outside think
                        if answer_match.start() < think_end:
                            base_reward = 0.0

                        # location must be outside think
                        if location_match.start() < think_end:
                            base_reward = 0.0

            # ---------------------------
            # 4. Add Gaussian smoothing
            # ---------------------------
            noise = np.random.normal(0, sigma)
            reward = base_reward + noise

            # Clamp reward to [0,1]
            reward = max(0.0, min(1.0, reward))

            # print("format:", reward)
            rewards.append(reward)

        # ---------------------------
        # Optional debug logging
        # ---------------------------
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "reward_debug.txt")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"----------- {current_time} Format MC reward -----------\n")
                for content, reward in zip(completion_contents, rewards):
                    f.write(f"Content:\n{content}\nReward: {reward}\n\n")
        return rewards

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        import os
        import re
        import json
        import numpy as np
        from datetime import datetime
        from PIL import Image
        from scipy.optimize import linear_sum_assignment

        # ----------------------------
        # Helper functions
        # ----------------------------

        def iou(box1, box2):
            """
            box format: [x1, y1, x2, y2], continuous coordinates
            """
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
            if union <= 0:
                return 0.0

            return inter / union

        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            x1, y1, x2, y2 = bbox
            return [
                x1 / input_width  * image_width,
                y1 / input_height * image_height,
                x2 / input_width  * image_width,
                y2 / input_height * image_height
            ]

        def sanitize_bbox(b):
            x1, y1, x2, y2 = b
            return [
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2)
            ]

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

            # Unmatched GT → IoU = 0
            if len(matched_ious) < num_gt:
                matched_ious.extend([0.0] * (num_gt - len(matched_ious)))

            return float(np.mean(matched_ious))

        # ----------------------------
        # Main reward logic
        # ----------------------------

        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        answer_tag_pattern = r"<answer>(.*?)</answer>"

        for i, (content, sol_str) in enumerate(zip(contents, solution)):
            reward = 0.0

            try:
                # ---- Load image info ----
                image_grid_thw = kwargs.get("image_grid_thw")[i]
                image_path = kwargs.get("image_path")[i][0]

                image = Image.open(image_path)
                image_width, image_height = image.size

                # grid format assumed: [T, H, W]
                input_height = int(image_grid_thw[1] * 14)
                input_width  = int(image_grid_thw[2] * 14)

                # ---- Parse GT boxes ----
                sol_match = re.findall(answer_tag_pattern, sol_str, re.DOTALL)
                if sol_match:
                    try:
                        gt_boxes = json.loads(sol_match[-1].strip())
                    except Exception:
                        gt_boxes = []
                else:
                    gt_boxes = []

                # ---- Parse predicted boxes ----
                content_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_match:
                    try:
                        pred_boxes = json.loads(content_match.group(1).strip())
                    except Exception:
                        pred_boxes = []
                else:
                    pred_boxes = []

                # ---- Resize + sanitize predicted boxes ----
                processed_preds = []
                for b in pred_boxes:
                    b = resize_bbox(b, input_height, input_width, image_height, image_width)
                    b = sanitize_bbox(b)
                    processed_preds.append(b)

                gt_boxes = [sanitize_bbox(b) for b in gt_boxes]

                # ---- Compute reward with Hungarian matching ----
                # print(gt_boxes, processed_preds)
                reward = compute_hungarian_iou_reward(gt_boxes, processed_preds)

            except Exception:
                reward = 0.0

            rewards.append(reward)

            # ---- Debug logging ----
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                problem = kwargs.get("problem")[i] if "problem" in kwargs else None

                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Ground truth: {gt_boxes}\n")

        return rewards

    @staticmethod
    def multiple_choice_reward(completions, solution, sigma=0.00, **kwargs):
        """
        Reward = 1.0 if model <answer> matches ground-truth choice
            = 0.0 otherwise
        Add Gaussian noise and clamp to [0,1]
        """

        import re
        import numpy as np
        import ast

        rewards = []
        answer_pattern = r"<answer>(.*?)</answer>"

        for completion, sol in zip(completions, solution):
            content = completion[0]["content"]
            # print("model content:", content, "ground-truth sol:", sol)

            # -------------------------
            # 1️⃣ Extract model answer
            # -------------------------
            model_match = re.search(answer_pattern, content, re.DOTALL | re.IGNORECASE)
            if not model_match:
                base_reward = 0.0
                model_choice = None
            else:
                model_raw = model_match.group(1).strip().upper()
                model_choices = re.findall(r"\b[A-D]\b", model_raw)
                model_choice = model_choices[0] if model_choices else None

            # -------------------------
            # 2️⃣ Extract ground-truth answer
            # -------------------------
            sol_choice = None
            try:
                # 去掉 <answer> ... </answer> 標籤
                sol_text_match = re.search(answer_pattern, sol, re.DOTALL | re.IGNORECASE)
                if sol_text_match:
                    sol_text = sol_text_match.group(1).strip()
                else:
                    sol_text = sol.strip()

                # 如果是列表字串
                if sol_text.startswith("["):
                    sol_list = ast.literal_eval(sol_text)
                    if sol_list:
                        sol_choice = str(sol_list[0]).strip().upper()
                else:
                    sol_choice = str(sol_text).strip().upper()

            except Exception as e:
                print("Warning: failed to parse solution:", sol, e)
                sol_choice = None

            # print("model_choice:", model_choice, "sol_choice:", sol_choice)

            # -------------------------
            # 3️⃣ Compute reward
            # -------------------------
            if model_choice and sol_choice and model_choice == sol_choice:
                base_reward = 1.0
            else:
                base_reward = 0.0

            # -------------------------
            # 4️⃣ Add Gaussian noise
            # -------------------------
            noise = np.random.normal(0, sigma)
            reward = base_reward + noise
            reward = max(0.0, min(1.0, reward))

            # print("mc:", reward)
            rewards.append(reward)

        return rewards

    @staticmethod
    def location_reward(completions, solution, sigma=0.00, **kwargs):
        import re
        import numpy as np
        import ast

        rewards = []

        location_pattern = r"<location>(.*?)</location>"
        answer_pattern = r"<answer>(.*?)</answer>"

        for completion, sol in zip(completions, solution):

            content = completion[0]["content"]

            # -------------------------
            # 1️⃣ Extract model location
            # -------------------------
            loc_match = re.search(location_pattern, content, re.DOTALL | re.IGNORECASE)
            if loc_match:
                model_loc_raw = loc_match.group(1).strip()
                model_set = set(c for c in model_loc_raw if c in "123456789")
            else:
                model_set = None

            # -------------------------
            # 2️⃣ Extract ground-truth location
            # -------------------------
            truth_set = None  # 初始化，避免 UnboundLocalError
            if model_set is not None:
                try:
                    if isinstance(sol, str) and sol.strip().startswith("["):
                        sol_list = ast.literal_eval(sol)
                        truth_raw = str(sol_list[1]).strip() if len(sol_list) > 1 else ""
                    else:
                        truth_raw = str(sol).strip()

                    answer_match = re.search(answer_pattern, truth_raw, re.DOTALL | re.IGNORECASE)
                    if answer_match:
                        truth_raw = answer_match.group(1).strip()

                    truth_set = set(c for c in truth_raw if c in "123456789")
                except Exception as e:
                    print("Warning parsing solution:", sol, e)
                    truth_set = None

            # -------------------------
            # 3️⃣ Compute IoU
            # -------------------------
            base_reward = 0.0
            if model_set is not None and truth_set is not None:
                intersection = len(model_set & truth_set)
                union = len(model_set | truth_set)
                base_reward = 1.0 if union == 0 else intersection / union

            # -------------------------
            # 4️⃣ Add Gaussian noise
            # -------------------------
            noise = np.random.normal(0, sigma)
            reward = max(0.0, min(1.0, base_reward + noise))

            # print("model_set:", model_set, "truth_set:", truth_set, "location reward:", reward)
            rewards.append(reward)

        return rewards    

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case "mc":  # 新增 multiple choice task type
                    return Qwen2VLModule.multiple_choice_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "mc":
                    return Qwen2VLModule.format_reward_mc
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "location":
            match task_type:
                case "mc":
                    return Qwen2VLModule.location_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
