# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Supervised fine-tuning script for vision-language models.
# Patched for Qwen3-VL support and absolute image paths.

import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default="", metadata={"help": "Root directory for images if dataset stores relative paths."})


processor = None


def resolve_image_path(raw_path: str, image_root: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    if image_root is None:
        image_root = ""
    return os.path.join(image_root, raw_path)


def load_vl_model(model_name_or_path: str, model_kwargs: dict):
    """
    Mirror the GRPO-side model selection logic:
    - Qwen2-VL
    - Qwen2.5-VL
    - Qwen3-VL
    Also remove use_cache for Qwen3-VL, same as your GRPO trainer path.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", None)

    kwargs = dict(model_kwargs)

    if model_type == "qwen3_vl" or "Qwen3-VL" in model_name_or_path:
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is unavailable. "
                "Please upgrade transformers to a version that supports Qwen3-VL."
            )
        kwargs.pop("use_cache", None)
        return Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)

    if model_type == "qwen2_5_vl" or "Qwen2.5-VL" in model_name_or_path:
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Qwen2_5_VLForConditionalGeneration is unavailable in current transformers.")
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)

    if model_type == "qwen2_vl" or "Qwen2-VL" in model_name_or_path:
        if Qwen2VLForConditionalGeneration is None:
            raise ImportError("Qwen2VLForConditionalGeneration is unavailable in current transformers.")
        return Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)

    raise ValueError(f"Unsupported model: {model_name_or_path} with type {model_type}")



class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: ScriptArguments):
        super().__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}. Expected a YAML dataset config.")

        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets_cfg = yaml_data.get("datasets", [])

        for data in datasets_cfg:
            json_path = data.get("json_path")
            sampling_strategy = data.get("sampling_strategy", "all")
            sampling_number = None

            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(json_path, "r") as json_file:
                    for line in json_file:
                        line = line.strip()
                        if line:
                            cur_data_dict.append(json.loads(line))
            elif json_path.endswith(".json"):
                with open(json_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            else:
                raise ValueError(f"Unsupported file type: {json_path}")

            if ":" in sampling_strategy:
                sampling_strategy, sampling_number = sampling_strategy.split(":")
                if "%" in sampling_number:
                    sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                else:
                    sampling_number = int(sampling_number)

            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-sampling_number:]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "all":
                pass
            else:
                if sampling_number is None and sampling_strategy != "all":
                    logger.warning(f"Unknown sampling strategy without count: {sampling_strategy}, fallback to all.")

            logger.info(f"Loaded {len(cur_data_dict)} samples from {json_path}")
            self.list_data_dict.extend(cur_data_dict)

    def __len__(self):
        return len(self.list_data_dict)

    # def __getitem__(self, i):
    #     example = self.list_data_dict[i]
    #     image_root = self.script_args.image_root or ""
    #     image_path = resolve_image_path(example["image"], image_root)

    #     x1, y1, x2, y2 = example["solution"]
    #     normal_caption = example["normal_caption"]

    #     example = dict(example)
    #     example["messages"] = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": f"file://{image_path}"},
    #                 {"type": "text", "text": example["problem"]},
    #             ],
    #         },
    #         {
    #             "role": "assistant",
    #             "content": (
    #                 "```json\n[\n\t"
    #                 f'{{"bbox_2d": [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], "label": "{normal_caption}"}}'
    #                 "\n]\n```"
    #             ),
    #         },
    #     ]
    #     example["resolved_image_path"] = image_path
    #     return example
    def __getitem__(self, i):
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root or ""

        # Case 1: new chat-style dataset
        # format:
        # {
        #   "images": ["/abs/path/or/relative/path.jpg", ...],
        #   "messages": [
        #       {"role": "user", "content": "<image>...."},
        #       {"role": "assistant", "content": "..."}
        #   ]
        # }
        if "messages" in example:
            messages = example["messages"]
            images = example.get("images", [])

            # normalize image paths
            resolved_images = []
            for raw_img in images:
                if os.path.isabs(raw_img):
                    resolved = raw_img
                else:
                    resolved = os.path.join(image_root, raw_img)
                resolved_images.append(f"file://{resolved}")

            converted_messages = []
            img_idx = 0

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                # user string content with <image> placeholders
                if role == "user" and isinstance(content, str):
                    parts = content.split("<image>")
                    blocks = []

                    for j, part in enumerate(parts):
                        # insert image before every segment after first split
                        if j > 0:
                            if img_idx >= len(resolved_images):
                                raise ValueError(
                                    f"Sample {i} has more <image> placeholders than actual images. "
                                    f"placeholders_seen={img_idx+1}, images={len(resolved_images)}"
                                )
                            blocks.append({"type": "image", "image": resolved_images[img_idx]})
                            img_idx += 1

                        if part.strip():
                            blocks.append({"type": "text", "text": part.strip()})

                    converted_messages.append({"role": role, "content": blocks})

                # already block-style content
                elif role == "user" and isinstance(content, list):
                    normalized_blocks = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            raw_img = item.get("image", "")
                            if raw_img.startswith("file://"):
                                raw_img = raw_img[len("file://"):]
                            if os.path.isabs(raw_img):
                                resolved = raw_img
                            else:
                                resolved = os.path.join(image_root, raw_img)
                            normalized_blocks.append({"type": "image", "image": f"file://{resolved}"})
                        else:
                            normalized_blocks.append(item)
                    converted_messages.append({"role": role, "content": normalized_blocks})

                # assistant / others keep as plain text
                else:
                    converted_messages.append({"role": role, "content": content})

            example = dict(example)
            example["messages"] = converted_messages
            return example

        # Case 2: old bbox-style dataset
        # format:
        # {
        #   "image": "...",
        #   "problem": "...",
        #   "solution": [x1,y1,x2,y2],
        #   "normal_caption": "..."
        # }
        raw_img = example["image"]
        if os.path.isabs(raw_img):
            image_path = raw_img
        else:
            image_path = os.path.join(image_root, raw_img)

        x1, y1, x2, y2 = example["solution"]
        normal_caption = example["normal_caption"]

        example = dict(example)
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": example["problem"]},
                ],
            },
            {
                "role": "assistant",
                "content": (
                    "```json\n[\n\t"
                    f'{{"bbox_2d": [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], "label": "{normal_caption}"}}'
                    "\n]\n```"
                ),
            },
        ]
        return example

def _find_image_token_ids(proc) -> set:
    token_ids = set()

    tokenizer = getattr(proc, "tokenizer", proc)

    image_token = getattr(proc, "image_token", None)
    if image_token is not None:
        try:
            tid = tokenizer.convert_tokens_to_ids(image_token)
            if tid is not None and tid >= 0:
                token_ids.add(tid)
        except Exception:
            pass

    image_token_id = getattr(proc, "image_token_id", None)
    if image_token_id is not None and image_token_id >= 0:
        token_ids.add(image_token_id)

    # Qwen3-VL may use vision special tokens in prompt text
    for tok in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                token_ids.add(tid)
        except Exception:
            pass

    return token_ids


def collate_fn(examples):
    global processor

    texts = [
        processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for example in examples
    ]

    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)

    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()

    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    for image_token_id in _find_image_token_ids(processor):
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f", distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at last_checkpoint={last_checkpoint}")

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    logger.info(f"Total loaded samples: {len(dataset)}")

    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
        )
        logger.info("Using AutoTokenizer for text-only model.")

    if hasattr(processor, "pad_token") and getattr(processor, "pad_token", None) is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = load_vl_model(model_args.model_name_or_path, model_kwargs)

    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        # Some TRL versions do not accept finetuned_from / dataset / dataset_tags kwargs
        try:
            trainer.create_model_card()
        except Exception as e:
            logger.warning(f"Skipping model card creation due to: {e}")

        if hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)