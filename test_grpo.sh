#!/bin/bash
set -euo pipefail

WORK=/work/foobarbaz911/vlmr1
REPO=/home/foobarbaz911/VLM-R1
DATA_GRPO=${WORK}/datasets/MMAD/mmad.jsonl
IMG_ROOT=${WORK}/datasets/MMAD           # 這裡底下有 DS-MVTec/...
MODEL=${WORK}/models/Qwen3-VL-8B-Instruct
OUTDIR=${WORK}/outputs/grpo_smoke_${SLURM_JOB_ID}

mkdir -p ${WORK}/logs ${OUTDIR} ${WORK}/hf_cache ${WORK}/xdg_cache ${WORK}/torch_cache

# conda functions for activate.sh
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate base >/dev/null 2>&1 || true

cd ${REPO}
source activate.sh

export HF_HOME=${WORK}/hf_cache
export XDG_CACHE_HOME=${WORK}/xdg_cache
export TORCH_HOME=${WORK}/torch_cache
export TOKENIZERS_PARALLELISM=false

# speed
export ATTN_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=flashinfer  # 如果使用 vLLM
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== JOB ${SLURM_JOB_ID} on $(hostname) ==="
python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__)"
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"

# ====== GRPO smoke ======
# 你剛剛錯誤是 model_name_or_path 沒傳 → 這裡補上
# 另外 data_file_paths / image_folders 用你本地 work 路徑
python -m open_r1.grpo_jsonl \
  --dataset_name "jsonl" \
  --data_file_paths "${DATA_GRPO}" \
  --image_folders "${IMG_ROOT}" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUTDIR}" \
  --do_train true \
  --do_eval false \
  --bf16 true \
  --tf32 true \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_generations 2 \
  --learning_rate 3e-6 \
  --max_steps 3 \
  --logging_steps 1 \
  --save_strategy "no" \
  --eval_strategy "no" \
  --reward_funcs "accuracy" \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --freeze_vision_modules true \
  --max_pixels 262144 \
  --use_peft true \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj
 