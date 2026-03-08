#!/bin/bash
#SBATCH -J grpo_qa_list
#SBATCH -p normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH -t 24:00:00
#SBATCH -o /work/foobarbaz911/vlmr1/logs/grpo_qa_list_%j.out
#SBATCH -e /work/foobarbaz911/vlmr1/logs/grpo_qa_list_%j.err
#SBATCH --account=mst114553

set -euo pipefail

WORK=/work/foobarbaz911/vlmr1
REPO=/home/foobarbaz911/VLM-R1
PKG_ROOT=/home/foobarbaz911/VLM-R1/src/open-r1-multimodal
VENV=${PKG_ROOT}/.venv310_cu124

MODEL=${WORK}/models/Qwen3-VL-8B-Instruct
DATA=${WORK}/datasets/grpo_qa_list.jsonl
IMG_ROOT=${WORK}/datasets/IAD256

OUT=/home/foobarbaz911/VLM-R1/temp/grpo_qa_list_${SLURM_JOB_ID}

mkdir -p "${WORK}/logs" "${OUT}" "${WORK}/hf_cache" "${WORK}/xdg_cache" "${WORK}/torch_cache"

########################################
# Toolchain
########################################
module purge
module load gcc/11.5.0
module load cuda/12.4

export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="$(dirname "$(gcc -print-file-name=libstdc++.so.6)"):${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

########################################
# Python env
########################################
cd "${PKG_ROOT}"
source "${VENV}/bin/activate"
cd "${REPO}"

########################################
# Runtime env
########################################
export HF_HOME=${WORK}/hf_cache
export XDG_CACHE_HOME=${WORK}/xdg_cache
export TORCH_HOME=${WORK}/torch_cache
export TRANSFORMERS_CACHE=${WORK}/hf_cache
export TOKENIZERS_PARALLELISM=false

export ATTN_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 先求穩
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

export MASTER_PORT=$((12000 + RANDOM % 20000))

########################################
# Sanity checks
########################################
echo "======================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "HOST: $(hostname)"
echo "PWD: $(pwd)"
echo "VENV: ${VENV}"
echo "OUT: ${OUT}"
echo "DATA: ${DATA}"
echo "IMG_ROOT: ${IMG_ROOT}"
echo "MODEL: ${MODEL}"
echo "--------------------------------------"
which python
python -V
which gcc
gcc --version
which g++
g++ --version
which nvcc
nvcc -V
echo "libstdc++ => $(gcc -print-file-name=libstdc++.so.6)"
echo "--------------------------------------"

python - <<'PY'
import torch, transformers, os
print("torch =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("transformers =", transformers.__version__)
print("cuda available =", torch.cuda.is_available())
print("gpu count =", torch.cuda.device_count())
PY

python - <<'PY'
import flash_attn
print("flash_attn =", flash_attn.__version__)
PY

python - <<'PY'
import deepspeed
print("deepspeed =", deepspeed.__version__)
PY

python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/work/foobarbaz911/vlmr1/models/Qwen3-VL-8B-Instruct", trust_remote_code=True)
print("model_type =", getattr(cfg, "model_type", None))
PY

echo "======================================"

########################################
# Train
########################################
deepspeed --num_gpus=4 --module open_r1.grpo_jsonl \
  --dataset_name jsonl \
  --data_file_paths "${DATA}" \
  --image_folders "${IMG_ROOT}" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUT}" \
  --do_train true \
  --do_eval false \
  --bf16 true \
  --tf32 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_generations 2 \
  --learning_rate 1e-5 \
  --max_steps 20 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 3 \
  --eval_strategy no \
  --reward_funcs accuracy format \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --deepspeed "${REPO}/ds_config_zero2.json" \
  --freeze_vision_modules true \
  --max_pixels 262144 \
  --use_peft true \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj

########################################
# Post-run summary
########################################
echo "======================================"
echo "TRAIN FINISHED"
echo "OUT DIR: ${OUT}"
echo "--------------------------------------"
find "${OUT}" -maxdepth 2 -type f | sort || true
echo "======================================"