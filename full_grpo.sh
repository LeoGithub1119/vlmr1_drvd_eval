#!/bin/bash
#SBATCH -J grpo_qa_list
#SBATCH -p normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH -t 24:00:00
#SBATCH -o /home/foobarbaz911/VLM-R1/temp/grpo_qa_list_%j.out
#SBATCH -e /home/foobarbaz911/VLM-R1/temp/grpo_qa_list_%j.err
#SBATCH --account=mst114553

set -euo pipefail

WORK=/work/foobarbaz911/vlmr1
REPO=/home/foobarbaz911/VLM-R1
PKG_ROOT=/home/foobarbaz911/VLM-R1/src/open-r1-multimodal
VENV=${PKG_ROOT}/.venv310_cu124

MODEL=${WORK}/models/Qwen3-VL-8B-Instruct
DATA=${WORK}/datasets/grpo_qa_list_fixed.jsonl
IMG_ROOT=${WORK}/datasets/IAD256

# OUT=/home/foobarbaz911/VLM-R1/temp/grpo_qa_list_${SLURM_JOB_ID}
OUT=/work/foobarbaz911/vlmr1/outputs/grpo_${SLURM_JOB_ID}
mkdir -p \
  /home/foobarbaz911/VLM-R1/temp \
  "${OUT}" \
  "${WORK}/hf_cache" \
  "${WORK}/xdg_cache" \
  "${WORK}/torch_cache"

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
export HF_HOME="${WORK}/hf_cache"
export XDG_CACHE_HOME="${WORK}/xdg_cache"
export TORCH_HOME="${WORK}/torch_cache"
export TRANSFORMERS_CACHE="${WORK}/hf_cache"
export TOKENIZERS_PARALLELISM=false

export ATTN_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 先求穩
# export NCCL_DEBUG=INFO
# 先不要亂關，除非真的確認 IB / P2P 有問題
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

export MASTER_PORT=$((15000 + (SLURM_JOB_ID % 10000)))

# export DEBUG_MODE=true
export LOG_PATH=/home/foobarbaz911/VLM-R1/temp/grpo_debug_${SLURM_JOB_ID}.txt

########################################
# Sanity checks
########################################
echo "======================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "HOST: $(hostname)"
echo "PWD: $(pwd)"
echo "VENV: ${VENV}"
echo "MODEL: ${MODEL}"
echo "DATA: ${DATA}"
echo "IMG_ROOT: ${IMG_ROOT}"
echo "OUT: ${OUT}"
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
import torch, transformers
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
cfg = AutoConfig.from_pretrained(
    "/work/foobarbaz911/vlmr1/models/Qwen3-VL-8B-Instruct",
    trust_remote_code=True
)
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
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_generations 4 \
  --learning_rate 2e-6 \
  --max_steps 5120 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 10 \
  --eval_strategy no \
  --reward_funcs accuracy format \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --deepspeed "${REPO}/ds_config_zero2.json" \
  --freeze_vision_modules true \
  --max_pixels 262144 \
  --use_peft true \
  --lora_r 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj \
  --resume_from_checkpoint "/work/foobarbaz911/vlmr1/outputs/grpo_133652/checkpoint-3320" 

########################################
# Post-run summary
########################################
echo "======================================"
echo "TRAIN FINISHED"
echo "OUT DIR: ${OUT}"
echo "--------------------------------------"
find "${OUT}" -maxdepth 2 -type f | sort || true
echo "======================================"