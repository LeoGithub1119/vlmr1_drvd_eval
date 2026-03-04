#!/bin/bash
#SBATCH --job-name=qwen3vl8b_grpo
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=mst114553
#SBATCH --output=/work/foobarbaz911/vlmr1/logs/grpo_%j.out
#SBATCH --error=/work/foobarbaz911/vlmr1/logs/grpo_%j.err


set -euo pipefail

########################
# Paths
########################

WORK=/work/foobarbaz911/vlmr1
REPO=/home/foobarbaz911/VLM-R1

DATA_GRPO=${WORK}/datasets/MMAD/mmad.jsonl
IMG_ROOT=${WORK}/datasets/MMAD
MODEL=${WORK}/models/Qwen3-VL-8B-Instruct

OUTDIR=${WORK}/outputs/grpo_full_${SLURM_JOB_ID}

mkdir -p \
${WORK}/logs \
${OUTDIR} \
${WORK}/hf_cache \
${WORK}/xdg_cache \
${WORK}/torch_cache


########################
# Conda
########################

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate base >/dev/null 2>&1 || true


########################
# Activate repo env
########################

cd ${REPO}
source activate.sh


########################
# Cache
########################

export HF_HOME=${WORK}/hf_cache
export XDG_CACHE_HOME=${WORK}/xdg_cache
export TORCH_HOME=${WORK}/torch_cache
export TOKENIZERS_PARALLELISM=false


########################
# Performance & Distributed
########################

export ATTN_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1


########################
# Debug info
########################

echo "======================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "HOST: $(hostname)"
echo "CUDA:"
nvidia-smi
echo "======================================"

python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__)"
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"


########################
# GRPO training (with DeepSpeed for 4 GPU)
########################

deepspeed --num_gpus=4 -m open_r1.grpo_jsonl \
  --dataset_name "jsonl" \
  --data_file_paths "${DATA_GRPO}" \
  --image_folders "${IMG_ROOT}" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUTDIR}" \
  --do_train true \
  --do_eval false \
  --bf16 true \
  --tf32 true \
  \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_generations 1 \
  \
  --learning_rate 1e-5 \
  --max_steps 1000 \
  \
  --logging_steps 20 \
  \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  \
  --eval_strategy "no" \
  \
  --reward_funcs "accuracy" \
  \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --deepspeed "${REPO}/ds_config_zero2.json" \
  \
  --freeze_vision_modules true \
  --max_pixels 262144 \
  \
  --use_peft true \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj


echo "======================================"
echo "GRPO training finished"
echo "Output dir: ${OUTDIR}"
echo "======================================"