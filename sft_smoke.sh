#!/bin/bash
#SBATCH -J sft_qwen3_smoke
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 0:20:00
#SBATCH --account=mst114553
#SBATCH -o /home/foobarbaz911/VLM-R1/temp//sft_qwen3_smoke_%j.out
#SBATCH -e /home/foobarbaz911/VLM-R1/temp/sft_qwen3_smoke_%j.err

set -euo pipefail

WORK=/work/foobarbaz911/vlmr1
REPO=/home/foobarbaz911/VLM-R1
MODEL=${WORK}/models/Qwen3-VL-8B-Instruct
OUTDIR=${WORK}/outputs/sft_qwen3_smoke_${SLURM_JOB_ID}

mkdir -p \
  ${WORK}/logs \
  ${OUTDIR} \
  ${WORK}/hf_cache \
  ${WORK}/xdg_cache \
  ${WORK}/torch_cache

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
export TRANSFORMERS_CACHE=${HF_HOME}
export TOKENIZERS_PARALLELISM=false
export ATTENTION_IMPL=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== JOB ${SLURM_JOB_ID} on $(hostname) ==="
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
python - <<'PY'
try:
    from transformers import Qwen3VLForConditionalGeneration
    print("Qwen3VLForConditionalGeneration: OK")
except Exception as e:
    print("Qwen3VLForConditionalGeneration import failed:", repr(e))
    raise
PY

python -m open_r1.sft \
  --dataset_name "${REPO}/sft_config.yaml" \
  --image_root "" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUTDIR}" \
  --do_train true \
  --do_eval false \
  --bf16 true \
  --tf32 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --max_steps 5 \
  --logging_steps 1 \
  --save_strategy "no" \
  --eval_strategy "no" \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --report_to none

echo "SFT smoke done: ${OUTDIR}"