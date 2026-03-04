#!/bin/bash
#SBATCH -J sft_smoke
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 0:20:00
#SBATCH --account=mst114553
#SBATCH -o /work/foobarbaz911/vlmr1/logs/sft_smoke_%j.out
#SBATCH -e /work/foobarbaz911/vlmr1/logs/sft_smoke_%j.err

#!/bin/bash
set -euo pipefail

# ====== paths you already have ======
WORK=/work/foobarbaz911/vlmr1
REPO=/home/foobarbaz911/VLM-R1
DATA_STAGE1=${REPO}/stage1.json            # 你 repo 根目錄那份 stage1.json（你已改過 img path）
MODEL=${WORK}/models/Qwen3-VL-8B-Instruct  # 你要主打的 Qwen3-VL-8B-Instruct
OUTDIR=${WORK}/outputs/sft_smoke_${SLURM_JOB_ID}

mkdir -p ${WORK}/logs ${OUTDIR} ${WORK}/hf_cache ${WORK}/xdg_cache ${WORK}/torch_cache

# ====== make conda functions available in non-interactive sbatch ======
# 你不用改 activate.sh；只要這行，conda deactivate 就不會噴 "conda init" 了
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate base >/dev/null 2>&1 || true

# ====== your existing activation flow ======
cd ${REPO}
source activate.sh

# ====== caches (HPC friendly) ======
export HF_HOME=${WORK}/hf_cache
export XDG_CACHE_HOME=${WORK}/xdg_cache
export TORCH_HOME=${WORK}/torch_cache
export TRANSFORMERS_CACHE=${HF_HOME}   # 讓 warning 少一點而已
export TOKENIZERS_PARALLELISM=false

# ====== speed knobs ======
export ATTENTION_IMPL=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=flashinfer  # 如果使用 vLLM
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== JOB ${SLURM_JOB_ID} on $(hostname) ==="
python -c "import torch, transformers; print('torch', torch.__version__, 'tf', transformers.__version__)"
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"

# ====== SFT smoke ======
# 關鍵：dataset_name 要傳 YAML 配置檔
python -m open_r1.sft \
  --dataset_name "${REPO}/sft_config.yaml" \
  --image_root "" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUTDIR}" \
  --do_train true \
  --do_eval false \
  --bf16 true \
  --tf32 true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --max_steps 5 \
  --logging_steps 1 \
  --save_strategy "no" \
  --eval_strategy "no" \
  --gradient_checkpointing

echo "SFT smoke done: ${OUTDIR}"