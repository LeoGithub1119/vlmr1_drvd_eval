#!/bin/bash
#SBATCH -J dl_drvd
#SBATCH -p normal
#SBATCH -c 2
#SBATCH -t 0:30:00
#SBATCH -o /work/foobarbaz911/vlmr1/logs/dl_drvd_%j.out
#SBATCH -e /work/foobarbaz911/vlmr1/logs/dl_drvd_%j.err
#SBATCH --account=mst114553

set -euo pipefail
cd /home/foobarbaz911/VLM-R1
source hpc/bin/activate.sh

mkdir -p /work/foobarbaz911/vlmr1/datasets/DrVD-Bench-repo
huggingface-cli download jerry1565/DrVD-Bench \
  --repo-type dataset \
  --local-dir /work/foobarbaz911/vlmr1/datasets/DrVD-Bench-repo \
  --local-dir-use-symlinks False
