# HPC Tools

Reusable HPC workflow toolkit for GPU research projects.

This toolkit separates:

- Environment layer
- Experiment configuration
- Task logic
- Slurm resource scheduling

It allows rapid iteration when cloning new repositories.

---

## Folder Structure

hpc-tools/
  bin/
    activate.sh
    load_env.sh
    run_eval.sh
    run_train.sh
    download_hf.sh
    common.sh
  slurm/
    sbatch_eval_1gpu.sbatch
    sbatch_train_1gpu.sbatch
    sbatch_train_4gpu.sbatch
    salloc_1gpu.sh
    srun_1gpu.sh
  templates/
    project.env.example
    project.config.example
  README.md

---

## How to Use in a New Project

1. Clone your research repo:

   git clone <repo>
   cd <repo>

2. Copy toolkit into repo:

   cp -r ~/hpc-tools ./hpc

3. Create project-specific configs:

   cp ./hpc/templates/project.env.example ./hpc/project.env
   cp ./hpc/templates/project.config.example ./hpc/project.config

4. Edit:

   - hpc/project.env  (paths, venv location, modules)
   - hpc/project.config  (model, dataset, experiment name)

5. Run:

   Interactive:
     bash ./hpc/bin/activate.sh
     bash ./hpc/bin/run_eval.sh

   Batch:
     sbatch ./hpc/slurm/sbatch_eval_1gpu.sbatch

---

## Design Philosophy

activate.sh      -> environment setup  
project.env      -> system paths  
project.config   -> experiment parameters  
run_*.sh         -> task logic  
slurm/*.sbatch   -> resource scheduling  

Environment stays stable.
Only experiment parameters change.

---

## Goal

- Faster iteration
- Reproducible experiments
- Clean separation of concerns
- Portable HPC workflow
