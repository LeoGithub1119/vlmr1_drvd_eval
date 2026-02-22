#!/usr/bin/env bash

export WORK=/work/foobarbaz911/vlmr1

mkdir -p $WORK/hf_cache \
         $WORK/xdg_cache \
         $WORK/datasets \
         $WORK/models \
         $WORK/outputs

export HF_HOME=$WORK/hf_cache
export HF_HUB_CACHE=$WORK/hf_cache
export HF_DATASETS_CACHE=$WORK/hf_cache
export TRANSFORMERS_CACHE=$WORK/hf_cache
export XDG_CACHE_HOME=$WORK/xdg_cache
export TOKENIZERS_PARALLELISM=false
