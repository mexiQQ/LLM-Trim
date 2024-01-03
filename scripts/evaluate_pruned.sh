#!/bin/bash

export PYTHONPATH='.'
CUDA_VISIBLE_DEVICES=0 python lm-evaluation-harness/main.py --model hf-causal-experimental \
       --model_args checkpoint=/mnt/beegfs/jli265/output/llm_pruner/c31/llama_prune/pytorch_model.bin,config_pretrained=baffo32/decapoda-research-llama-7B-hf \
       --tasks boolq \
       --device cuda:0 --no_cache \
       --batch_size 1 \
       --output_path /home/jli265/projects/LLM-Pruner/evaluation_logs/result.json

# CUDA_VISIBLE_DEVICES=0 python lm-evaluation-harness/main.py --model hf-causal-experimental \
#        --model_args checkpoint=/mnt/beegfs/jli265/output/llm_pruner/c31/llama_prune/pytorch_model.bin,peft=/mnt/beegfs/jli265/output/llm_pruner/c31/llama_tune,config_pretrained=baffo32/decapoda-research-llama-7B-hf \
#        --tasks boolq \
#        --device cuda:0 --no_cache \
#        --batch_size 1 \
#        --output_path /home/jli265/projects/LLM-Pruner/evaluation_logs/result.json

# openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq