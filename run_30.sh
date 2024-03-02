#!/bin/bash
#SBATCH --job-name=llm_evaluation
#SBATCH --output=job_logs/eval_logs_%j.out
#SBATCH --error=job_logs/eval_logs_%j.err
#SBATCH --partition=a6000
#SBATCH --nodelist=c30
#SBATCH --exclusive

cd /home/jli265/projects/LLM-Pruner
source ~/.bashrc
conda activate llm_pruner

echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=1 python -u hf_prune_llama_layerwise_parallel.py \
    --base_model baffo32/decapoda-research-llama-7B-hf \
    --device cuda \
    --eval_device cuda \
    --block_mlp_layer_start 4 \
    --block_mlp_layer_end 30 \
    --block_attention_layer_start 4 \
    --block_attention_layer_end 30 \
    --pruning_ratio_attn 0.25 \
    --pruning_ratio_mlp 0.25 \
    --attn_mode 1 \
    --kq_mode qr_pivot \
    --seed 42 \
    --batch_size 64 \
    --nbatches 50 \
    --max_seq_len 128 \
    --save_model \
    --prune_mlp \
    --prune_attn \
    --reconstruct >> logs/output_025_64x50_mlp_attn_reconstruct.log


CUDA_VISIBLE_DEVICES=1 python -u hf_prune_llama_layerwise_parallel.py \
    --base_model baffo32/decapoda-research-llama-7B-hf \
    --device cuda \
    --eval_device cuda \
    --block_mlp_layer_start 0 \
    --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 \
    --block_attention_layer_end 32 \
    --pruning_ratio_attn 0.25 \
    --pruning_ratio_mlp 0.25 \
    --attn_mode 2 \
    --kq_mode qr_pivot \
    --seed 42 \
    --batch_size 64 \
    --nbatches 50 \
    --max_seq_len 128 \
    --save_model \
    --prune_mlp \
    --prune_attn \
    --reconstruct >> logs/output_025_64x50_0-32_mlp_attn_reconstruct.log