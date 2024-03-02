echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=1 python hf_prune_llama_layerwise_parallel.py \
    --base_model baffo32/decapoda-research-llama-7B-hf \
    --device cuda \
    --eval_device cuda \
    --block_mlp_layer_start 4 \
    --block_mlp_layer_end 32 \
    --block_attention_layer_start 4 \
    --block_attention_layer_end 32 \
    --pruning_ratio_attn 0.25 \
    --pruning_ratio_mlp 0.25 \
    --kq_mode qr_pivot \
    --seed 42 \
    --batch_size 64 \
    --nbatches 10 \
    --max_seq_len 128 \
    --save_model \
    --prune_mlp \
    --prune_attn \
    --reconstruct
echo "[FINISH] - Finish Pruning Model"

