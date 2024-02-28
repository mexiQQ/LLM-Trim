echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=2 python hf_prune_llama_layerwise.py \
    --base_model baffo32/decapoda-research-llama-7B-hf \
    --device cuda \
    --eval_device cuda \
    --block_mlp_layer_start 4 \
    --block_mlp_layer_end 30 \
    --block_attention_layer_start 4 \
    --block_attention_layer_end 30 \
    --pruning_ratio_attn 0.5 \
    --pruning_ratio_mlp 0.5 \
    --kq_mode qr_pivot \
    --seed 42 \
    --batch_size 64 \
    --max_seq_len 128 \
    --save_model \
    --prune_mlp
echo "[FINISH] - Finish Pruning Model"

