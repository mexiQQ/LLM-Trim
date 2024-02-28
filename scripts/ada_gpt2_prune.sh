echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=1 python hf_prune_gpt2_end2end.py \
    --base_model openai-community/gpt2 \
    --device cuda \
    --eval_device cuda \
    --block_mlp_layer_start 1 \
    --block_mlp_layer_end 11 \
    --block_attention_layer_start 1 \
    --block_attention_layer_end 11 \
    --pruning_ratio_attn 0.167 \
    --pruning_ratio_mlp 0.167 \
    --kq_mode qr_pivot \
    --seed 42 \
    --num_examples 16 \
    --max_seq_len 1024 \
    --test_after_prune \
    --prune_attn \
    --prune_mlp \
    --reconstruct
echo "[FINISH] - Finish Pruning Model"

