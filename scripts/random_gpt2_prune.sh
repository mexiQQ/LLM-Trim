echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=3 python hf_prune_gpt2_random.py \
    --base_model openai-community/gpt2 \
    --device cuda \
    --eval_device cuda \
    --block_mlp_layer_start 1 \
    --block_mlp_layer_end 9 \
    --block_attention_layer_start 1 \
    --block_attention_layer_end 9 \
    --pruning_ratio_attn 0.75 \
    --pruning_ratio_mlp 0.75 \
    --seed 42 \
    --num_examples 16 \
    --max_seq_len 1024 \
    --test_after_prune \
    --prune_attn \
    --prune_mlp \
    --reconstruct
echo "[FINISH] - Finish Pruning Model"
