echo "[START] - Start Pruning Model"
rm /mnt/beegfs/jli265/output/llm_pruner/c31/hdf5_file.h5
CUDA_VISIBLE_DEVICES=1 python hf_prune_llama_layerwise.py \
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
    --nbatches 1 \
    --max_seq_len 128 \
    --save_model \
    --prune_mlp \
    --prune_attn
echo "[FINISH] - Finish Pruning Model"

