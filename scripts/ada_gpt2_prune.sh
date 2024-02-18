echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=2 python hf_prune_gpt2_layerwise.py --base_model baffo32/decapoda-research-llama-7B-hf --pruning_ratio 0.25 --device cuda --eval_device cuda --block_mlp_layer_start 11 --block_mlp_layer_end 12 --block_attention_layer_start 11 --block_attention_layer_end 12
echo "[FINISH] - Finish Pruning Model"