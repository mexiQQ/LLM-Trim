python hf_prune_gpt2_layerwise.py --base_model openai-community/gpt2 --device cpu --eval_device cuda --block_mlp_layer_start 1 --block_mlp_layer_end 10 --block_attention_layer_start 1 --block_attention_layer_end 10 --pruning_ratio_attn 0.25 --pruning_ratio_mlp 0.25 --kq_mode qr_pivot --seed 42 --num_examples 16 --max_seq_len 1024 --test_after_prune --prune_attn --prune_mlp