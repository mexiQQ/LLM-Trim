echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 python -u hf_prune_llama_layerwise.py \
    --base_model baffo32/decapoda-research-llama-7B-hf \
    --device cuda \
    --eval_device cuda \
    --block_mlp_layer_start 4 \
    --block_mlp_layer_end 30 \
    --block_attention_layer_start 4 \
    --block_attention_layer_end 30 \
    --pruning_ratio_attn 0.6 \
    --pruning_ratio_mlp 0.6 \
    --kq_mode qr_pivot \
    --seed 42 \
    --batch_size 64 \
    --nbatches 20 \
    --max_seq_len 128 \
    --attn_mode 0 \
    --save_model \
    --prune_attn \
    --prune_mlp >> logs/output.txt
echo "[FINISH] - Finish Pruning Model"


# jeffwan/llama-13b-hf
# baffo32/decapoda-research-llama-7B-hf

# [17,  7, 18,  0, 15,  4,  2, 27,             1, 26, 28, 12, 20,  3, 24, 25, 10, 14, 23,  8, 30, 29, 11, 22,  6,  5, 19, 31, 13,  9, 21, 16]
# tensor([59.8818, 60.3009, 60.2716, 62.5571, 59.9677, 66.2664, 66.1047, 59.7818,                                                        │
#         64.7888, 67.6574, 63.5279, 65.9848, 62.4856, 67.5574, 63.7467, 59.9393,                                                        │
#         70.4125, 56.0045, 59.8531, 67.1376, 62.5541, 67.7264, 66.0408, 64.6910,                                                        │
#         63.1245, 63.2783, 61.5817, 60.2747, 62.2423, 65.4539, 64.9794, 67.4208],                                                       │
#        device='cuda:0') 

# 56
# 63
# 70
# 3.23
# 12.61


# [22, 28, 11, 21, 12, 16,  6,  0]
# 13.32

# [22, 13, 15,  4,  3, 29, 26, 27,              9, 28, 31, 19,  8, 25, 17,  2, 18, 16,                                                        │
#         14, 23,  5,  0,  7, 11,  6, 12, 20, 30, 21, 10,  1, 24]
# tensor([ 580.3528,  705.3680,  540.0167,  421.9088,  354.8526,  580.2692,                                                              │
#          592.2835,  584.1385,  511.5545,  454.2831,  658.8547,  591.0197,                                                              │
#          619.8187,  315.0815,  560.0627,  336.1240,  548.3038,  538.7919,                                                              │
#          540.4589,  493.2682,  638.2990,  650.3068,  101.6596,  575.1187,                                                              │
#         1015.8547,  530.9937,  437.7695,  449.0978,  473.8790,  437.6954,                                                              │
#          649.3920,  488.4930], device='cuda:0', grad_fn=<SumBackward1>)
# 12.74

# 101.6
# 530.48
# 1015.8547
# 149.9
