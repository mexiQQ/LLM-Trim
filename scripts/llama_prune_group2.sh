prune_ckpt_path='/mnt/beegfs/jli265/output/llm_pruner/c31/llama_prune_l2'

# Define the array of local modes

local_modes=("k_proj" "q_proj" "v_proj" "o_proj" "kq_proj" "vo_proj" "kqv_proj" "kqo_proj" "kqvo_proj")  # Add or modify modes as needed

# Define the array of pruning ratios
pruning_ratios=(0.25)  # Add or modify ratios as needed

# sampling_ps=(1 2 3 4 5 6)

# Loop over each local mode
for pruning_ratio in "${pruning_ratios[@]}"; do
    # Loop over each pruning ratio
    # echo "[START] - Start Pruning Model with Local Mode: l2 and Pruning Ratio: $pruning_ratio"
    # CUDA_VISIBLE_DEVICES=0 python hf_prune.py --base_model baffo32/decapoda-research-llama-7B-hf --pruning_ratio $pruning_ratio --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type l2 --test_after_train --taylor param_first --save_model 
    # echo "[FINISH] - Finish Pruning Model with Local Mode: l2 and Pruning Ratio: $pruning_ratio"

    # echo "[START] - Start Pruning Model with Local Mode: random and Pruning Ratio: $pruning_ratio"
    # CUDA_VISIBLE_DEVICES=0 python hf_prune.py --base_model baffo32/decapoda-research-llama-7B-hf --pruning_ratio $pruning_ratio --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type random --test_after_train --taylor param_first --save_model 
    # echo "[FINISH] - Finish Pruning Model with Local Mode: random and Pruning Ratio: $pruning_ratio"

    for local_mode in "${local_modes[@]}"; do
        echo "[START] - Start Pruning Model with Local Mode: $local_mode and Pruning Ratio: $pruning_ratio"
        CUDA_VISIBLE_DEVICES=0 LOCAL_MODE=gate SAMPLE_P=3 LOCAL_MODE_2=$local_mode python hf_prune.py --base_model baffo32/decapoda-research-llama-7B-hf --pruning_ratio $pruning_ratio --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model 
        echo "[FINISH] - Finish Pruning Model with Local Mode: $local_mode and Pruning Ratio: $pruning_ratio"
    done

    # for sampling_p in "${sampling_ps[@]}"; do
    #     echo "[START] - Start Pruning Model with Local Mode: random_gate_p_$sampling_p and Pruning Ratio: $pruning_ratio"
    #     CUDA_VISIBLE_DEVICES=0 LOCAL_MODE=gate SAMPLE_P=$sampling_p python hf_prune.py --base_model baffo32/decapoda-research-llama-7B-hf --pruning_ratio $pruning_ratio --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model 
    #     echo "[FINISH] - Finish Pruning Model with Local Mode: random_gate_p_$sampling_p and Pruning Ratio: $pruning_ratio"
    # done
done




