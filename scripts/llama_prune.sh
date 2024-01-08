prune_ckpt_path='/mnt/beegfs/jli265/output/llm_pruner/c31/llama_prune_l2'
# tune_ckpt_path='/mnt/beegfs/jli265/output/llm_pruner/c30/llama_tune'

echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=0 LOCAL_MODE=gate SAMPLE_P=3 python hf_prune.py --base_model baffo32/decapoda-research-llama-7B-hf --pruning_ratio 0.35 --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model #--global_pruning 
echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model $prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir $tune_ckpt_path --wandb_project llama_tune --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {$tune_ckpt_path}/"

# echo "You can use the command:"
# echo "       python generate.py --model_type tune_prune_LLM --ckpt $prune_ckpt_path/pytorch_model.bin --lora_ckpt $tune_ckpt_path"
# echo "to use the pruned model"



