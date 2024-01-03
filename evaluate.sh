#!/bin/bash
#SBATCH --job-name=llm_evaluation
#SBATCH --output=job_logs/eval_logs_%j.out
#SBATCH --error=job_logs/eval_logs_%j.err
#SBATCH --partition=a6000
#SBATCH --nodelist=c31
#SBATCH --exclusive

cd /home/jli265/projects/LLM-Pruner
source ~/.bashrc
conda activate llm_pruner
export PYTHONPATH='.'

run_evaluation() {
    local gpu_id="$1"
    local ckpt="$2"
    local task="$3"
    local evaluation_id="$4"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args checkpoint=/mnt/beegfs/jli265/output/llm_pruner/$ckpt,config_pretrained=baffo32/decapoda-research-llama-7B-hf \
        --tasks $task \
        --device cuda:0 --no_cache \
        --batch_size 4 \
        --output_path ./evaluation_logs/eval_$evaluation_id.json >> ./evaluation_logs/eval_$evaluation_id.log 2>&1 &
}

run_evaluation_peft() {
    local gpu_id="$1"
    local ckpt="$2"
    local peft="$3"
    local task="$4"
    local evaluation_id="$5"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args checkpoint=/mnt/beegfs/jli265/output/llm_pruner/$ckpt,peft=/mnt/beegfs/jli265/output/llm_pruner/$peft,config_pretrained=baffo32/decapoda-research-llama-7B-hf \
        --tasks $task \
        --device cuda:0 --no_cache \
        --batch_size 4 \
        --output_path ./evaluation_logs/eval_$evaluation_id.json >> ./evaluation_logs/eval_$evaluation_id.log 2>&1 &
}

# openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq

# run_evaluation 0 c30/llama2_prune/pytorch_model.bin boolq 1
# run_evaluation 1 c30/llama2_prune/pytorch_model.bin openbookqa 2
# run_evaluation 2 c30/llama2_prune/pytorch_model.bin arc_easy 3
# run_evaluation 3 c30/llama2_prune/pytorch_model.bin hellaswag 4
# run_evaluation 0 c31/llama_prune/pytorch_model.bin boolq 5
# run_evaluation 1 c31/llama_prune/pytorch_model.bin openbookqa 6
# run_evaluation 2 c31/llama_prune/pytorch_model.bin arc_easy 7
# run_evaluation 3 c31/llama_prune/pytorch_model.bin hellaswag 8
# run_evaluation_peft 0 c31/llama_prune/pytorch_model.bin c31/llama_tune boolq 9
# run_evaluation_peft 1 c31/llama_prune/pytorch_model.bin c31/llama_tune openbookqa 10 
# run_evaluation_peft 2 c31/llama_prune/pytorch_model.bin c31/llama_tune arc_easy 11 
# run_evaluation_peft 3 c31/llama_prune/pytorch_model.bin c31/llama_tune hellaswag 12 

run_evaluation 0 c31/llama_prune_wo_data/pytorch_model.bin boolq 13 
run_evaluation 1 c31/llama_prune_wo_data/pytorch_model.bin openbookqa 14
run_evaluation 2 c31/llama_prune_wo_data/pytorch_model.bin arc_easy 15 
run_evaluation 3 c31/llama_prune_wo_data/pytorch_model.bin hellaswag 16 
run_evaluation 0 c31/llama_prune_wo_data/pytorch_model.bin winogrande 17 
run_evaluation 1 c31/llama_prune_wo_data/pytorch_model.bin arc_challenge 18 
run_evaluation 2 c31/llama_prune_wo_data/pytorch_model.bin piqa 19

wait

# Additional commands if needed after the evaluation
cd ~/GrabGPU
./gg 42 144 0,1,2,3