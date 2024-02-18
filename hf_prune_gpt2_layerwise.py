import os
import gc
import sys
import time
import json
import copy
import re
import random
import argparse
from typing import Callable, Sequence, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, GPT2Tokenizer, GPT2Config
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from LLMPruner.models.hf_bert.modeling_bert import BertForSequenceClassification
from LLMPruner.models.hf_gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2MLP, GPT2Attention

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_gpt2_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric, PPLMse, PPLMse2
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.datasets.ppl_dataset import get_loaders
import copy
from scipy import linalg

class LinearPruner(object):
    def __init__(self, pruning_dim=1):
        self.pruning_dim = pruning_dim

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        keep_idxs.sort()
        layer.out_features = layer.out_features-len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data[keep_idxs])
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        keep_idxs.sort()
        layer.in_features = layer.in_features-len(idxs)
        layer.weight = torch.nn.Parameter(
            layer.weight.data[:, keep_idxs])
        return layer

    def get_out_channels(self, layer):
        return layer.out_features

    def get_in_channels(self, layer):
        return layer.in_features
    
def invert(H: torch.Tensor, ridge_regular=1e-4):
    try:
        ridge = ridge_regular * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], device=H.device)
        H.add_(ridge)
        del ridge
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except RuntimeError:
        return invert(H=H, ridge_regular=ridge_regular * 10)
    return Hinv

def reconstruct_best_weight(xtx: torch.Tensor, x:torch.Tensor, y:torch.Tensor, w:torch.Tensor, ridge_regular=1e-4):
    mse = nn.MSELoss()
    # for ridge_regular in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    for ridge_regular in [ridge_regular]:
        w_star = torch.matmul(torch.matmul(invert(xtx, ridge_regular=ridge_regular), x.T), y)
        error = mse(w, w_star)
        if error.item() < 1:
            # print(error.item())
            break
    return w_star

def reconstruct_best_weight_xty(xtx: torch.Tensor, xty:torch.Tensor, w:torch.Tensor, ridge_regular=1e-4):
    mse = nn.MSELoss()
    for ridge_regular in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    # for ridge_regular in [ridge_regular]:
        invert_h = invert(xtx, ridge_regular=ridge_regular)
        w_star = torch.matmul(invert_h, xty)
        del invert_h
        error = mse(w, w_star)
        if error.item() < 1:
            print("Reconstruct weight error", error.item())
            del error
            break
    return w_star
     
def bias_and_weights(module: torch.nn.Linear):
    weights = module.weight.detach().T
    if module.bias is not None:
        weights = torch.cat((weights, module.bias.detach().view(-1, weights.shape[-1])))
    return weights

def restore_layer_weights(layer, weights):
    if type(layer) == torch.nn.Linear:
        if layer.bias is not None:
            layer.weight.data = weights[:-1, :].reshape(layer.weight.data.shape[::-1]).float().T
            layer.bias.data = weights[-1, :].float()
        else:
            layer.weight.data = weights.reshape(layer.weight.data.shape[::-1]).float().T
    else:
        assert False, "Not support"

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
lm_head_hook_forbidden = False 
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_logs',
        setup_sublogger=True
    )

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", 
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    )

    for i in range(12):
        block = model.transformer.h[i]
        attn = block.attn.c_attn
        attn_o = block.attn.c_proj

        q, k, v = attn.weight.data.split(768, dim=1)
        q_b, k_b, v_b = attn.bias.data.split(768, dim=0)

        model.transformer.h[i].attn.k_proj.weight = torch.nn.Parameter(k.T)
        model.transformer.h[i].attn.k_proj.bias = torch.nn.Parameter(k_b)
        model.transformer.h[i].attn.q_proj.weight = torch.nn.Parameter(q.T)
        model.transformer.h[i].attn.q_proj.bias = torch.nn.Parameter(q_b)
        model.transformer.h[i].attn.v_proj.weight = torch.nn.Parameter(v.T)
        model.transformer.h[i].attn.v_proj.bias = torch.nn.Parameter(v_b)

        model.transformer.h[i].attn.o_proj.weight = torch.nn.Parameter(attn_o.weight.data.T)
        model.transformer.h[i].attn.o_proj.bias = torch.nn.Parameter(attn_o.bias.data)

        del model.transformer.h[i].attn.c_attn
        del model.transformer.h[i].attn.c_proj

        up_proj = model.transformer.h[i].mlp.c_fc
        down_proj = model.transformer.h[i].mlp.c_proj

        model.transformer.h[i].mlp.up_proj.weight = torch.nn.Parameter(up_proj.weight.data.T)
        model.transformer.h[i].mlp.up_proj.bias = torch.nn.Parameter(up_proj.bias.data)
        model.transformer.h[i].mlp.down_proj.weight = torch.nn.Parameter(down_proj.weight.data.T)
        model.transformer.h[i].mlp.down_proj.bias = torch.nn.Parameter(down_proj.bias.data)

        del model.transformer.h[i].mlp.c_fc
        del model.transformer.h[i].mlp.c_proj

    target_model = copy.deepcopy(model)
    model.to(args.device)
    print("*" * 15)
    print("*" * 15)
    print("*" * 15)

    outputs = {}
    def create_hook(name):
        def hook(model: torch.nn.Linear, input, output):
            outputs[name] = output.cpu()
        return hook

    handlers={}
    for name, layer in model.named_modules():
        if type(layer) == torch.nn.Linear:
            handlers[name] = layer.register_forward_hook(create_hook(name))

    example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 128).to(args.device)
    _ = model(example_prompts)
    for _,handler in handlers.items():
        handler.remove()
    # import pdb; pdb.set_trace()
    
    pruner = LinearPruner()
    mse = nn.MSELoss()
    def create_hook2(name):
        def hook2(module: torch.nn.Module, input, output):
            if "lm_head" in name:
                global lm_head_hook_forbidden
                if lm_head_hook_forbidden:
                    return
                lm_head_hook_forbidden = True
            else:
                if module.processing_in_hook:
                    return
                module.processing_in_hook = True
            print(f"Pruning {name}...")

            ######################################################
            ######################################################
            ######################################################
            ######################################################

            # import pdb; pdb.set_trace()
            if False:
                if "lm_head" in name:
                    x = input[0]
                else:
                    x = torch.cat((input[0], torch.ones(*(*input[0].shape[:-1], 1), device=input[0].device)), dim=-1)
                x = x.view(-1, x.shape[-1])
                xtx = torch.matmul(x.T, x).cuda()

                if "attn" in name:
                    y_k_proj = outputs[f"{name}.k_proj"].to(args.device)
                    y_q_proj = outputs[f"{name}.q_proj"].to(args.device) 
                    y_v_proj = outputs[f"{name}.v_proj"].to(args.device) 
                elif "mlp" in name:
                    y_up_proj = outputs[f"{name}.up_proj"].to(args.device)
                elif "lm_head" in name:
                    y_lm_head = outputs[name].to(args.device)
                else:
                    raise NotImplementedError()
                
                if "attn" in name:
                    w_k_proj_star = reconstruct_best_weight(xtx, x, y_k_proj.view(-1, y_k_proj.shape[-1]), w=bias_and_weights(module.k_proj), ridge_regular=1e-4)
                    w_q_proj_star = reconstruct_best_weight(xtx, x, y_q_proj.view(-1, y_q_proj.shape[-1]), w=bias_and_weights(module.q_proj), ridge_regular=1e-4)
                    w_v_proj_star = reconstruct_best_weight(xtx, x, y_v_proj.view(-1, y_v_proj.shape[-1]), w=bias_and_weights(module.v_proj), ridge_regular=1e-4)
                    
                    restore_layer_weights(module.k_proj, w_k_proj_star)
                    restore_layer_weights(module.q_proj, w_q_proj_star)
                    restore_layer_weights(module.v_proj, w_v_proj_star)
                    # xty = torch.matmul(x.T, y.view(-1, y.shape[-1])).cuda()
                    # w_star = reconstruct_best_weight_xty(xtx, xty, w=bias_and_weights(module).cuda(), ridge_regular=1e-4)
                elif "mlp" in name:
                    w_up_proj_star = reconstruct_best_weight(xtx, x, y_up_proj.view(-1, y_up_proj.shape[-1]), w=bias_and_weights(module.up_proj), ridge_regular=1e-4)
                
                    restore_layer_weights(module.up_proj, w_up_proj_star)
                elif "lm_head" in name:
                    w_lm_head_star = reconstruct_best_weight(xtx, x, y_lm_head.view(-1, y_lm_head.shape[-1]), w=bias_and_weights(module), ridge_regular=1e-4)

                    restore_layer_weights(module, w_lm_head_star)
                del xtx

            ######################################################
            ######################################################
            ######################################################
            ######################################################

            # import pdb; pdb.set_trace()
            if "lm_head" not in name:
                layer_number = int(re.findall(r"h\.(\d+)\.(?:mlp|attn)", name)[0])

            if "mlp" in name and layer_number in [args.block_mlp_layer_start, args.block_mlp_layer_end]:
                A = module.up_proj.weight
                B = module.down_proj.weight
                cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov/cov_matrix_{layer_number}.pt")
                imp = torch.mul(
                    torch.norm(B, p=2, dim=0),
                    torch.nn.functional.gelu(
                        torch.mul(torch.matmul(A, cov_matrix), A).sum(1)
                    )
                )

                current_channels = A.shape[0] 
                # n_pruned = int(current_channels * args.pruning_ratio)
                n_pruned = int(current_channels * 0.2)
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:n_pruned]
                _ = pruner.prune_out_channels(module.up_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_in_channels(module.down_proj, idxs=pruning_idxs.tolist())

            if "attn" in name and layer_number in [args.block_attention_layer_start, args.block_attention_layer_end]:
                A = module.v_proj.weight
                B = module.o_proj.weight
                cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt") 
                imp = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))
                
                current_channels = A.shape[0] 
                consecutive_groups = 64 # head dim
                head_number = 12
                kq_mode = args.kq_mode 

                # n_pruned = int(current_channels * args.pruning_ratio)
                n_pruned = int(current_channels * 0.25)
                max_val = imp.max() 
                scaled_imp = imp / max_val
                imp = scaled_imp.view(-1, consecutive_groups) 
                imp_argsort = torch.argsort(imp, dim=1)

                pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).cuda()
                pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                pruning_idxs = pruning_idxs.view(-1)

                _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())

                q_weight = module.q_proj.weight.view(head_number, consecutive_groups, -1)
                k_weight = module.k_proj.weight.view(head_number, consecutive_groups, -1)
                if kq_mode == "qr_pivot":
                    k_list = []
                    q_list = []
                    K = (current_channels - n_pruned) // head_number
                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt") # 4096 x 4096 
                    for i in range(12):
                        A = torch.matmul(torch.matmul(q_weight[i], cov_matrix), k_weight[i].T).float()
                        A = A.detach().cpu().numpy()
                        Q, R, P = linalg.qr(A, mode="economic", pivoting=True)
                        new_k = k_weight[i][P[:K], :]
                        new_q = q_weight[i][P[:K], :]
                        k_list.append(new_k)
                        q_list.append(new_q)
                    new_k = torch.stack(k_list).view(-1, 768)
                    new_q = torch.stack(q_list).view(-1, 768)
                elif kq_mode == "svd":
                    layer_number = re.findall(r"h\.(\d+)\.attn", group[0][0].target.name)[0]
                    print(f"Pruning Layer Number:{layer_number}....")
                    K = (current_channels - n_pruned) // 12
                    kq_batch = torch.matmul(k_weight.transpose(-2, -1), q_weight).float()
                    U, S, V = torch.linalg.svd(kq_batch)

                    new_k_batch = U[..., :K] @ torch.diag_embed(S[..., :K])
                    new_q_batch = V[:, :K, :]

                    new_k = new_k_batch.transpose(-2, -1).reshape(-1, 768)
                    new_q = new_q_batch.reshape(-1, 768)
                elif kq_mode == "svd_inverse":
                    K = (current_channels - n_pruned) // 12
                    layer_number = re.findall(r"h\.(\d+)\.attn", group[0][0].target.name)[0]
                    print(f"Pruning Layer Number:{layer_number}....SVD")
                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt") # 4096 x 4096 

                    kq_batch = torch.matmul(torch.matmul(k_weight.transpose(-2, -1), q_weight), cov_matrix).float()
                    U, S, V = torch.linalg.svd(kq_batch)

                    print(f"Pruning Layer Number:{layer_number}....Inverse")
                    # cov_matrix_inv = invert(cov_matrix)
                    # cov_matrix_inv = torch.inverse(cov_matrix.float())
                    U_, S_, V_h = torch.svd(cov_matrix.float())
                    threshold = 0.00001
                    S_inv = torch.where(S_ > threshold, 1.0 / S_, torch.zeros_like(S_))
                    S_inv_diag = torch.diag(S_inv)
                    cov_matrix_inv = V_h @ S_inv_diag @ U_.t()

                    new_k_batch = (U[..., :K] @ torch.diag_embed(S[..., :K])).transpose(-2, -1)
                    new_q_batch = V[:, :K, :] @ cov_matrix_inv

                    new_k = new_k_batch.reshape(-1, 768)
                    new_q = new_q_batch.reshape(-1, 768)

                print("new_k shape:", new_k.shape)
                print("new_q shape:", new_q.shape)
                
                _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())
                module.k_proj.weight = torch.nn.Parameter(new_q)
                module.q_proj.weight = torch.nn.Parameter(new_q)

                module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads
                module.v_head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads
                module.head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads

            ######################################################
            ######################################################
            ######################################################
            ######################################################

            # import pdb; pdb.set_trace()
            new_output = module(input[0])
            error = mse(output[0].cpu(), new_output[0].cpu())
            print(f"{name} reconstruct error: ", error.item())
            
            if "lm_head" in name:
                lm_head_hook_forbidden = False
            else:
                module.processing_in_hook = False
            return new_output 
        return hook2

    handlers={}
    for name, layer in model.named_modules():
        if type(layer) in [
            GPT2Attention, 
            GPT2MLP
        ] or "lm_head" in name:
            if "lm_head" not in name:
                layer_number = int(re.findall(r"h\.(\d+)\.(?:mlp|attn)", name)[0])
                if type(layer) is GPT2Attention and layer_number >= args.block_attention_layer_start:
                        handlers[name] = layer.register_forward_hook(create_hook2(name))
                if type(layer) is GPT2MLP and layer_number >= args.block_mlp_layer_start:
                        handlers[name] = layer.register_forward_hook(create_hook2(name))
            else:
                handlers[name] = layer.register_forward_hook(create_hook2(name))

    with torch.no_grad():
        _ = model(example_prompts)
        for _,handler in handlers.items():
            handler.remove()
    
    # import pdb; pdb.set_trace()
    print("*" * 15)
    print("*" * 15)
    print("*" * 15)
    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    import os
    os.environ["debug"] = "1"
    target_model.to(args.device)
    target_model.eval()
    model.eval()
    with torch.no_grad():
        metric = PPLMse2(model, target_model, tokenizer, [
            'wikitext2', 
            # 'ptb'
        ], 1024, batch_size=8, device=args.eval_device)
    logger.log("metric after pruning: {}".format(metric))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')

    # argument for generation
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--num_examples', type=int, default=16)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--kq_mode', type=str, default="qr_pivot", help='kq prune')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
