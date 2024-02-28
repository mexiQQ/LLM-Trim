import os
import gc
import sys
import time
import math
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
        # print("Weight reconstruct error:", error.item())
        if error.item() < 1:
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
    
def find_first_element_in_recursive_tuple(x):
    if type(x) is tuple:
        return find_first_element_in_recursive_tuple(x[0])
    else:
        return x

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
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    example_prompts = get_examples(
            args.dataset, 
            tokenizer, 
            args.num_examples, 
            seq_len = args.max_seq_len
        ).to(args.device)

    outputs = {}
    def create_hook(name):
        def hook(model: torch.nn.Linear, input, output):
            outputs[name] = find_first_element_in_recursive_tuple(output).cpu()
        return hook

    handlers={}
    for name, layer in model.named_modules():
        if "attn" in name and type(layer) in [torch.nn.Linear, GPT2Attention]:
            layer_number = int(re.findall(r"h\.(\d+)\.(?:mlp|attn)", name)[0])
            if args.prune_attn and layer_number >= args.block_attention_layer_start:
                handlers[name] = layer.register_forward_hook(create_hook(name))
        if "mlp" in name and type(layer) in [torch.nn.Linear, GPT2MLP]:
            layer_number = int(re.findall(r"h\.(\d+)\.(?:mlp|attn)", name)[0])
            if args.prune_mlp and layer_number >= args.block_mlp_layer_start:
                handlers[name] = layer.register_forward_hook(create_hook(name))
        if "lm_head" in name and type(layer) == torch.nn.Linear:
            handlers[name] = layer.register_forward_hook(create_hook(name))

    _ = model(example_prompts)
    for _,handler in handlers.items():
        handler.remove()
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
    print("cleaning memory...")

    print("*" * 15)
    print("*" * 15)

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

            ######################################################
            ######################################################
            ######################################################
            ######################################################
            
            if args.reconstruct:
                if "lm_head" in name:
                    x = input[0]
                else:
                    x = torch.cat((input[0], torch.ones(*(*input[0].shape[:-1], 1), device=input[0].device)), dim=-1)
                x = x.view(-1, x.shape[-1])
                xtx = torch.matmul(x.T, x)

                if "attn" in name:
                    y_k_proj = outputs[f"{name}.k_proj"].to(args.device)
                    y_q_proj = outputs[f"{name}.q_proj"].to(args.device) 
                    y_v_proj = outputs[f"{name}.v_proj"].to(args.device) 
                    y_o_proj = outputs[f"{name}.o_proj"].to(args.device) 
                elif "mlp" in name:
                    y_up_proj = outputs[f"{name}.up_proj"].to(args.device)
                    y_down_proj = outputs[f"{name}.down_proj"].to(args.device)
                elif "lm_head" in name:
                    y_lm_head = outputs[name].to(args.device)
                else:
                    raise NotImplementedError()
                
                if "attn" in name:
                    layer_number = int(re.findall(r"h\.(\d+)\.(?:mlp|attn)", name)[0])
                    if layer_number > args.block_attention_layer_start:
                        # print(f"Reconstructing {name}...")
                        w_k_proj_star = reconstruct_best_weight(xtx, x, y_k_proj.view(-1, y_k_proj.shape[-1]), w=bias_and_weights(module.k_proj), ridge_regular=1e-4)
                        w_q_proj_star = reconstruct_best_weight(xtx, x, y_q_proj.view(-1, y_q_proj.shape[-1]), w=bias_and_weights(module.q_proj), ridge_regular=1e-4)
                        w_v_proj_star = reconstruct_best_weight(xtx, x, y_v_proj.view(-1, y_v_proj.shape[-1]), w=bias_and_weights(module.v_proj), ridge_regular=1e-4)
                        
                        restore_layer_weights(module.k_proj, w_k_proj_star)
                        restore_layer_weights(module.q_proj, w_q_proj_star)
                        restore_layer_weights(module.v_proj, w_v_proj_star)
                        
                        ######################################################
                        ######################################################
                        ######################################################
                        ######################################################
                        with torch.no_grad():
                            _, input_for_o_proj = module(input[0])
                        xo = torch.cat((input_for_o_proj, torch.ones(*(*input_for_o_proj.shape[:-1], 1), device=input_for_o_proj.device)), dim=-1)
                        xo = xo.view(-1, xo.shape[-1])
                        xotxo = torch.matmul(xo.T, xo)

                        w_o_proj_star = reconstruct_best_weight(xotxo, xo, y_o_proj.view(-1, y_o_proj.shape[-1]), w=bias_and_weights(module.o_proj), ridge_regular=1e-4)
                        restore_layer_weights(module.o_proj, w_o_proj_star)
                        
                        del xotxo
                    # xty = torch.matmul(x.T, y.view(-1, y.shape[-1]))
                    # w_star = reconstruct_best_weight_xty(xtx, xty, w=bias_and_weights(module), ridge_regular=1e-4)
                elif "mlp" in name:
                    # print(f"Reconstructing {name}...")
                    w_up_proj_star = reconstruct_best_weight(xtx, x, y_up_proj.view(-1, y_up_proj.shape[-1]), w=bias_and_weights(module.up_proj), ridge_regular=1e-4)
                
                    restore_layer_weights(module.up_proj, w_up_proj_star)

                    ######################################################
                    ######################################################
                    ######################################################
                    ######################################################
                    with torch.no_grad():
                        _, input_for_down_proj = module(input[0])
                    xd = torch.cat((input_for_down_proj, torch.ones(*(*input_for_down_proj.shape[:-1], 1), device=input_for_down_proj.device)), dim=-1)
                    xd = xd.view(-1, xd.shape[-1])
                    xdtxd = torch.matmul(xd.T, xd)

                    w_down_proj_star = reconstruct_best_weight(xdtxd, xd, y_down_proj.view(-1, y_down_proj.shape[-1]), w=bias_and_weights(module.down_proj), ridge_regular=1e-4)
                    restore_layer_weights(module.down_proj, w_down_proj_star)
                    
                    del xdtxd
                elif "lm_head" in name:
                    w_lm_head_star = reconstruct_best_weight(xtx, x, y_lm_head.view(-1, y_lm_head.shape[-1]), w=bias_and_weights(module), ridge_regular=1e-4)

                #     restore_layer_weights(module, w_lm_head_star)
                del xtx

            ######################################################
            ######################################################
            ######################################################
            ######################################################

            # import pdb; pdb.set_trace()
            if "lm_head" not in name:
                layer_number = int(re.findall(r"h\.(\d+)\.(?:mlp|attn)", name)[0])

            if "mlp" in name and layer_number >= args.block_mlp_layer_start and layer_number < args.block_mlp_layer_end:
                # print(f"Pruning {name}...")
                A = module.up_proj.weight
                B = module.down_proj.weight
                cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov/cov_matrix_{layer_number}.pt").to(args.device)
                imp = torch.mul(
                    torch.norm(B, p=2, dim=0),
                    torch.nn.functional.gelu(
                        torch.mul(torch.matmul(A, cov_matrix), A).sum(1)
                    )
                )

                current_channels = A.shape[0]
                # imp = torch.rand(current_channels).to(args.device)
                n_pruned = math.ceil(current_channels * args.pruning_ratio_mlp)
                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:n_pruned]
                _ = pruner.prune_out_channels(module.up_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_in_channels(module.down_proj, idxs=pruning_idxs.tolist())

            if "attn" in name and layer_number >= args.block_attention_layer_start and layer_number < args.block_attention_layer_end:
                # print(f"Pruning {name}...")
                mode = 8
                if mode == 1: # random prune k q v o in head mode
                    current_channels = module.v_proj.weight.shape[0] 
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    consecutive_groups = 64
                    head_number = 12
                    imp = torch.rand(current_channels).to(args.device)
                    imp = imp.view(-1, consecutive_groups).sum(1)
                    imp_argsort = torch.argsort(imp)

                    pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)] # 129
                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in pruning_groups], 0)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_num_heads = module.q_proj.weight.data.shape[0] // module.q_head_dim
                    module.k_num_heads = module.k_proj.weight.data.shape[0] // module.k_head_dim
                    module.v_num_heads = module.v_proj.weight.data.shape[0] // module.v_head_dim
                    module.num_heads = module.v_proj.weight.data.shape[0] // module.head_dim
                
                if mode == 2: # random prune v and o in channel wise
                    current_channels = module.v_proj.weight.shape[0] 
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    consecutive_groups = 64
                    head_number = 12

                    imp = torch.rand(current_channels).to(args.device)
                    max_val = imp.max() 
                    scaled_imp = imp / max_val
                    imp = scaled_imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
                    pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                    pruning_idxs = pruning_idxs.view(-1)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())

                    module.v_head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads
                    module.head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads

                if mode == 2.5: # 2nd moment prune v and o in channel wise 
                    A = module.v_proj.weight
                    B = module.o_proj.weight
                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    imp = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))
                    
                    current_channels = A.shape[0] 
                    consecutive_groups = 64 # head dim
                    head_number = 12

                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    max_val = imp.max() 
                    scaled_imp = imp / max_val
                    imp = scaled_imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
                    pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                    pruning_idxs = pruning_idxs.view(-1)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())

                    module.v_head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads
                    module.head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads

                if mode == 3: # ov 2nd moment prune k, q, v, o in head mode
                    current_channels = module.v_proj.weight.shape[0] 
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    consecutive_groups = 64
                    head_number = 12
                    A = module.v_proj.weight
                    B = module.o_proj.weight
                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    imp = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))
                    imp = imp.view(-1, consecutive_groups).sum(1)
                    imp_argsort = torch.argsort(imp)

                    pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]
                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in pruning_groups], 0)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_num_heads = module.q_proj.weight.data.shape[0] // module.q_head_dim
                    module.k_num_heads = module.k_proj.weight.data.shape[0] // module.k_head_dim
                    module.v_num_heads = module.v_proj.weight.data.shape[0] // module.v_head_dim
                    module.num_heads = module.v_proj.weight.data.shape[0] // module.head_dim

                # 2nd momend prune v and o in channel wise
                # low rank greedy prune k and q in channel wise
                if mode == 4: 
                    A = module.v_proj.weight
                    B = module.o_proj.weight
                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    imp = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))
                    
                    current_channels = A.shape[0] 
                    consecutive_groups = 64 # head dim
                    head_number = 12
                    # kq_mode = args.kq_mode 
                    kq_mode = "qr_pivot"

                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    max_val = imp.max() 
                    scaled_imp = imp / max_val
                    imp = scaled_imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
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
                        cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
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
                        K = (current_channels - n_pruned) // 12
                        kq_batch = torch.matmul(k_weight.transpose(-2, -1), q_weight).float()
                        U, S, V = torch.linalg.svd(kq_batch)

                        new_k_batch = U[..., :K] @ torch.diag_embed(S[..., :K])
                        new_q_batch = V[:, :K, :]

                        new_k = new_k_batch.transpose(-2, -1).reshape(-1, 768)
                        new_q = new_q_batch.reshape(-1, 768)
                    elif kq_mode == "svd_inverse":
                        K = (current_channels - n_pruned) // 12
                        cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)

                        kq_batch = torch.matmul(torch.matmul(k_weight.transpose(-2, -1), q_weight), cov_matrix).float()
                        U, S, V = torch.linalg.svd(kq_batch)

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

                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())
                    module.k_proj.weight = torch.nn.Parameter(new_k)
                    module.q_proj.weight = torch.nn.Parameter(new_q)

                    module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                    module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads
                    module.v_head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads
                    module.head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads

                # random prune k and q in channel wise
                if mode == 5: 
                    current_channels = module.v_proj.weight.shape[0] 
                    consecutive_groups = 64
                    head_number = 12
                    imp = torch.rand(current_channels).to(args.device)

                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    imp = imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
                    pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                    pruning_idxs = pruning_idxs.view(-1)

                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                    module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads

                # low rank greedy prune k and q in channel wise
                if mode == 5.5:
                    current_channels = module.v_proj.weight.shape[0] 
                    consecutive_groups = 64
                    head_number = 12
                    kq_mode = "svd"
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)

                    kq_before_prune = module.q_proj.weight.T @ module.k_proj.weight
                    query_before_prune = module.q_proj(input[0])
                    key_before_prune = module.k_proj(input[0])
                    value_before_prune = module.v_proj(input[0])
                    
                    new_shape = query_before_prune.size()[:-1] + (head_number, -1)
                    query_before_prune = query_before_prune.view(new_shape).permute(0,2,1,3)
                    key_before_prune = key_before_prune.view(new_shape).permute(0,2,1,3)
                    value_before_prune = value_before_prune.view(new_shape).permute(0,2,1,3)
                    attn_output_before_prune, attn_weight_before_prune = module._attn(
                        query_before_prune, key_before_prune, value_before_prune
                    )
                    attn_weight_hand_before_prune = torch.matmul(
                        query_before_prune, key_before_prune.transpose(-1, -2))

                    q_weight = module.q_proj.weight.view(head_number, consecutive_groups, -1)
                    k_weight = module.k_proj.weight.view(head_number, consecutive_groups, -1)
                    if kq_mode == "qr_pivot":
                        k_list = []
                        q_list = []
                        K = (current_channels - n_pruned) // head_number
                        cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
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
                        K = (current_channels - n_pruned) // 12
                        kq_batch = torch.matmul(k_weight.transpose(-2, -1), q_weight).float()
                        U, S, V = torch.linalg.svd(kq_batch)

                        new_k_batch = U[..., :K] @ torch.diag_embed(S[..., :K])
                        new_q_batch = V[:, :K, :]

                        new_k = new_k_batch.transpose(-2, -1).reshape(-1, 768)
                        new_q = new_q_batch.reshape(-1, 768)
                    elif kq_mode == "svd_inverse":
                        K = (current_channels - n_pruned) // 12
                        cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)

                        kq_batch = torch.matmul(torch.matmul(k_weight.transpose(-2, -1), q_weight), cov_matrix).float()
                        U, S, V = torch.linalg.svd(kq_batch)

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

                    imp = torch.rand(current_channels).to(args.device)
                    pruning_idxs = torch.argsort(imp)[:n_pruned]
                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.k_proj.weight = torch.nn.Parameter(new_k)
                    module.q_proj.weight = torch.nn.Parameter(new_q)
                    module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                    module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads

                    kq_after_prune = module.q_proj.weight.T @ module.k_proj.weight
                    query_after_prune = module.q_proj(input[0])
                    key_after_prune = module.k_proj(input[0])
                    value_after_prune = module.v_proj(input[0])
                    
                    new_shape = query_after_prune.size()[:-1] + (head_number, -1)
                    query_after_prune = query_after_prune.view(new_shape).permute(0,2,1,3)
                    key_after_prune = key_after_prune.view(new_shape).permute(0,2,1,3)
                    value_after_prune = value_after_prune.view(new_shape).permute(0,2,1,3)
                    attn_output_after_prune, attn_weight_after_prune = module._attn(
                        query_after_prune, key_after_prune, value_after_prune
                    )
                    attn_weight_hand_after_prune = torch.matmul(
                        query_after_prune, key_after_prune.transpose(-1, -2))
                    
                    print("KQ reconstruct error:", mse(kq_before_prune, 
                        kq_after_prune).item())
                    print("attn output error:", mse(attn_output_before_prune, 
                        attn_output_after_prune).item())
                    print("attn weight error:", mse(attn_weight_before_prune, 
                        attn_weight_after_prune).item())
                    print("attn weight hand error:", mse(attn_weight_hand_before_prune, 
                        attn_weight_hand_after_prune).item())

                # l2 norm greedy k and q in channel wise
                if mode == 6: 
                    current_channels = module.v_proj.weight.shape[0] 
                    consecutive_groups = 64
                    head_number = 12
                    imp_k = torch.norm(module.k_proj.weight, p=2, dim=1)
                    imp_q = torch.norm(module.q_proj.weight, p=2, dim=1)
                    imp = torch.mul(imp_k, imp_q)

                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    imp = imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
                    pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                    pruning_idxs = pruning_idxs.view(-1)

                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                    module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads

                # 2nd moment greedy k and q in channel wise
                if mode == 7: 
                    current_channels = module.v_proj.weight.shape[0] 
                    consecutive_groups = 64
                    head_number = 12

                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    C = module.q_proj.weight
                    D = module.k_proj.weight
                    C_var = torch.mul(torch.matmul(C, cov_matrix), C).sum(1)
                    D_var = torch.mul(torch.matmul(D, cov_matrix), D).sum(1)
                    imp = torch.mul(C_var, D_var)

                    kq_before_prune = module.q_proj.weight.T @ module.k_proj.weight
                    query_before_prune = module.q_proj(input[0])
                    key_before_prune = module.k_proj(input[0])
                    value_before_prune = module.v_proj(input[0])
                    
                    new_shape = query_before_prune.size()[:-1] + (head_number, -1)
                    query_before_prune = query_before_prune.view(new_shape).permute(0,2,1,3)
                    key_before_prune = key_before_prune.view(new_shape).permute(0,2,1,3)
                    value_before_prune = value_before_prune.view(new_shape).permute(0,2,1,3)
                    attn_output_before_prune, attn_weight_before_prune = module._attn(
                        query_before_prune, key_before_prune, value_before_prune
                    )
                    attn_weight_hand_before_prune = torch.matmul(
                        query_before_prune, key_before_prune.transpose(-1, -2))

                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    imp = imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
                    pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                    pruning_idxs = pruning_idxs.view(-1)

                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())
                    module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                    module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads

                    kq_after_prune = module.q_proj.weight.T @ module.k_proj.weight
                    query_after_prune = module.q_proj(input[0])
                    key_after_prune = module.k_proj(input[0])
                    value_after_prune = module.v_proj(input[0])
                    
                    new_shape = query_after_prune.size()[:-1] + (head_number, -1)
                    query_after_prune = query_after_prune.view(new_shape).permute(0,2,1,3)
                    key_after_prune = key_after_prune.view(new_shape).permute(0,2,1,3)
                    value_after_prune = value_after_prune.view(new_shape).permute(0,2,1,3)
                    attn_output_after_prune, attn_weight_after_prune = module._attn(
                        query_after_prune, key_after_prune, value_after_prune
                    )
                    attn_weight_hand_after_prune = torch.matmul(
                        query_after_prune, key_after_prune.transpose(-1, -2))

                    print("KQ reconstruct error:", mse(kq_before_prune, 
                        kq_after_prune).item())
                    print("attn output error:", mse(attn_output_before_prune, 
                        attn_output_after_prune).item())
                    print("attn weight error:", mse(attn_weight_before_prune, 
                        attn_weight_after_prune).item())
                    print("attn weight hand error:", mse(attn_weight_hand_before_prune, 
                        attn_weight_hand_after_prune).item())

                # 2nd moment greedy k q v o in channel wise
                if mode == 8:
                    A = module.v_proj.weight
                    B = module.o_proj.weight
                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    imp = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))
                    
                    current_channels = A.shape[0] 
                    consecutive_groups = 64 # head dim
                    head_number = 12

                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    max_val = imp.max() 
                    scaled_imp = imp / max_val
                    imp = scaled_imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    # pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs = imp_argsort[:, :11]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
                    pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                    pruning_idxs = pruning_idxs.view(-1)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())

                    module.v_head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads
                    module.head_dim = module.v_proj.weight.data.shape[0] // module.v_num_heads

                    current_channels = module.v_proj.weight.shape[0] 
                    consecutive_groups = 64
                    head_number = 12

                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    C = module.q_proj.weight
                    D = module.k_proj.weight
                    C_var = torch.mul(torch.matmul(C, cov_matrix), C).sum(1)
                    D_var = torch.mul(torch.matmul(D, cov_matrix), D).sum(1)
                    imp = torch.mul(C_var, D_var)

                    # imp_k = torch.norm(module.k_proj.weight, p=2, dim=1)
                    # imp_q = torch.norm(module.q_proj.weight, p=2, dim=1)
                    # imp = torch.mul(imp_k, imp_q)

                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    imp = imp.view(-1, consecutive_groups) 
                    imp_argsort = torch.argsort(imp, dim=1)

                    # pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                    pruning_idxs = imp_argsort[:, :11]
                    pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                    pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).to(args.device)
                    pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                    pruning_idxs = pruning_idxs.view(-1)

                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                    module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads

                # svd(bias) greedy prune k and v in channel wise
                if mode == 9:
                    current_channels = module.v_proj.weight.shape[0] 
                    consecutive_groups = 64
                    head_number = 12
                    kq_mode = "svd"
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)

                    query_before_prune = module.q_proj(input[0])
                    key_before_prune = module.k_proj(input[0])
                    value_before_prune = module.v_proj(input[0])
                    
                    new_shape = query_before_prune.size()[:-1] + (head_number, -1)
                    query_before_prune = query_before_prune.view(new_shape).permute(0,2,1,3)
                    key_before_prune = key_before_prune.view(new_shape).permute(0,2,1,3)
                    value_before_prune = value_before_prune.view(new_shape).permute(0,2,1,3)
                    attn_output_before_prune, attn_weight_before_prune = module._attn(
                        query_before_prune, key_before_prune, value_before_prune
                    )
                    attn_weight_hand_before_prune = torch.matmul(
                        query_before_prune, key_before_prune.transpose(-1, -2))

                    k_weight = module.k_proj.weight
                    k_bias = module.k_proj.bias
                    q_weight = module.q_proj.weight
                    q_bias = module.q_proj.bias
                    extended_k_weight = torch.cat([k_weight, k_bias.unsqueeze(1)], dim=1)
                    extended_q_weight = torch.cat([q_weight, q_bias.unsqueeze(1)], dim=1)
                    kq_before_prune = extended_q_weight.T @ extended_k_weight

                    extended_k_weight = extended_k_weight.view(head_number, consecutive_groups, -1)
                    extended_q_weight = extended_q_weight.view(head_number, consecutive_groups, -1)

                    K = (current_channels - n_pruned) // 12
                    kq_batch = torch.matmul(extended_k_weight.transpose(-2, -1), extended_q_weight)
                    U, S, V = torch.linalg.svd(kq_batch)

                    new_k_batch = U[..., :K] @ torch.diag_embed(S[..., :K])
                    new_q_batch = V[..., :K, :] 

                    new_k_and_bias = new_k_batch.transpose(-2, -1).reshape(-1, 769)
                    new_q_and_bias = new_q_batch.reshape(-1, 769)
                    new_k, new_k_bias = new_k_and_bias[:, :-1], new_k_and_bias[:, -1]
                    new_q, new_q_bias = new_q_and_bias[:, :-1], new_q_and_bias[:, -1]

                    imp = torch.rand(current_channels).to(args.device)
                    pruning_idxs = torch.argsort(imp)[:n_pruned]
                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.k_proj.weight = torch.nn.Parameter(new_k)
                    module.k_proj.bias = torch.nn.Parameter(new_k_bias)
                    module.q_proj.weight = torch.nn.Parameter(new_q)
                    module.q_proj.bias = torch.nn.Parameter(new_q_bias)
                    module.q_head_dim = module.q_proj.weight.data.shape[0] // module.q_num_heads
                    module.k_head_dim = module.k_proj.weight.data.shape[0] // module.k_num_heads

                    kq_after_prune = new_q_and_bias.T @ new_k_and_bias
                    query_after_prune = module.q_proj(input[0])
                    key_after_prune = module.k_proj(input[0])
                    value_after_prune = module.v_proj(input[0])
                    
                    new_shape = query_after_prune.size()[:-1] + (head_number, -1)
                    query_after_prune = query_after_prune.view(new_shape).permute(0,2,1,3)
                    key_after_prune = key_after_prune.view(new_shape).permute(0,2,1,3)
                    value_after_prune = value_after_prune.view(new_shape).permute(0,2,1,3)
                    attn_output_after_prune, attn_weight_after_prune = module._attn(
                        query_after_prune, key_after_prune, value_after_prune
                    )
                    attn_weight_hand_after_prune = torch.matmul(
                        query_after_prune, key_after_prune.transpose(-1, -2))
                    
                    print("KQ reconstruct error:", mse(kq_before_prune, 
                        kq_after_prune).item())
                    print("attn output error:", mse(attn_output_before_prune, 
                        attn_output_after_prune).item())
                    print("attn weight error:", mse(attn_weight_before_prune, 
                        attn_weight_after_prune).item())
                    print("attn weight hand error:", mse(attn_weight_hand_before_prune, 
                        attn_weight_hand_after_prune).item())

                # 2nd moment (k,q,v,o) greedy prune k q v o in head mode
                if mode == 10: 
                    current_channels = module.v_proj.weight.shape[0] 
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    consecutive_groups = 64
                    head_number = 12

                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    A = module.v_proj.weight
                    B = module.o_proj.weight
                    imp_ov = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))

                    C = module.q_proj.weight
                    D = module.k_proj.weight
                    C_var = torch.mul(torch.matmul(C, cov_matrix), C).sum(1)
                    D_var = torch.mul(torch.matmul(D, cov_matrix), D).sum(1)
                    imp_kq = torch.mul(C_var, D_var)

                    imp = torch.mul(imp_ov, imp_kq)
                    # imp = torch.add(imp_ov, imp_kq)
                    imp = imp.view(-1, consecutive_groups).sum(1)

                    sampling=True
                    if sampling:
                        imp = imp / imp.sum()
                        imp_neg = 1 - imp
                        pruning_groups = torch.multinomial(imp_neg, n_pruned//consecutive_groups, replacement=False)
                    else:
                        imp_argsort = torch.argsort(imp)
                        pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]

                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in pruning_groups], 0)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_num_heads = module.q_proj.weight.data.shape[0] // module.q_head_dim
                    module.k_num_heads = module.k_proj.weight.data.shape[0] // module.k_head_dim
                    module.v_num_heads = module.v_proj.weight.data.shape[0] // module.v_head_dim
                    module.num_heads = module.v_proj.weight.data.shape[0] // module.head_dim

                
                # l2 greedy prune k q v o in head mode
                if mode == 11: 
                    current_channels = module.v_proj.weight.shape[0] 
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    consecutive_groups = 64
                    head_number = 12

                    imp_v = torch.norm(module.v_proj.weight, p=2, dim=1)
                    imp_o = torch.norm(module.o_proj.weight, p=2, dim=0)
                    imp_ov = torch.mul(imp_v, imp_o)

                    imp_k = torch.norm(module.k_proj.weight, p=2, dim=1)
                    imp_q = torch.norm(module.q_proj.weight, p=2, dim=1)
                    imp_kq = torch.mul(imp_k, imp_q)

                    imp = torch.mul(imp_ov, imp_kq)
                    # imp = torch.add(imp_ov, imp_kq)
                    imp = imp.view(-1, consecutive_groups).sum(1)
                
                    sampling=True
                    if sampling:
                        imp = imp / imp.sum()
                        imp_neg = 1 - imp
                        pruning_groups = torch.multinomial(imp_neg, n_pruned//consecutive_groups, replacement=False)
                    else:
                        imp_argsort = torch.argsort(imp)
                        pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]

                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in pruning_groups], 0)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_num_heads = module.q_proj.weight.data.shape[0] // module.q_head_dim
                    module.k_num_heads = module.k_proj.weight.data.shape[0] // module.k_head_dim
                    module.v_num_heads = module.v_proj.weight.data.shape[0] // module.v_head_dim
                    module.num_heads = module.v_proj.weight.data.shape[0] // module.head_dim

                # l2 greedy prune k q; 2nd moment v o in head mode 
                if mode == 12: 
                    current_channels = module.v_proj.weight.shape[0] 
                    n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                    consecutive_groups = 64
                    head_number = 12

                    A = module.v_proj.weight
                    B = module.o_proj.weight
                    cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt").to(args.device)
                    imp_ov = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))

                    imp_k = torch.norm(module.k_proj.weight, p=2, dim=1)
                    imp_q = torch.norm(module.q_proj.weight, p=2, dim=1)
                    imp_kq = torch.mul(imp_k, imp_q)

                    imp = torch.mul(imp_ov, imp_kq)
                    # imp = torch.add(imp_ov, imp_kq)
                    imp = imp.view(-1, consecutive_groups).sum(1)
                    imp_argsort = torch.argsort(imp)

                    pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)] # 129
                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in pruning_groups], 0)

                    _ = pruner.prune_out_channels(module.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(module.o_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(module.q_proj, idxs=pruning_idxs.tolist())

                    module.q_num_heads = module.q_proj.weight.data.shape[0] // module.q_head_dim
                    module.k_num_heads = module.k_proj.weight.data.shape[0] // module.k_head_dim
                    module.v_num_heads = module.v_proj.weight.data.shape[0] // module.v_head_dim
                    module.num_heads = module.v_proj.weight.data.shape[0] // module.head_dim

            ######################################################
            ######################################################
            ######################################################
            ######################################################

            with torch.no_grad():
                new_output = module(input[0])
            ori_output = outputs[name]
            error = mse(
                ori_output, 
                find_first_element_in_recursive_tuple(
                    new_output
                ).cpu())
            print(f"{name} reconstruct error: ", error.item())

            # import pdb;pdb.set_trace()
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
                if args.prune_attn:
                    if type(layer) is GPT2Attention and layer_number >= args.block_attention_layer_start:
                            handlers[name] = layer.register_forward_hook(create_hook2(name))
                if args.prune_mlp:
                    if type(layer) is GPT2MLP and layer_number >= args.block_mlp_layer_start:
                            handlers[name] = layer.register_forward_hook(create_hook2(name))
            else:
                handlers[name] = layer.register_forward_hook(create_hook2(name))

    with torch.no_grad():
        _ = model(example_prompts)
        for _,handler in handlers.items():
            handler.remove()
    
    torch.save(model, "temp/model.pt")
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
    print("cleaning memory...")
    gc.collect()
    torch.cuda.empty_cache()
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    # import pdb; pdb.set_trace()
    if args.test_after_prune:
        print("*" * 15)
        print("*" * 15)
        os.environ["debug"] = "1"
        target_model.to(args.eval_device)
        target_model.eval()
        model.to(args.eval_device)
        model.eval()
        with torch.no_grad():
            metric = PPLMse2(model, target_model, tokenizer, [
                'wikitext2', 
                # 'ptb'
            ], args.max_seq_len, batch_size=8, device=args.eval_device)
        
        for key, value in metric.items():
            logger.log("{} metric after pruning: {}".format(key, value))
        logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning GPT2 (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="openai-community/gpt2", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="gpt_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')

    # argument for pruning 
    parser.add_argument('--kq_mode', type=str, default="qr_pivot", help='kq prune')
    parser.add_argument('--reconstruct', action='store_true', help='if reconstruct weight')
    parser.add_argument('--prune_attn', action='store_true', help='if prune attention')
    parser.add_argument('--prune_mlp', action='store_true', help='if prune mlp')
    parser.add_argument('--pruning_ratio_attn', type=float, default=0.5, help='pruning ratio attn')
    parser.add_argument('--pruning_ratio_mlp', type=float, default=0.5, help='pruning ratio mlp')
    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    # argument for generation dataset
    parser.add_argument('--dataset', type=str, default="wikitext2", help='data set name')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='max sequence length')
    parser.add_argument('--num_examples', type=int, default=16)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_prune', action='store_true', help='whether test after train')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
