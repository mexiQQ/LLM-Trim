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
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaMLP, LlamaAttention 
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
import networkx as nx
import matplotlib.pyplot as plt

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
            # print("Reconstruct weight error", error.item())
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

def kl_divergence(p, q, epsilon=1e-9):
    p_safe = torch.clamp(p, min=epsilon).view(-1, p.shape[-1])
    q_safe = torch.clamp(q, min=epsilon).view(-1, p.shape[-1])
    p_safe /= torch.sum(p_safe, dim=-1, keepdim=True)
    q_safe /= torch.sum(q_safe, dim=-1, keepdim=True)
    kl_div = torch.sum(q_safe * torch.log(q_safe/p_safe), dim=-1)
    kl_div = torch.clamp(kl_div, min=0)
    return kl_div

def js_divergence(p, q):
    m = 0.5 * (p + q)
    js_div = torch.sqrt(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(p, m))
    return js_div

def pairwise_js_divergence(tensor):
    N, n_heads, d, _ = tensor.shape
    pairwise_divergence = torch.zeros((n_heads, n_heads), dtype=tensor.dtype, device=tensor.device)

    for i in range(n_heads):
        for j in range(i+1, n_heads):
            p = tensor[:, i, :, :]
            q = tensor[:, j, :, :]
            div = js_divergence(p, q)
            pairwise_divergence[i, j] = div.mean(dim=-1)
            pairwise_divergence[j, i] = pairwise_divergence[i, j]

    return pairwise_divergence

def main(args):
    set_random_seed(args.seed)
    nbatches = args.nbatches 
    batch_size = args.batch_size 
    nsamples = batch_size * nbatches 

    # logger = LoggerWithDepth(
    #     env_name="{}".format(args.save_ckpt_log_name), 
    #     config=args.__dict__,
    #     root_dir='prune_logs',
    #     setup_sublogger=True
    # )

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    ) 

    target_model = copy.deepcopy(model)
    model.to(args.device)

    ########################## first  stage #################################
    #########################################################################
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset, dataloader = get_loaders(
        args.dataset, 
        tokenizer,
        args.max_seq_len,
        batch_size, 
        train=True
    )

    ori_inputs = {}
    ori_attention_mask = {}
    ori_position_ids = {}
    start_key = f"model.layers.{args.block_attention_layer_start}"
    def create_hook(name):
        def hook(module, input, output):
            if name == start_key:
                if name not in ori_attention_mask:
                    ori_attention_mask[name] = input[1][:1].cpu()
                if name not in ori_position_ids:
                    ori_position_ids[name] = input[2].cpu()
                
                if name not in ori_inputs:
                    ori_inputs[name] = find_first_element_in_recursive_tuple(input).cpu()
                else:
                    ori_inputs[name] = torch.cat((
                        ori_inputs[name],
                        find_first_element_in_recursive_tuple(input).cpu()
                    ), dim=0)
            return output
        return hook

    handlers={}
    for name, layer in model.named_modules():
        if name == start_key:
            handlers[name] = layer.register_forward_hook(create_hook(name))

    for idx, batch in enumerate(dataloader):
        if idx >= nbatches:
            break
        print(f"mini batch {idx}")
        with torch.no_grad():
            batch = batch.to(args.device)
            model(batch)
            del batch
            torch.cuda.empty_cache()

    for _,handler in handlers.items():
        handler.remove()
    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

    del model
    torch.cuda.empty_cache()
    print("*" * 15)
    print("*" * 15)

    ########################## second stage #################################
    #########################################################################
    model = copy.deepcopy(target_model) 
    pruner = LinearPruner()
    mse = nn.MSELoss()

    xtxs = {}
    xtys = {}
    normalize_term = {}
    xs = {}
    data_index = {"start": 0}
    ori_outputs = {}

    def create_hook2(name, module_batch_size):
        def hook2(module: torch.nn.Module, input, output):
            x = input[0]
            if name not in xs:
                xs[name] = x.cpu()
            else:
                xs[name] = torch.cat((xs[name], x.cpu()), dim=0)

            x = x.view(-1, x.shape[-1])
            y = ori_outputs[name][data_index["start"]:data_index["start"]+module_batch_size].to(args.device)

            xtx = torch.matmul(x.T, x).cpu()
            xty = torch.matmul(x.T, y.view(-1, y.shape[-1])).cpu()

            if name not in xtxs:
                xtxs[name] = xtx
            else:
                xtxs[name] += xtx

            if name not in xtys:
                xtys[name] = xty
            else:
                xtys[name] += xty

            if name not in normalize_term:
                normalize_term[name] = x.shape[0]
            else:
                normalize_term[name] += x.shape[0]

            return output
        return hook2

    def create_hook3(name):
        def hook(module, input, output):
            if name not in ori_outputs:
                ori_outputs[name] = find_first_element_in_recursive_tuple(output).cpu()
            else:
                ori_outputs[name] = torch.cat((
                    ori_outputs[name], 
                    find_first_element_in_recursive_tuple(output).cpu()
                ), dim=0)
            return output
        return hook

    # inputs of decoder layer 0
    module_batch_size = batch_size * 10 
    inputs = ori_inputs[start_key]
    ori_inputs = copy.deepcopy(inputs)
    pos_ids = ori_position_ids[start_key].cuda()
    attn_mask = ori_attention_mask[start_key].expand(module_batch_size, -1, -1, -1).cuda()

    # recursive to prune llama block
    for index in range(args.block_attention_layer_start, len(model.model.layers)):
        print("*" * 10, "Block", index, "*" * 10)
        module = model.model.layers[index]
        target_module = target_model.model.layers[index]

        handlers = {}
        for name, layer in module.named_modules():
            if type(layer) in [nn.Linear, LlamaAttention, LlamaMLP]:
                ori_name = f"model.layers.{index}.{name}"
                handlers[ori_name] = layer.register_forward_hook(create_hook2(ori_name, module_batch_size))
        module.to(args.device)

        target_handlers = {}
        for name, layer in target_module.named_modules():
            if type(layer) in [nn.Linear, LlamaAttention, LlamaMLP]:
                ori_name = f"model.layers.{index}.{name}"
                target_handlers[ori_name] = layer.register_forward_hook(create_hook3(ori_name))
        target_module.to(args.device)

        # generate original output
        target_outputs = {}
        with torch.no_grad():
            for input_index in range(0, nsamples, module_batch_size):
                inp = ori_inputs[input_index:input_index+module_batch_size, :, :].cuda()
                res, _, _ = target_module(inp, attn_mask, pos_ids)
                if "outputs" not in target_outputs:
                    target_outputs["outputs"] = res[0].cpu()
                else:
                    target_outputs["outputs"] = torch.cat((target_outputs["outputs"], res[0].cpu()), dim=0)
                del res
                del inp
                torch.cuda.empty_cache()
                data_index["start"] += module_batch_size 
        data_index = {"start": 0}

        for key in target_handlers.keys():
            target_handlers[key].remove()

        #########################################################################
        #########################################################################
        
        name_ = f"model.layers.{index}.self_attn"
        layer_ = module.self_attn

        if args.reconstruct and index > args.block_attention_layer_start and (index < args.block_attention_layer_end or args.reconstruct_dense):
            with torch.no_grad():
                for input_index in range(0, nsamples, module_batch_size):
                    inp = inputs[input_index:input_index+module_batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += module_batch_size 
            
            nterm = normalize_term[name_]
            xtx = (xtxs[name_]/nterm).cuda()
            xty_k_proj = (xtys[f"{name_}.k_proj"]/nterm).cuda()
            xty_q_proj = (xtys[f"{name_}.q_proj"]/nterm).cuda()
            xty_v_proj = (xtys[f"{name_}.v_proj"]/nterm).cuda()

            w_k_proj_star = reconstruct_best_weight_xty(
                xtx,
                xty_k_proj,
                w=bias_and_weights(layer_.k_proj).cuda(),
                ridge_regular=1e-4
            )
            del xty_k_proj
            torch.cuda.empty_cache()

            w_q_proj_star = reconstruct_best_weight_xty(
                xtx,
                xty_q_proj,
                w=bias_and_weights(layer_.q_proj).cuda(),
                ridge_regular=1e-4
            )
            del xty_q_proj
            torch.cuda.empty_cache()

            w_v_proj_star = reconstruct_best_weight_xty(
                xtx,
                xty_v_proj,
                w=bias_and_weights(layer_.v_proj).cuda(),
                ridge_regular=1e-4
            )
            del xty_v_proj
            torch.cuda.empty_cache()

            restore_layer_weights(layer_.k_proj, w_k_proj_star)
            restore_layer_weights(layer_.q_proj, w_q_proj_star)
            restore_layer_weights(layer_.v_proj, w_v_proj_star)

            del w_k_proj_star
            del w_q_proj_star
            del w_v_proj_star
            torch.cuda.empty_cache()

            # clear old xtx and xty
            xtxs = {}
            xtys = {}
            xs = {}
            normalize_term = {}
            data_index = {"start": 0}
            torch.cuda.empty_cache()

        #########################################################################
        #########################################################################

        if args.reconstruct and index > args.block_attention_layer_start and (index < args.block_attention_layer_end or args.reconstruct_dense):
            with torch.no_grad():
                for input_index in range(0, nsamples, module_batch_size):
                    inp = inputs[input_index:input_index+module_batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += module_batch_size 

            nterm = normalize_term[f"{name_}.o_proj"]
            xtx = (xtxs[f"{name_}.o_proj"]/nterm).cuda()
            xty_o_proj = (xtys[f"{name_}.o_proj"]/nterm).cuda()
            
            w_o_proj_star = reconstruct_best_weight_xty(
                xtx,
                xty_o_proj,
                w=bias_and_weights(layer_.o_proj).cuda(),
                ridge_regular=1e-4
            )
            del xty_o_proj
            torch.cuda.empty_cache()

            restore_layer_weights(layer_.o_proj, w_o_proj_star)
            del w_o_proj_star
            torch.cuda.empty_cache()

            # clear old xtx and xty
            xtxs = {}
            xtys = {}
            xs = {}
            normalize_term = {}
            data_index = {"start": 0}
            torch.cuda.empty_cache()

        #########################################################################
        #########################################################################

        if args.prune_attn and index >= args.block_attention_layer_start and index < args.block_attention_layer_end:

            mode = args.attn_mode
            if mode == 0:
                imp = None
                similarity = None
                with torch.no_grad():
                    for input_index in range(0, nsamples, batch_size):
                        temp_attn_mask = ori_attention_mask[start_key].expand(batch_size, -1, -1, -1).cuda()
                        inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                        res, _, _ = target_module(inp, temp_attn_mask, pos_ids, output_attentions=True)
                        attn_matrix = res[1]
                        epsilon = 1e-10
                        entropy = -torch.sum(attn_matrix * torch.log2(attn_matrix + epsilon), dim=-1)
                        entropy = torch.mean(entropy, dim=-1)
                        if imp is None:
                            imp = torch.mean(entropy, dim = 0)
                        else:
                            imp += torch.mean(entropy, dim = 0)
                        if similarity is None:
                            similarity = pairwise_js_divergence(attn_matrix)
                        else:
                            similarity += pairwise_js_divergence(attn_matrix)

                        del res
                        del inp
                        del temp_attn_mask
                        torch.cuda.empty_cache()
                        data_index["start"] += batch_size 
                        break
                data_index = {"start": 0}

                current_channels = layer_.v_proj.weight.shape[0] 
                n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                consecutive_groups = 128
                head_number = 32 
                imp_argsort = torch.argsort(imp)
                n_pruned_head = n_pruned//consecutive_groups
                print(imp_argsort)

                print(similarity)
                flat_similarity = similarity.view(-1)
                sorted_indices = torch.argsort(flat_similarity)
                # plt.clf()
                # G = nx.Graph()
                edges = []
                candidates_removed_head = []
                for idx in sorted_indices:
                    i = (idx // similarity.shape[-1]).item()
                    j = (idx % similarity.shape[-1]).item()
                    if flat_similarity[idx] < 0.2 and\
                        (i, j) not in edges and\
                        (j, i) not in edges and i != j:# and\
                    #   len(candidates_removed_head) < n_pruned_head:
                        print(f"Index: ({i}, {j}), Value: {flat_similarity[idx]}") 
                        edges.append((i, j))
                        if i not in candidates_removed_head and j not in candidates_removed_head:
                            candidates_removed_head.append(i)
                # G.add_edges_from(edges)
                # nx.draw(G, with_labels=True, font_weight='bold')
                # plt.savefig(f"figures/head_similarity_{index}.png")
            
                print(f"Number of duplicate heads for block {index}: {len(candidates_removed_head)}")
                # for i in imp_argsort:
                #     if len(candidates_removed_head) < n_pruned_head and\
                #      i not in candidates_removed_head:
                #         candidates_removed_head.append(i)

                if candidates_removed_head:
                    group_size = consecutive_groups
                    pruning_idxs = torch.cat(
                        [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in candidates_removed_head], 0)

                    _ = pruner.prune_out_channels(layer_.v_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_in_channels(layer_.o_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(layer_.k_proj, idxs=pruning_idxs.tolist())
                    _ = pruner.prune_out_channels(layer_.q_proj, idxs=pruning_idxs.tolist())

                    layer_.q_num_heads = layer_.q_proj.weight.data.shape[0] // layer_.q_head_dim
                    layer_.k_num_heads = layer_.k_proj.weight.data.shape[0] // layer_.k_head_dim
                    layer_.v_num_heads = layer_.v_proj.weight.data.shape[0] // layer_.v_head_dim
                    layer_.num_heads = layer_.v_proj.weight.data.shape[0] // layer_.head_dim  
                
            if mode == 1:
                current_channels = layer_.v_proj.weight.shape[0] 
                n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                consecutive_groups = 128
                head_number = 32 
                imp = torch.rand(current_channels).to(args.device)
                imp = imp.view(-1, consecutive_groups).sum(1)
                imp_argsort = torch.argsort(imp)

                pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)] # 129
                group_size = consecutive_groups
                pruning_idxs = torch.cat(
                    [torch.tensor([j+group_size*i for j in range(group_size)])
                        for i in pruning_groups], 0)

                _ = pruner.prune_out_channels(layer_.v_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_in_channels(layer_.o_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_out_channels(layer_.k_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_out_channels(layer_.q_proj, idxs=pruning_idxs.tolist())

                layer_.q_num_heads = layer_.q_proj.weight.data.shape[0] // layer_.q_head_dim
                layer_.k_num_heads = layer_.k_proj.weight.data.shape[0] // layer_.k_head_dim
                layer_.v_num_heads = layer_.v_proj.weight.data.shape[0] // layer_.v_head_dim
                layer_.num_heads = layer_.v_proj.weight.data.shape[0] // layer_.head_dim

            if mode == 2:
                A = layer_.v_proj.weight
                B = layer_.o_proj.weight
                cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/cov_attn/cov_matrix_{index}.pt").to(args.device).float()
                imp = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))
                
                current_channels = A.shape[0] 
                consecutive_groups = 128 # head dim
                head_number = 32 

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

                _ = pruner.prune_out_channels(layer_.v_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_in_channels(layer_.o_proj, idxs=pruning_idxs.tolist())

                layer_.v_head_dim = layer_.v_proj.weight.data.shape[0] // layer_.v_num_heads

        #########################################################################
        #########################################################################

        ori_y = ori_outputs[name_]
        attn_loss = 0
        with torch.no_grad():
            for input_index in range(0, nsamples, module_batch_size):
                inp = inputs[input_index:input_index+module_batch_size, :, :].cuda()
                _, res, _ = module(inp, attn_mask, pos_ids)
                y_label = ori_y[input_index:input_index+module_batch_size, :, :]
                attn_loss += mse(find_first_element_in_recursive_tuple(res).cpu(), y_label).item()
                del res
                del inp
                torch.cuda.empty_cache()
                data_index["start"] += module_batch_size 

        attn_loss /= math.ceil(nsamples/module_batch_size)
        print(f"{name_} reconstruct error is", attn_loss)

        xtxs = {}
        xtys = {}
        xs = {}
        normalize_term = {}
        data_index = {"start": 0}
        torch.cuda.empty_cache()

        #########################################################################
        #########################################################################

        name_ = f"model.layers.{index}.mlp"
        layer_ = module.mlp

        if args.reconstruct and (index < args.block_mlp_layer_end or args.reconstruct_dense):
            with torch.no_grad():
                for input_index in range(0, nsamples, module_batch_size):
                    inp = inputs[input_index:input_index+module_batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += module_batch_size 

            nterm = normalize_term[f"{name_}.up_proj"]
            xtx = (xtxs[f"{name_}.up_proj"]/nterm).cuda()
            xty_up_proj = (xtys[f"{name_}.up_proj"]/nterm).cuda()
            xty_gate_proj = (xtys[f"{name_}.gate_proj"]/nterm).cuda()
            
            w_up_proj_star = reconstruct_best_weight_xty(
                xtx,
                xty_up_proj,
                w=bias_and_weights(layer_.up_proj).cuda(),
                ridge_regular=1e-4
            )
            del xty_up_proj
            torch.cuda.empty_cache()

            w_gate_proj_star = reconstruct_best_weight_xty(
                xtx,
                xty_gate_proj,
                w=bias_and_weights(layer_.gate_proj).cuda(),
                ridge_regular=1e-4
            )
            del xty_gate_proj
            torch.cuda.empty_cache()

            restore_layer_weights(layer_.up_proj, w_up_proj_star)
            restore_layer_weights(layer_.gate_proj, w_gate_proj_star)
            del w_up_proj_star
            del w_gate_proj_star
            torch.cuda.empty_cache()

            # clear old xtx and xty
            xtxs = {}
            xtys = {}
            xs = {}
            normalize_term = {}
            data_index = {"start": 0}
            torch.cuda.empty_cache()

        #########################################################################
        #########################################################################

        if args.reconstruct and (index < args.block_attention_layer_end or args.reconstruct_dense):
            with torch.no_grad():
                for input_index in range(0, nsamples, module_batch_size):
                    inp = inputs[input_index:input_index+module_batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += module_batch_size 

            nterm = normalize_term[f"{name_}.down_proj"]
            xtx = (xtxs[f"{name_}.down_proj"]/nterm).cuda()
            xty_down_proj = (xtys[f"{name_}.down_proj"]/nterm).cuda()
            
            w_down_proj_star = reconstruct_best_weight_xty(
                xtx,
                xty_down_proj,
                w=bias_and_weights(layer_.down_proj).cuda(),
                ridge_regular=1e-4
            )
            del xty_down_proj
            torch.cuda.empty_cache()

            restore_layer_weights(layer_.down_proj, w_down_proj_star)
            del w_down_proj_star
            torch.cuda.empty_cache()

            # clear old xtx and xty
            xtxs = {}
            xtys = {}
            xs = {}
            normalize_term = {}
            data_index = {"start": 0}
            torch.cuda.empty_cache()

        #########################################################################
        #########################################################################

        if args.prune_mlp and index >= args.block_mlp_layer_start and index < args.block_mlp_layer_end:
            A = layer_.gate_proj.weight
            B = layer_.up_proj.weight
            C = layer_.down_proj.weight
            cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/cov/cov_matrix_{index}.pt").to(args.device).float()
            imp = torch.mul(
                torch.norm(C, p=2, dim=0),
                torch.mul(
                    torch.nn.functional.silu(
                        torch.mul(torch.matmul(A, cov_matrix), A).sum(1)
                    ), 
                    torch.mul(torch.matmul(B, cov_matrix), B).sum(1)
                )
            )

            current_channels = A.shape[0]
            # imp = torch.rand(current_channels).to(args.device)
            n_pruned = math.ceil(current_channels * args.pruning_ratio_mlp)
            imp_argsort = torch.argsort(imp)
            pruning_idxs = imp_argsort[:n_pruned]
            
            _ = pruner.prune_out_channels(layer_.gate_proj, idxs=pruning_idxs.tolist())
            _ = pruner.prune_out_channels(layer_.up_proj, idxs=pruning_idxs.tolist())
            _ = pruner.prune_in_channels(layer_.down_proj, idxs=pruning_idxs.tolist())

        #########################################################################
        #########################################################################

        ori_y = ori_outputs[name_]
        mlp_loss = 0
        with torch.no_grad():
            for input_index in range(0, nsamples, module_batch_size):
                inp = inputs[input_index:input_index+module_batch_size, :, :].cuda()
                _, _, res = module(inp, attn_mask, pos_ids)
                y_label = ori_y[input_index:input_index+module_batch_size, :, :]
                mlp_loss += mse(find_first_element_in_recursive_tuple(res).cpu(), y_label).item()
                del res
                del inp
                torch.cuda.empty_cache()
                data_index["start"] += module_batch_size 

        mlp_loss /= math.ceil(nsamples/module_batch_size)
        print(f"{name_} reconstruct error is", mlp_loss)

        xtxs = {}
        xtys = {}
        xs = {}
        normalize_term = {}
        data_index = {"start": 0}
        torch.cuda.empty_cache()

        #########################################################################
        #########################################################################

        module_outputs = {}
        with torch.no_grad():
            for input_index in range(0, nsamples, module_batch_size):
                inp = inputs[input_index:input_index+module_batch_size, :, :].cuda()
                res, _, _ = module(inp, attn_mask, pos_ids)
                if "outputs" not in module_outputs:
                    module_outputs["outputs"] = res[0].cpu()
                else:
                    module_outputs["outputs"] = torch.cat((module_outputs["outputs"], res[0].cpu()), dim=0)
                del res
                del inp
                torch.cuda.empty_cache()
                data_index["start"] += module_batch_size

        xtxs = {}
        xtys = {}
        xs = {}
        normalize_term = {}
        data_index = {"start": 0}
        ori_outputs = {}
        for key in handlers.keys():
            handlers[key].remove()
 
        module.to("cpu")
        target_module.to("cpu")

        del inputs         
        inputs = module_outputs["outputs"]
        del ori_inputs
        ori_inputs = target_outputs["outputs"]
        torch.cuda.empty_cache()

    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    if args.save_model:
        # torch.save(model, f"llama_sparse.pt")
        with torch.no_grad():
            model = model.half()
            model.to(args.device)
            ppl = PPLMetric(model, tokenizer, [
                'wikitext2', 
                # 'ptb'
            ], args.max_seq_len, batch_size=args.batch_size, device=args.eval_device, train=False)
            print("PPL after pruning: {}".format(ppl))

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
            ], args.max_seq_len, batch_size=args.batch_size, device=args.eval_device)
        
        for key, value in metric.items():
            print("{} metric after pruning: {}".format(key, value))
        print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning GPT2 (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="gpt_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')

    # argument for pruning 
    parser.add_argument('--kq_mode', type=str, default="qr_pivot", help='kq prune')
    parser.add_argument('--reconstruct', action='store_true', help='if reconstruct weight')
    parser.add_argument('--reconstruct_dense', action='store_true', help='if reconstruct weight of dense blocks')
    parser.add_argument('--prune_attn', action='store_true', help='if prune attention')
    parser.add_argument('--prune_mlp', action='store_true', help='if prune mlp')
    parser.add_argument('--pruning_ratio_attn', type=float, default=0.5, help='pruning ratio attn')
    parser.add_argument('--pruning_ratio_mlp', type=float, default=0.5, help='pruning ratio mlp')
    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)
    parser.add_argument('--attn_mode', type=int, default=1, help='attn prune mode')

    # argument for generation dataset
    parser.add_argument('--dataset', type=str, default="wikitext2", help='data set name')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nbatches', type=int, default=8)

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
