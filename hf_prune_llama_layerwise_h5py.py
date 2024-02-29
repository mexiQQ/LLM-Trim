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
import h5py
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

def load_data_from_disk(hdpy_file, name, start_index, end_index):
    data = hdpy_file[name][start_index:end_index]
    return torch.from_numpy(data)

lm_head_hook_forbidden = False 
def main(args):
    set_random_seed(args.seed)
    nbatches = args.nbatches
    batch_size = args.batch_size 
    nsamples = batch_size * nbatches 
    file_path = "/mnt/beegfs/jli265/output/llm_pruner/c31/hdf5_file.h5"

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_logs',
        setup_sublogger=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >=1.9 else False
    ) 

    # if args.device != "cpu":
    #     model.half()
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

    ori_outputs = {}
    ori_inputs = {}
    ori_attention_mask = {}
    ori_position_ids = {}
    dataset_configs = {}
    file = h5py.File(file_path, 'a')
    start_key = f"model.layers.{args.block_attention_layer_start}"

    def create_hook(name, file, dataset_configs):
        def hook(module, input, output):
            if name == start_key:
                if name not in ori_attention_mask:
                    ori_attention_mask[name] = input[1].cpu()
                if name not in ori_position_ids:
                    ori_position_ids[name] = input[2].cpu()

                if name not in ori_inputs:
                    ori_inputs[name] = find_first_element_in_recursive_tuple(input).cpu()
                else:
                    ori_inputs[name] = torch.cat((
                        ori_inputs[name],
                        find_first_element_in_recursive_tuple(input).cpu()
                    ), dim=0)

            output_data = find_first_element_in_recursive_tuple(output).cpu().numpy()
            output_dataset_name = f"{name}/output"

            if output_dataset_name not in dataset_configs:
                maxshape = (None,)+output_data.shape[1:]
                chunksize = (args.batch_size,)+output_data.shape[1:]
                dataset_configs[output_dataset_name] = file.create_dataset(output_dataset_name, data=output_data, maxshape=maxshape, chunks=chunksize)
            else:
                dataset = dataset_configs[output_dataset_name]
                dataset.resize((dataset.shape[0] + output_data.shape[0]), axis=0)
                dataset[-output_data.shape[0]:] = output_data
        return hook

    handlers={}
    for name, layer in model.named_modules():
        if "self_attn" in name and type(layer) in [LlamaAttention, nn.Linear]:
            layer_number = int(re.findall(r"layers\.(\d+)\.(?:mlp|self_attn)", name)[0])
            if layer_number >= args.block_attention_layer_start:
                handlers[name] = layer.register_forward_hook(create_hook(name, file, dataset_configs))
        if "mlp" in name and type(layer) in [LlamaMLP, nn.Linear]:
            layer_number = int(re.findall(r"layers\.(\d+)\.(?:mlp|self_attn)", name)[0])
            if  layer_number >= args.block_mlp_layer_start:
                handlers[name] = layer.register_forward_hook(create_hook(name, file, dataset_configs))
        if "lm_head" in name and type(layer) == torch.nn.Linear:
            handlers[name] = layer.register_forward_hook(create_hook(name, file, dataset_configs))
        if name == start_key:
            handlers[name] = layer.register_forward_hook(create_hook(name, file, dataset_configs))

    for idx, batch in enumerate(dataloader):
        if idx >= nbatches:
            break
        logger.log(f"mini batch {idx}")
        with torch.no_grad():
            batch = batch.to(args.device)
            model(batch)
            del batch
            torch.cuda.empty_cache()

    for _,handler in handlers.items():
        handler.remove()
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
    del model
    torch.cuda.empty_cache()

    print("*" * 15)
    print("*" * 15)

    ########################## second stage #################################
    #########################################################################
    model = target_model
    pruner = LinearPruner()
    mse = nn.MSELoss()

    xtxs = {}
    xtys = {}
    normalize_term = {}
    xs = {}
    data_index = {"start": 0}

    def create_hook2(name, file, dataset_configs):
        def hook2(module: torch.nn.Module, input, output):
            x = input[0]
            if name not in xs:
                xs[name] = x.cpu()
            else:
                xs[name] = torch.cat((xs[name], x.cpu()), dim=0)

            x = x.view(-1, x.shape[-1])
            output_dataset_name = f"{name}/output"
            y = load_data_from_disk(
                dataset_configs,
                output_dataset_name,
                data_index["start"],
                data_index["start"]+args.batch_size
            ).to(args.device)

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

    # inputs of decoder layer 0
    inputs = ori_inputs[start_key]
    attn_mask = ori_attention_mask[start_key].cuda()
    pos_ids = ori_position_ids[start_key].cuda()

    # recursive to prune llama block
    for index in range(args.block_attention_layer_start, len(model.model.layers)):
        print("*" * 10, "Block", index, "*" * 10)
        module = model.model.layers[index]

        handlers = {}
        for name, layer in module.named_modules():
            if type(layer) in [nn.Linear, LlamaAttention, LlamaMLP]:
                ori_name = f"model.layers.{index}.{name}"
                handlers[ori_name] = layer.register_forward_hook(create_hook2(ori_name, file, dataset_configs))
        module.to(args.device)

        #########################################################################
        #########################################################################
        
        name_ = f"model.layers.{index}.self_attn"
        layer_ = module.self_attn

        if args.reconstruct and index > args.block_attention_layer_start and index < args.block_attention_layer_end:
            with torch.no_grad():
                for input_index in range(0, nsamples, batch_size):
                    inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += batch_size 
            
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

        if args.reconstruct and index > args.block_attention_layer_start and index < args.block_attention_layer_end:
            with torch.no_grad():
                for input_index in range(0, nsamples, batch_size):
                    inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += batch_size 

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

            mode = 1
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
                current_channels = layer_.v_proj.weight.shape[0] 
                n_pruned = math.ceil(current_channels * args.pruning_ratio_attn)
                consecutive_groups = 128 
                head_number = 32

                A = layer_.v_proj.weight
                B = layer_.o_proj.weight
                cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/cov_attn/cov_matrix_{index}.pt").to(args.device).float()
                imp_ov = torch.mul(torch.norm(B, p=2, dim=0), torch.mul(torch.matmul(A, cov_matrix), A).sum(1))

                imp_k = torch.norm(layer_.k_proj.weight, p=2, dim=1)
                imp_q = torch.norm(layer_.q_proj.weight, p=2, dim=1)
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

                _ = pruner.prune_out_channels(layer_.v_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_in_channels(layer_.o_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_out_channels(layer_.k_proj, idxs=pruning_idxs.tolist())
                _ = pruner.prune_out_channels(layer_.q_proj, idxs=pruning_idxs.tolist())

                layer_.q_num_heads = layer_.q_proj.weight.data.shape[0] // layer_.q_head_dim
                layer_.k_num_heads = layer_.k_proj.weight.data.shape[0] // layer_.k_head_dim
                layer_.v_num_heads = layer_.v_proj.weight.data.shape[0] // layer_.v_head_dim
                layer_.num_heads = layer_.v_proj.weight.data.shape[0] // layer_.head_dim

        #########################################################################
        #########################################################################

        ori_y_key = f"{name_}/output"
        attn_loss = 0
        with torch.no_grad():
            for input_index in range(0, nsamples, batch_size):
                inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                _, res, _ = module(inp, attn_mask, pos_ids)
                # y_label = ori_y[input_index:input_index+batch_size, :, :]
                y_label = load_data_from_disk(dataset_configs, ori_y_key, input_index, input_index+batch_size)
                attn_loss += mse(find_first_element_in_recursive_tuple(res).cpu(), y_label).item()
                del res
                del inp
                torch.cuda.empty_cache()
                data_index["start"] += batch_size 

        attn_loss /= math.ceil(nsamples/batch_size)
        logger.log(f"{name_} reconstruct error is {attn_loss}")

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

        if args.reconstruct and index < args.block_mlp_layer_end:
            with torch.no_grad():
                for input_index in range(0, nsamples, batch_size):
                    inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += batch_size 

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

        if args.reconstruct and index < args.block_attention_layer_end:
            with torch.no_grad():
                for input_index in range(0, nsamples, batch_size):
                    inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                    res, _, _ = module(inp, attn_mask, pos_ids)
                    del res
                    del inp
                    torch.cuda.empty_cache()
                    data_index["start"] += batch_size 

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

        ori_y_key = f"{name_}/output"
        mlp_loss = 0
        with torch.no_grad():
            for input_index in range(0, nsamples, batch_size):
                inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                _, _, res = module(inp, attn_mask, pos_ids)
                # y_label = ori_y[input_index:input_index+batch_size, :, :]
                y_label = load_data_from_disk(dataset_configs, ori_y_key, input_index, input_index+batch_size)
                mlp_loss += mse(find_first_element_in_recursive_tuple(res).cpu(), y_label).item()
                del res
                del inp
                torch.cuda.empty_cache()
                data_index["start"] += batch_size 

        mlp_loss /= math.ceil(nsamples/batch_size)
        logger.log(f"{name_} reconstruct error is {mlp_loss}")

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
            for input_index in range(0, nsamples, batch_size):
                inp = inputs[input_index:input_index+batch_size, :, :].cuda()
                res, _, _ = module(inp, attn_mask, pos_ids)
                
                if "outputs" not in module_outputs:
                    module_outputs["outputs"] = res[0].cpu()
                else:
                    module_outputs["outputs"] = torch.cat((module_outputs["outputs"], res[0].cpu()), dim=0)
                del res
                del inp
                torch.cuda.empty_cache()

                data_index["start"] += batch_size

        xtxs = {}
        xtys = {}
        xs = {}
        normalize_term = {}
        data_index = {"start": 0}
        for key in handlers.keys():
            handlers[key].remove()

        module.to("cpu")
        del inputs         
        inputs = module_outputs["outputs"]
        torch.cuda.empty_cache()

    # close hdpy file
    file.close() 
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    if args.save_model:
        # torch.save(model, f"llama_sparse.pt")
        with torch.no_grad():
            model = model.half()
            model.to(args.device)
            ppl = PPLMetric(model, tokenizer, [
                'wikitext2', 
                # 'ptb'
            ], args.max_seq_len, batch_size=args.batch_size, device=args.eval_device, train=False)
            logger.log("PPL after pruning: {}".format(ppl))

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
            logger.log("{} metric after pruning: {}".format(key, value))
        logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning GPT2 (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='base model name')
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
