import torch
import numpy as np
from tqdm import tqdm

from LLMPruner.datasets.ppl_dataset import get_loaders

def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric

@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()

def PPLMse(model, target_model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        loss = llama_eval_logits(model, target_model, test_loader, device)
        metric[dataset] = loss
        print(metric)
    return metric

@torch.no_grad()
def llama_eval_logits(model, target_model, test_loader, device):
    loss = 0
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        output, all_atten_outputs = model(batch, output_atten_outputs=True)
        output_target, all_atten_outputs_target = target_model(batch, output_atten_outputs=True)
        loss += torch.nn.functional.mse_loss(
            all_atten_outputs[-1], 
            all_atten_outputs_target[-1], 
            reduction="mean")
    loss /= len(test_loader)
    return loss

def PPLMse2(model, target_model, tokenizer, datasets, seq_len=128, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        loss_atten, loss_mlp, ppl  = llama_eval_logits2(model, target_model, test_loader, device)
        metric[dataset] = {"loss_attn": loss_atten, "loss_mlp": loss_mlp, "ppl": ppl} 
        # print(metric)
    return metric

@torch.no_grad()
def llama_eval_logits2(model, target_model, test_loader, device):
    loss_atten = 0
    loss_mlp = 0
    nlls = []
    n_samples = 0
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        output, all_atten_outputs = model(batch, output_atten_outputs=True, output_hidden_states=True)
        output_target, all_atten_outputs_target = target_model(batch, output_atten_outputs=True, output_hidden_states=True)
        loss_mlp += torch.nn.functional.mse_loss(
            output.hidden_states[-1], 
            output_target.hidden_states[-1], 
            reduction="mean")

        loss_atten += torch.nn.functional.mse_loss(
            all_atten_outputs[-1], 
            all_atten_outputs_target[-1], 
            reduction="mean")

        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
            
    loss_atten /= len(test_loader)
    loss_mlp /= len(test_loader)
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return loss_atten.item(), loss_mlp.item(), ppl.item()