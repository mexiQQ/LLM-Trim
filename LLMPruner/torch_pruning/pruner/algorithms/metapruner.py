import torch
import torch.nn as nn
import typing
import re
from scipy import linalg

from .scheduler import linear_scheduler
from ..import function
from ... import ops, dependency

def invert(H: torch.Tensor, ridge_regular=1e-4):
    try:
        H = H.float()
        ridge = ridge_regular * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], device=H.device)
        H.add_(ridge)
        del ridge
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except RuntimeError:
        return invert(H=H, ridge_regular=ridge_regular * 10)
    return Hinv

def invert_batch(H_batch: torch.Tensor, ridge_regular=1e-4):
    try:
        ridge = ridge_regular * torch.mean(torch.diagonal(H_batch, dim1=-2, dim2=-1), dim=-1)
        ridge = ridge.unsqueeze(-1).unsqueeze(-1) * torch.eye(H_batch.shape[-1], device=H_batch.device).expand_as(H_batch)
        
        H_batch_ridge = H_batch + ridge

        Hinv_batch = torch.cholesky_inverse(torch.linalg.cholesky(H_batch_ridge))
    except RuntimeError:
        return invert_batch(H_batch, ridge_regular=ridge_regular * 10)
    return Hinv_batch

def sample_from_vector_exclude_indices(vector, p=1, exclude_num_samples=1, key=""):
    # L1 norm is the absolute values
    weights = torch.abs(vector) ** p

    # Normalize the weights to create a probability distribution
    probabilities = (weights / weights.sum()).cuda()
    # probabilities = probabilities ** 2 

    if "attn" in key:
        # noise_level = 0.9  # Vary this between 0 (no noise) and 1 (full noise)
        # num_choices = vector.shape[0] * 128
        # uniform_probabilities = torch.rand(num_choices).view(-1,128).sum(1).cuda()
        # uniform_probabilities = uniform_probabilities / uniform_probabilities.max()
        # # import pdb; pdb.set_trace()
        # blended_probabilities = (1 - noise_level) * probabilities + noise_level * uniform_probabilities
        # imp_argsort = torch.argsort(blended_probabilities)
        # non_sampled_indices = imp_argsort[:exclude_num_samples]

        noise_level = 0  # Vary this between 0 (no noise) and 1 (full noise)
        num_choices = vector.shape[0] * 128
        uniform_probabilities = torch.rand(num_choices).view(-1,128).sum(1).cuda()
        blended_probabilities = (1 - noise_level) * probabilities + noise_level * uniform_probabilities
        sampled_indices = torch.multinomial(blended_probabilities, len(vector) - exclude_num_samples, replacement=False)
        mask = torch.ones(len(vector), dtype=torch.bool)
        mask[sampled_indices] = False
        non_sampled_indices = torch.arange(len(vector))[mask]
    else:
        # Step 1: Sample using the blended distribution
        sampled_indices = torch.multinomial(probabilities, len(vector) - exclude_num_samples, replacement=False)

        # Create a mask for all indices
        mask = torch.ones(len(vector), dtype=torch.bool)

        # Mark the sampled indices as False
        mask[sampled_indices] = False

        # Get the non-sampled indices
        non_sampled_indices = torch.arange(len(vector))[mask]

    return non_sampled_indices

class MetaPruner:
    """
        Meta Pruner for structural pruning.

        Args:
            model (nn.Module): A to-be-pruned model
            example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
            importance (Callable): importance estimator.
            global_pruning (bool): enable global pruning. 
            ch_sparsity (float): global channel sparisty.
            ch_sparsity_dict (Dict[nn.Module, float]): layer-specific sparsity.
            iterative_steps (int): number of steps for iterative pruning.
            iterative_sparsity_scheduler (Callable): scheduler for iterative pruning.
            max_ch_sparsity (float): maximum channel sparsity.
            ignored_layers (List[nn.Module]): ignored modules.

            round_to (int): channel rounding.
            customized_pruners (dict): a dict containing module-pruner pairs.
            unwrapped_parameters (list): nn.Parameter that does not belong to any supported layerss.
            root_module_types (list): types of prunable modules.
            output_transform (Callable): A function to transform network outputs.
        """

    def __init__(
        self,
        # Basic
        model: nn.Module,
        example_inputs: torch.Tensor,
        importance: typing.Callable,
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        global_pruning: bool = False,
        ch_sparsity: float = 0.5,  # channel/dim sparsity
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
        max_ch_sparsity: float = 1.0,
        iterative_steps: int = 1,  # for iterative pruning
        iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
        ignored_layers: typing.List[nn.Module] = None,

        # Advanced
        round_to: int = None,  # round channels to 8x, 16x, ...
        # for grouped channels.
        channel_groups: typing.Dict[nn.Module, int] = dict(),
        # for consecutive channels.
        consecutive_groups: typing.Dict[nn.Module, int] = dict(),
        # pruners for customized layers
        customized_pruners: typing.Dict[typing.Any,
                                        function.BasePruningFunc] = None,
        # unwrapped nn.Parameters like ViT.pos_emb
        unwrapped_parameters: typing.List[nn.Parameter] = None,
        root_module_types: typing.List = [
            ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        root_instances: typing.List = None,
        forward_fn: typing.Callable = None,
        output_transform: typing.Callable = None,
        enable_index_mapping: bool = False,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
        self.max_ch_sparsity = max_ch_sparsity
        self.global_pruning = global_pruning

        self.channel_groups = channel_groups
        self.consecutive_groups = consecutive_groups
        self.root_module_types = root_module_types
        self.root_instances = root_instances
        self.round_to = round_to

        # import pdb; pdb.set_trace()
        # Build dependency graph
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            forward_fn=forward_fn,
            output_transform=output_transform,
            unwrapped_parameters=unwrapped_parameters,
            customized_pruners=customized_pruners,
        )

        # import pdb; pdb.set_trace()
        self.ignored_layers = []
        if ignored_layers:
            for layer in ignored_layers:
                self.ignored_layers.extend(list(layer.modules()))

        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        # Record initial status
        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

        # global channel sparsity for each iterative step
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )

        # The customized channel sparsity for different layers
        self.ch_sparsity_dict = {}
        if ch_sparsity_dict is not None:
            for module in ch_sparsity_dict:
                sparsity = ch_sparsity_dict[module]
                for submodule in module.modules():
                    prunable_types = tuple([ops.type2class(
                        prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
                    if isinstance(submodule, prunable_types):
                        self.ch_sparsity_dict[submodule] = self.iterative_sparsity_scheduler(
                            sparsity, self.iterative_steps
                        )

        # detect group convs & group norms
        for m in self.model.modules():
            if isinstance(m, ops.TORCH_CONV) \
                and m.groups > 1 \
                    and m.groups != m.out_channels:
                self.channel_groups[m] = m.groups
            if isinstance(m, ops.TORCH_GROUPNORM):
                self.channel_groups[m] = m.num_groups
        
        if self.global_pruning: # TODO: Support both ch_groups and consecutive_groups in a single forward
            initial_total_channels = 0
            for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                # utils.count_prunable_out_channels( group[0][0].target.module )

                if ch_groups > 1:
                    initial_total_channels += (self.DG.get_out_channels(
                        group[0][0].target.module) // ch_groups)
                elif consecutive_groups > 1:
                    initial_total_channels += (self.DG.get_out_channels(
                        group[0][0].target.module) // consecutive_groups)
                else:
                    initial_total_channels += self.DG.get_out_channels(group[0][0].target.module) 
                
            self.initial_total_channels = initial_total_channels
        
        if enable_index_mapping:
            for node in self.DG.module2node.values():
                node.enable_index_mapping = True
    
    def pruning_history(self):
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history):
        self.DG.load_pruning_history(pruning_history)

    def get_target_sparsity(self, module):
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
            self.current_step]
        return min(s, self.max_ch_sparsity)

    def reset(self):
        self.current_step = 0

    def regularize(self, model, loss):
        """ Model regularizor
        """
        pass

    def step(self, interactive=False):
        # import pdb; pdb.set_trace()
        self.current_step += 1
        if self.global_pruning:
            if interactive:
                return self.prune_global()
            else:
                for group in self.prune_global():
                    group.prune()
        else:
            if interactive:
                return self.prune_local()
            else:
                # for group in self.prune_local():
                #     group.prune()
                self.prune_local()

    def estimate_importance(self, group, ch_groups=1, consecutive_groups=1):
        # import pdb; pdb.set_trace()
        return self.importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)

    def _check_sparsity(self, group):
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                target_sparsity = self.get_target_sparsity(module)
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None: continue
                if layer_out_ch < self.layer_init_out_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_out_ch == 1:
                    return False

            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None: continue
                if layer_in_ch < self.layer_init_in_ch[module] * (
                    1 - self.max_ch_sparsity
                ) or layer_in_ch == 1:
                    return False
        return True

    def get_channel_groups(self, group):
        if isinstance(self.channel_groups, int):
            return self.channel_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.channel_groups:
                return self.channel_groups[module]
        return 1  # no channel grouping
    
    def get_consecutive_groups(self, group):
        if isinstance(self.consecutive_groups, int):
            return self.consecutive_groups
        for dep, _ in group:
            module = dep.target.module
            if module in self.consecutive_groups:
                return self.consecutive_groups[module]
        return 1  # no channel grouping

    def prune_local(self):
        # import pdb; pdb.set_trace()
        if self.current_step > self.iterative_steps:
            return
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            # check pruning rate
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler

                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                # import pdb; pdb.set_trace()
                imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
                if isinstance(imp, list):
                    if len(imp) == 2:
                        imp_kq = imp[1]
                        imp_ov = imp[0]
                        imp = imp_ov
                    else:
                        imp = imp[0]

                if imp is None: continue
                current_channels = self.DG.get_out_channels(module)
                # import pdb;pdb.set_trace()

                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )

                if self.round_to:
                    n_pruned = n_pruned - (n_pruned % self.round_to)
                    
                if n_pruned <= 0:
                    continue

                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]

                random = False 
                kq_mode = 3 

                if consecutive_groups > 1:
                    max_val = imp.max() 
                    scaled_imp = imp / max_val
                    log_scaled_imp = torch.log(imp / max_val)
                    if random:
                        imp = scaled_imp.view(-1, consecutive_groups).sum(1)
                        imp_argsort = torch.argsort(imp)
                    else:
                        imp = scaled_imp.view(-1, consecutive_groups)
                        imp_argsort = torch.argsort(imp, dim=1)

                        max_val_kq = imp_kq.max()
                        scaled_imp_kq = imp_kq / max_val_kq
                        imp_kq = scaled_imp_kq.view(-1, consecutive_groups)
                        imp_kq_argsort = torch.argsort(imp_kq, dim=1)
                else:
                    imp_argsort = torch.argsort(imp)
                
                # import pdb; pdb.set_trace()
                import os
                do_sampling = False 
                if self.importance.name == "taylor":
                    if "attn" in group[0][0].target.name:
                        do_sampling = os.environ.get('DO_SAMPLE')
                        if do_sampling == "true":
                            do_sampling = False 
                        else:
                            do_sampling = False

                    elif "mlp" in group[0][0].target.name:
                        do_sampling = False 

                if do_sampling:
                    # import pdb; pdb.set_trace()
                    if "attn" in group[0][0].target.name:
                        sample_p = os.environ.get('SAMPLE_P_2')
                        if sample_p is None:
                            sample_p = 1
                        else:
                            sample_p = int(sample_p)
                    elif "mlp" in group[0][0].target.name:
                        sample_p = os.environ.get('SAMPLE_P')
                        if sample_p is None:
                            sample_p = 1
                        else:
                            sample_p = int(sample_p) 

                    if ch_groups > 1:
                        pruning_idxs = sample_from_vector_exclude_indices(imp, p=sample_p, exclude_num_samples=n_pruned//ch_groups, key=group[0][0].target.name)
                        group_size = current_channels//ch_groups
                        pruning_idxs = torch.cat(
                            [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                    elif consecutive_groups > 1:
                        pruning_groups = sample_from_vector_exclude_indices(imp, p=sample_p, exclude_num_samples=n_pruned//consecutive_groups, key=group[0][0].target.name)
                        group_size = consecutive_groups
                        pruning_idxs = torch.cat(
                            [torch.tensor([j+group_size*i for j in range(group_size)])
                            for i in pruning_groups], 0)
                    else:
                        pruning_idxs = sample_from_vector_exclude_indices(imp, p=sample_p, exclude_num_samples=n_pruned, key=group[0][0].target.name)
                else:
                    if ch_groups > 1:
                        pruning_idxs = imp_argsort[:(n_pruned//())]
                        group_size = current_channels//ch_groups
                        pruning_idxs = torch.cat(
                            [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
                    elif consecutive_groups > 1:
                        # import pdb; pdb.set_trace()
                        if random:
                            pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]
                            group_size = consecutive_groups
                            pruning_idxs = torch.cat(
                                [torch.tensor([j+group_size*i for j in range(group_size)])
                                for i in pruning_groups], 0)
                        else:
                            # import pdb; pdb.set_trace()
                            pruning_idxs = imp_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                            pruning_idxs_base = torch.arange(pruning_idxs.size(0)).view(-1, 1)
                            pruning_idxs_base = pruning_idxs_base.expand_as(pruning_idxs).cuda()
                            pruning_idxs = pruning_idxs_base * consecutive_groups + pruning_idxs
                            pruning_idxs = pruning_idxs.view(-1)

                            pruning_idxs_kq = imp_kq_argsort[:, :(n_pruned//(current_channels//consecutive_groups))]
                            pruning_idxs_kq_base = torch.arange(pruning_idxs_kq.size(0)).view(-1, 1)
                            pruning_idxs_kq_base = pruning_idxs_kq_base.expand_as(pruning_idxs_kq).cuda()
                            pruning_idxs_kq = pruning_idxs_kq_base * consecutive_groups + pruning_idxs_kq
                            pruning_idxs_kq = pruning_idxs_kq.view(-1)
                    else:
                        pruning_idxs = imp_argsort[:n_pruned]

                if "attn" in group[0][0].target.name:

                    if not random: 
                        for dep, idx in group:
                            layer = dep.target.module
                            prune_fn = dep.handler
                            layer_name = dep.target.name

                            if prune_fn not in [
                                function.prune_linear_out_channels, function.prune_linear_in_channels, 
                                # hf_rmsnorm_pruner.prune_out_channels, tp.prune_embedding_out_channels, hf_attention_pruner.prune_out_channels,
                                # hf_linear_pruner.prune_out_channels, hf_linear_pruner.prune_in_channels
                            ]:
                                continue

                            # dep(idx=pruning_idxs.tolist())
                            # import pdb; pdb.set_trace()
                            if "q_proj" in layer_name:
                                q_weight = layer.weight.view(12, 64, 768)
                            elif "k_proj" in layer_name:
                                k_weight = layer.weight.view(12, 64, 768)

                        if kq_mode == 1:
                            layer_number = re.findall(r"h\.(\d+)\.attn", group[0][0].target.name)[0]
                            print(f"Pruning Layer Number:{layer_number}....")
                            k_list = []
                            q_list = []
                            K = (current_channels - n_pruned) // 12
                            for i in range(12):
                                kq = torch.matmul(k_weight[i].T, q_weight[i]).float()
                                U, S, V = torch.linalg.svd(kq)
                                new_k = (U[:, :K] @ torch.diag(S[:K])).T
                                new_q = (V[:, :K]).T
                                k_list.append(new_k)
                                q_list.append(new_q)
                            new_k = torch.stack(k_list).view(-1, 768)
                            new_q = torch.stack(q_list).view(-1, 768)
                            print("new_k shape:", new_k.shape)
                            print("new_q shape:", new_q.shape)
                        elif kq_mode == 2:
                            k_list = []
                            q_list = []
                            K = (current_channels - n_pruned) // 12
                            layer_number = re.findall(r"h\.(\d+)\.attn", group[0][0].target.name)[0]
                            print(f"Pruning Layer Number:{layer_number}....")
                            cov_matrix = torch.load(f"/home/jli265/projects/LLM-Pruner/gpt_cov_attn/cov_matrix_{layer_number}.pt") # 4096 x 4096 
                            for i in range(12):
                                kq = torch.matmul(torch.matmul(k_weight[i].T, q_weight[i]), cov_matrix).float()
                                U, S, V = torch.linalg.svd(kq)
                                new_k = (U[:, :K] @ torch.diag(S[:K])).T
                                new_q = (V[:, :K]).T @ invert(cov_matrix)
                                k_list.append(new_k)
                                q_list.append(new_q)
                            new_k = torch.stack(k_list).view(-1, 768)
                            new_q = torch.stack(q_list).view(-1, 768)
                            print("new_k shape:", new_k.shape)
                            print("new_q shape:", new_q.shape)
                        elif kq_mode == 3:
                            k_list = []
                            q_list = []
                            K = (current_channels - n_pruned) // 12
                            layer_number = re.findall(r"h\.(\d+)\.attn", group[0][0].target.name)[0]
                            print(f"Pruning Layer Number:{layer_number}....")
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
                            print("new_k shape:", new_k.shape)
                            print("new_q shape:", new_q.shape)
                        elif kq_mode == 4:
                            # import pdb; pdb.set_trace()

                            layer_number = re.findall(r"h\.(\d+)\.attn", group[0][0].target.name)[0]
                            print(f"Pruning Layer Number:{layer_number}....")
                            K = (current_channels - n_pruned) // 12
                            # query_states, key_states.transpose(2, 3)
                            kq_batch = torch.matmul(k_weight.transpose(-2, -1), q_weight).float()
                            U, S, V = torch.linalg.svd(kq_batch)

                            new_k_batch = U[..., :K] @ torch.diag_embed(S[..., :K])
                            new_q_batch = V[:, :K, :]

                            new_k = new_k_batch.transpose(-2, -1).reshape(-1, 768)
                            new_q = new_q_batch.reshape(-1, 768)
                            print("new_k shape:", new_k.shape)
                            print("new_q shape:", new_q.shape)
                        elif kq_mode == 5:
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
                            # S_inv_diag = torch.diag(1/S_)
                            cov_matrix_inv = V_h @ S_inv_diag @ U_.t()
                            # import pdb; pdb.set_trace()

                            new_k_batch = (U[..., :K] @ torch.diag_embed(S[..., :K])).transpose(-2, -1)
                            new_q_batch = V[:, :K, :] @ cov_matrix_inv

                            new_k = new_k_batch.reshape(-1, 768)
                            new_q = new_q_batch.reshape(-1, 768)

                            print("new_k shape:", new_k.shape)
                            print("new_q shape:", new_q.shape)

                    for dep, idx in group:
                        layer = dep.target.module
                        prune_fn = dep.handler
                        layer_name = dep.target.name

                        if prune_fn not in [
                            function.prune_linear_out_channels, function.prune_linear_in_channels, 
                            # hf_rmsnorm_pruner.prune_out_channels, tp.prune_embedding_out_channels, hf_attention_pruner.prune_out_channels,
                            # hf_linear_pruner.prune_out_channels, hf_linear_pruner.prune_in_channels
                        ]:
                            continue

                        # dep(idx=pruning_idxs.tolist())
                        # import pdb; pdb.set_trace()
                        if "q_proj" in layer_name:
                            pass
                            # import pdb; pdb.set_trace()
                            if not random:
                                dep(idxs=pruning_idxs_kq.tolist())
                                layer.weight = torch.nn.Parameter(new_q)
                            else:
                                dep(idxs=pruning_idxs.tolist())
                        elif "k_proj" in layer_name:
                            pass
                            if not random:
                                dep(idxs=pruning_idxs_kq.tolist())
                                layer.weight = torch.nn.Parameter(new_k)
                            else:
                                dep(idxs=pruning_idxs.tolist())
                        elif "v_proj" in layer_name:
                            pass
                            dep(idxs=pruning_idxs.tolist())
                        elif "o_proj" in layer_name:
                            pass
                            dep(idxs=pruning_idxs.tolist())
                else:
                    group = self.DG.get_pruning_group(
                        module, pruning_fn, pruning_idxs.tolist())
                    if self.DG.check_pruning_group(group):
                        group.prune()

    def prune_global(self):

        # import pdb; pdb.set_trace()
        if self.current_step > self.iterative_steps:
            return
        global_importance = []
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups(group)
                consecutive_groups = self.get_consecutive_groups(group)
                imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
                if imp is None: continue
                if ch_groups > 1:
                    imp = imp[:len(imp)//ch_groups]
                if consecutive_groups > 1:
                    imp = imp.view(-1, consecutive_groups).sum(1)
                global_importance.append((group, ch_groups, consecutive_groups, imp))

        imp = torch.cat([local_imp[-1]
                        for local_imp in global_importance], dim=0)
        print(imp.shape, len(global_importance))
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        n_pruned = len(imp) - int(
            self.initial_total_channels *
            (1 - target_sparsity)
        )
        print(n_pruned, target_sparsity, self.initial_total_channels)
        if n_pruned <= 0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        
        # global pruning through thresholding
        thres = topk_imp[-1]
        for group, ch_groups, consecutive_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = (imp <= thres).nonzero().view(-1)
            
            # import pdb; pdb.set_trace()
            if pruning_indices.size(-1) == 0:
                continue
            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module)//ch_groups
                pruning_indices = torch.cat(
                    [pruning_indices+group_size*i for i in range(ch_groups)], 0)
            if consecutive_groups > 1:
                group_size = consecutive_groups
                pruning_indices = torch.cat(
                    [torch.tensor([j+group_size*i for j in range(group_size)])
                    for i in pruning_indices], 0)
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices.tolist())
            if self.DG.check_pruning_group(group):
                yield group