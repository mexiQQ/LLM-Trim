import torch
import torch.nn as nn

class SimpleModule(nn.Module):
    def forward(self, x, y):
        return x + y

def hook(module, input, output):
    print(f"Input: {input}")
    print(f"Output: {output}")

# 实例化模块并注册钩子
module = SimpleModule()
module.register_forward_hook(hook)

# 调用模块
output = module(torch.randn(5), torch.randn(5))
