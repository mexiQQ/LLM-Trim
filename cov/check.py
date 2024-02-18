import torch

for i in range(32):
    cov_variance = torch.load(f"./cov_matrix_{i}.pt").view(-1)
    print(torch.max(cov_variance))
    print(torch.mean(cov_variance))
    print(torch.var(cov_variance))
    print("*" * 10)
    print("*" * 10)
    print("*" * 10)
