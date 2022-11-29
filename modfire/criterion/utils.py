import torch

def pairwiseHamming(X: torch.Tensor):
    X = X.float() * 2 - 1
    # [N, N, D]
    pairwise = (X.shape[-1] - X @ X.T) / 2
    # mask diagonal
    pairwise[torch.eye(len(pairwise), dtype=torch.bool, device=X.device)] = -1
    return pairwise
