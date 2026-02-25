import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Linear, self).__init__()
        self.W = torch.nn.Parameter(torch.empty(in_features, out_features,device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "in_features out_features, ... in_features -> ... out_features")
    

