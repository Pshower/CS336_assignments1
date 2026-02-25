import torch
from einops import einsum

class RMSnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(RMSnorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch seq_len d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # print(f"===>x shape {x.shape}")
        rsm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # print(f"===>rsm shape {rsm.shape}")
        result = (self.g * x / rsm)
        return result.to(in_dtype)
