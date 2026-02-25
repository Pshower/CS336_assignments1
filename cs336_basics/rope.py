import torch
from einops import einsum
from jaxtyping import Float

class RoPE(torch.nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None=None):
        super(RoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算cos/sin缓存
        if not hasattr(self, "cos_cached") or not hasattr(self, "sin_cached"):
            # 1. 频率矩阵: shape (d_k // 2)
            freqs_d = 1 / (theta**(torch.arange(0, d_k, 2, device=device) / d_k))
            # 2. 位置矩阵: shape (max_seq_len)
            pos_i = torch.arange(max_seq_len, device=device).float()
            # 3. 频率-位置外积: shape (max_seq_len, d_k // 2)
            freqs = einsum(freqs_d, pos_i, "d_half, max_seq_len -> max_seq_len d_half")

            # 计算cos和sin值
            self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
            self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self,
                x:torch.Tensor, # (..., seq_len, d_k)
                token_positions:torch.Tensor # (..., seq_len)
                )-> torch.Tensor:
        # 按d_k维度分组
        x_odd = x[..., 1::2] # 1, 3, 5 ...
        x_even = x[..., ::2] # 2, 4, 6 ...

        # 对应三角函数
        cos = self.cos_cached[token_positions] # (..., max_seq_len, d_k // 2)
        sin = self.sin_cached[token_positions] # (..., max_seq_len, d_k // 2)

        # 旋转公式
        out_even = cos * x_even - sin * x_odd # 偶数维度
        out_odd = sin * x_even + cos * x_odd # 奇数维度

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2) # (..., seq_len, d_k)





