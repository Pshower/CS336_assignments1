import torch
import sys
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent.parent.resolve()))


class RMSnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(RMSnorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch seq_len d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # print(f"===>x shape {x.shape}")
        rsm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # print(f"===>rsm shape {rsm.shape}")
        result = (self.weight * x / rsm)
        return result.to(in_dtype)
    
def silu(x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
    sigmoid = torch.nn.Sigmoid()
    return sigmoid(x) * x

def glu(x: Float[torch.Tensor, " ... d_model"], # d_model
        w1_weight: Float[torch.Tensor, " d_ff d_model"],
        w2_weight: Float[torch.Tensor, " d_model d_ff"]
        ) -> Float[torch.Tensor, " ... d_model"]:
    sigmoid = torch.nn.Sigmoid()
    part1 = sigmoid(einsum(w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff"))
    part2 = einsum(w2_weight, x, "d_model d_ff, ... d_model -> ... d_ff")
    return part1 * part2


class SwiGLU(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(SwiGLU, self).__init__()
        # w1_weight: Float[torch.Tensor, " d_ff d_model"],
        # w2_weight: Float[torch.Tensor, " d_model d_ff"],
        # w3_weight: Float[torch.Tensor, " d_ff d_model"]
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = torch.nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)

        std = (2 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1.weight, mean=0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w2.weight, mean=0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w3.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, in_features: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        part2 = silu(self.w1(in_features))
        part3 = self.w3(in_features)
        return self.w2(part2 * part3)


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


def softmax(in_features: Float[torch.Tensor, " ..."],
            dim: int = -1
            ) -> Float[torch.Tensor, " ..."]:
    c = torch.max(in_features, dim=dim, keepdim=True).values
    in_features = torch.exp(in_features - c) / torch.sum(torch.exp(in_features - c), dim=dim, keepdim=True)
    return in_features

def scaled_dot_product_attention(Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    attention_scores : Float[torch.Tensor, " ... queries keys"] = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k) ** 0.5
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

    attention_scores = softmax(attention_scores)
    
    output = einsum(attention_scores, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return output

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            theta: int | None = None,
            max_seq_len: int | None = None,
            device: torch.device | None=None
        ):
        super(MultiheadSelfAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = torch.nn.Linear(num_heads * self.d_k, d_model, bias=False, device=device)
        self.k_proj = torch.nn.Linear(num_heads * self.d_k, d_model, bias=False, device=device)
        self.v_proj = torch.nn.Linear(num_heads * self.d_k, d_model, bias=False, device=device)

        self.output_proj = torch.nn.Linear(d_model, num_heads * self.d_k, bias=False, device=device)

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)

    def forward(self,
                x: Float[torch.Tensor, " ... sequence_length d_in"],
                mask: Bool[torch.Tensor, " ... d_model d_model"] | None = None,
                token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None
                ) -> Float[torch.Tensor, " ... sequence_length d_out"]:
        *batch_dims, seq_len, _ = x.shape

        # 投影

        x_q = self.q_proj(x) # "nxd_k d_model, ... seq_len d_model -> ... seq_len nxd_k"
        x_k = self.k_proj(x) # "nxd_k d_model, ... seq_len d_model -> ... seq_len nxd_k"
        x_v = self.v_proj(x) # "nxd_k d_model, ... seq_len d_model -> ... seq_len nxd_k"

        # 拆分多头
        x_q = rearrange(x_q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        x_k = rearrange(x_k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        x_v = rearrange(x_v, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)

        # 应用rope
        if hasattr(self, "rope"):
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            x_q = self.rope(x_q, token_positions)
            x_k = self.rope(x_k, token_positions)
        
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        # mask = mask.view(*([1] * len(batch_dims)), 1, seq_len, seq_len)   # 广播准备
        
        attn_output = scaled_dot_product_attention(x_q, x_k, x_v, mask)

        attn_output = rearrange(attn_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        output = self.output_proj(attn_output)  # 直接调用 Linear 层 # "... d_model d_model, ... seq_len d_model -> ... seq_len d_model"

        return output

    

    
if __name__ == "__main__":
    print("====== test softmax =====")
    x = torch.tensor(
            [
                [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
                [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
                [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
            ]
        )
    expected = torch.nn.functional.softmax(x, dim=-1)
    homemade = softmax(x, dim=-1)

    print(f"expected: {expected}")
    print(f"homemade: {homemade}")

    print("====== test attention =====")

    seq_len_q = 4
    seq_len_k = 5
    d_k = 6
    d_v = 3
    q = torch.randn(seq_len_q, d_k)
    k = torch.randn(seq_len_k, d_k)
    v = torch.randn(seq_len_k, d_v)
    mask = torch.zeros(seq_len_q, seq_len_k, dtype=bool)

    mask[:2, :3] = True
    mask[2, :2] = True
    mask[3:, :4] = True
    print(f"mask: {mask}")

    attention_scores : Float[torch.Tensor, " ... queries keys"] = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k) ** 0.5
    
    if mask is not None:
        attention_scores_1 = attention_scores.masked_fill(~mask, float("-inf"))

    attention_scores_1 = softmax(attention_scores_1)
    print(f"attention_scores_1: {attention_scores_1}")

    attention_scores_2 = softmax(attention_scores)
    print(f"attention_scores_2: {attention_scores_2}")
    # print(f"softmax attention_scores_2: {softmax(attention_scores_2)}")

    attention_scores_3 = attention_scores * mask
    print(f"attention_scores_3: {attention_scores_3}")
    attention_scores_3 = softmax(attention_scores_3)
    print(f"attention_scores_3: {attention_scores_3}")
    # 先softmax后mask会导致每行和不唯一，所以必须先将attention scores置为-inf

