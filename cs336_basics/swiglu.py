import torch
from einops import einsum
from jaxtyping import Float

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


class SWIglu(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(SWIglu, self).__init__()
        # w1_weight: Float[torch.Tensor, " d_ff d_model"],
        # w2_weight: Float[torch.Tensor, " d_model d_ff"],
        # w3_weight: Float[torch.Tensor, " d_ff d_model"]
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2_weight = torch.nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3_weight = torch.nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        std = (2 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1_weight, mean=0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w2_weight, mean=0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w3_weight, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, in_features: Float[torch.Tensor, " ... d_model"]):
        part2 = silu(einsum(self.w1_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff"))
        part3 = einsum(self.w3_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2_weight, part2 * part3, "d_model d_ff, ... d_ff -> ... d_model")
