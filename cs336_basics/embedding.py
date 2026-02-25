import torch
from einops import einsum

class Embedding(torch.nn.Module):
    def __init__(self,
                 num_embedding: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Embedding, self).__init__()
        self.W_embedding = torch.nn.Parameter(torch.empty(num_embedding, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W_embedding, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # W_embedding: vocab_size, d_model
        # token_ids: batch_size, sequence_length
        # print(f"===>self.W_embedding shape {self.W_embedding.shape}")
        # print(f"===>token_ids shape {token_ids.shape}")
        result = self.W_embedding[token_ids]
        # print(f"===>result shape {result.shape}")
        return result

    