import torch
from einops import rearrange, einsum

D = torch.tensor([[1, 3, 4, 6, 4],
                  [4, 5, 6, 3, 9],
                  [5, 8, 1, 7, 2],
                  [3, 4, 2, 2, 4]])

A = torch.tensor([[5, 8, 1, 7, 2],
                  [3, 4, 2, 2, 4]])

Y1 = D @ A.T
print(f"mmatual: {Y1}")

Y2 = einsum(D, A, "batch_sequence d_in, d_out d_in -> batch_sequence d_out")
print(f"einsum: {Y2}")