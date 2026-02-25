import torch
from einops import rearrange, einsum

channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## Rearrange an image tensor for mixing across all pixels
print(f"channels_last.size(0): {channels_last.size(0)}")
print(f"channels_last.size(1): {channels_last.size(1)}")
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
print(f"channels_last_flat.size(): {channels_last_flat.size()}")
channels_first_flat = channels_last_flat.transpose(1, 2)
print(f"channels_first_flat.size(): {channels_first_flat.size()}")
channels_first_flat_transformed = channels_first_flat @ B.T
print(f"channels_first_flat_transformed.size(): {channels_first_flat_transformed.size()}")
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
print(f"channels_last_flat_transformed.size(): {channels_last_flat_transformed.size()}")

