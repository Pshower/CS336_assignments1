import torch
from einops import rearrange, einsum

images = torch.randn(64, 128, 128, 3)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images_1 = images_rearr * dim_value

print(f"reshape and multiply: {dimmed_images_1}")

## Or in one go:
dimmed_images_2 = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)

print(f"einsum: {dimmed_images_2}")

print(f"equal: {(dimmed_images_1 == dimmed_images_2).sum()}, nums: {64 * 128 * 128 * 3 * 10}")
print(f"values_equal: {dimmed_images_1.values == dimmed_images_2.values}")

