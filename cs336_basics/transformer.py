import torch
import sys
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from cs336_basics.attention import RMSnorm, SwiGLU, MultiheadSelfAttention
from cs336_basics.embedding import Embedding


class PreNormTransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None):
        super(PreNormTransformerBlock, self).__init__()

        self.ln1 = RMSnorm(d_model)
        self.ln2 = RMSnorm(d_model)

        self.attn = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len)

        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self,
                x: Float[torch.Tensor, " batch sequence_length d_model"],
                mask: Bool[torch.Tensor, "... sequence_length sequence_length"] | None = None,
                token_positions: Int[torch.Tensor, "sequence_length"] | None = None) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        
        x = x + self.attn(self.ln1(x), mask = mask, token_positions = token_positions)
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(torch.nn.Module):
    def __init__(self,
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float):
        super(TransformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = torch.nn.ModuleList(
            [PreNormTransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)]
        )

        self.ln_final = RMSnorm(d_model)

        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]):
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)

        return self.lm_head(x)







if __name__ == "__main__":
    print("=" * 60)
    print("Testing PreNormTransformerBlock")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    batch_size = 2
    sequence_length = 5
    d_model = 8
    num_heads = 2
    d_ff = 16
    max_seq_len = 10
    theta = 10000.0
    
    print(f"\nHyperparameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  sequence_length: {sequence_length}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  theta: {theta}")
    
    print(f"\nStep 1: Initialize PreNormTransformerBlock")
    transformer_block = PreNormTransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len)
    print(f"  PreNormTransformerBlock created successfully")
    
    print(f"\nStep 2: Create input tensor x")
    x = torch.randn(batch_size, sequence_length, d_model)
    print(f"  x shape: {x.shape}")
    
    print(f"\nStep 3: Create attention mask (causal mask)")
    mask = torch.tril(torch.ones(sequence_length, sequence_length)).bool()
    print(f"  mask shape: {mask.shape}")
    
    print(f"\nStep 4: Create token positions")
    token_positions = torch.arange(sequence_length)
    print(f"  token_positions shape: {token_positions.shape}")
    
    print(f"\nStep 5: Apply RMSnorm (ln1) to x")
    x_ln1 = transformer_block.ln1(x)
    print(f"  x_ln1 shape: {x_ln1.shape}")
    
    print(f"\nStep 6: Apply MultiheadSelfAttention to normalized x")
    attn_output = transformer_block.attn(x_ln1, mask=mask, token_positions=token_positions)
    print(f"  attn_output shape: {attn_output.shape}")
    
    print(f"\nStep 7: Add residual connection (x + attention)")
    x = x + attn_output
    print(f"  x shape after attention: {x.shape}")
    
    print(f"\nStep 8: Apply RMSnorm (ln2) to x")
    x_ln2 = transformer_block.ln2(x)
    print(f"  x_ln2 shape: {x_ln2.shape}")
    
    print(f"\nStep 9: Apply SwiGLU FFN to normalized x")
    ffn_output = transformer_block.ffn(x_ln2)
    print(f"  ffn_output shape: {ffn_output.shape}")
    
    print(f"\nStep 10: Add residual connection (x + FFN)")
    x = x + ffn_output
    print(f"  final output shape: {x.shape}")
    
    print(f"\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    
    print(f"\nAdditional test: Forward pass with all parameters")
    x_test = torch.randn(batch_size, sequence_length, d_model)
    print(f"  Input x shape: {x_test.shape}")
    output = transformer_block(x_test, mask=mask, token_positions=token_positions)
    print(f"  Output shape: {output.shape}")
    
    print(f"\nTest without mask and token_positions")
    x_test2 = torch.randn(batch_size, sequence_length, d_model)
    print(f"  Input x shape: {x_test2.shape}")
    output2 = transformer_block(x_test2)
    print(f"  Output shape: {output2.shape}")
