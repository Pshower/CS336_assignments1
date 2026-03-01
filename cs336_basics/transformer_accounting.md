vocab_size: 50257
context_length: 1024
num_layers: 48
d_model: 1600
num_heads: 25
d_ff: 6400

1. memory
Embedding: vocab_size * d_model
RMSnorm: d_model
SwiGLU: 3 * d_ff * d_model
RoPE: d_k
MultiheadSelfAttention: 4 * d_model * d_model + d_k = 4 * d_model * d_model + d_model // num_heads
PreNormTransformerBlock: 2 * d_model + 4 * d_model * d_model + d_k + 3 * d_ff * d_model = 2 * d_model + 4 * d_model * d_model + d_model // num_heads + 3 * d_ff * d_model
TransformerLM: vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model * d_model + d_model // num_heads + 3 * d_ff * d_model) + d_model + d_model * vocab_size = 2,127,060,672 参数
8.5 GB内存

2. FLOPs
Embedding: 0
RMSnorm: 0
SwiGLU: 6 * d_ff * d_model * context_length
RoPE: 0
MultiheadSelfAttention: 3 * 2 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 2 * context_length * d_model * d_model = 8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model
PreNormTransformerBlock: 8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length
TransformerLM: 2 * context_length * d_model * context_length + num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length) = 4,352,009,830,400

3. most
SwiGLU

4. small large
460,635,242,496
1,277,752,770,560
2,491,081,031,680
ffn and self attention

5. ffn
147,746,874,982,400
33倍




