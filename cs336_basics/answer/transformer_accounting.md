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
lm_head: 2 * context_length * d_model * vocab_size
TransformerLM: 2 * context_length * d_model * vocab_size + num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length) = 4,513,336,524,800

3. most
SwiGLU

4. small large
538,072,055,808
1,381,001,854,976
2,620,142,387,200
ffn and self attention

5. ffn
149,522,795,724,800
33倍




