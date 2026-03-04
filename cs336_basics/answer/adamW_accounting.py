vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

P = 2 * vocab_size * d_model + num_layers * (16 * d_model ** 2 + 2 * d_model) + d_model

print(P)
print(16 * P)
active_mem = 4 * (context_length * d_model * (24 * num_layers + 2) + num_layers * num_heads * context_length**2 + context_length * vocab_size)
print(active_mem)
print(14 * P)
