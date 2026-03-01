vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400


TransformerLM = vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model * d_model + d_model // num_heads + 3 * d_ff * d_model) + d_model + d_model * vocab_size
print(TransformerLM)
print(TransformerLM * 4)

print(2 * context_length * d_model * context_length + num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length))

num_layers = 12
d_model = 768
num_heads = 12

print(2 * context_length * d_model * context_length + num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length))


num_layers = 24
d_model = 1024
num_heads = 16


print(2 * context_length * d_model * context_length + num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length))

num_layers = 36
d_model = 1280
num_heads = 20

print(2 * context_length * d_model * context_length + num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length))

vocab_size = 50257
context_length = 1024
context_length_ll = 16384
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

print(2 * context_length * d_model * context_length + num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * d_ff * d_model * context_length))
print(2 * context_length_ll * d_model * context_length_ll + num_layers * (8 * context_length_ll * d_model * d_model + 4 * context_length_ll * context_length_ll * d_model + 6 * d_ff * d_model * context_length_ll))