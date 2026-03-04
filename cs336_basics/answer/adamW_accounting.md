vocab_size: 50257
context_length: 1024
num_layers: 48
d_model: 1600
num_heads: 25
d_ff: 6400

(a)

参数量 P = 2 * vocab_size * d_model + num_layers * (16 * d_model ** 2 + 2 * d_model) + d_model
参数内存 4P，梯度内存 4P，优化器内存 8P
激活内存 4(context_length * d_model * (24 * num_layers + 2) + num_layers * num_heads * context_length**2 + context_length * vocab_size) + 16 * P

(b)
P = 2,127,057,600
16P = 34,032,921,600
激活 = 12,801,871,872
激活3个，batch_size = 3

(c)
m = beta1 * m + (1 - beta1) * grad 
3 次
v = beta2 * v + (1 - beta2) * grad**2
3 + 1 次
lr_t = lr * (1 - beta2**t)**0.5 / (1 - beta1**t)
3次
p.data-= lr_t * m / (v**0.5 + eps)
3次
p.data-= lr * weight_decay * p.data
1次
共14P
29,778,806,400

(d)
根据transformer_accounting，一次前向传播4,352,009,830,400次FLOPs
一个step 3 * 4.513e12 * 1024 = 1.387e16
A100 19.5e12 FLOP/s 计算效率50%
一个step需要 1421 s
400K个step需要 568,400,000 s (6579天)
