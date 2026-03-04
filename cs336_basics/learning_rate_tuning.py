import torch
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from train_llm import SGD, adamW


weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
for learning_rate in [1, 10, 100, 1000]:
    opt = SGD([weights], lr=learning_rate)
    print(f"===== start trainning when lr == {learning_rate} =====")
    start_time = time.time()
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step
    end_time = time.time()
    elapsed_time = end_time - start_time # 计算运行时间
    print(f"程序运行时间：{elapsed_time}秒")

# lr == 100 最佳
learning_rate = 0.001
opt = adamW([weights], lr=learning_rate)
print(f"===== start trainning when lr == {learning_rate} =====")
start_time = time.time()
for t in range(1, 11):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step
end_time = time.time()
elapsed_time = end_time - start_time # 计算运行时间
print(f"程序运行时间：{elapsed_time}秒")