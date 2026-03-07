import torch
import sys
import os
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
from typing import Tuple
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import IO, Any, BinaryIO
from typing import Optional
import numpy as np
import numpy.typing as npt
import torch
import math

sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from attention import softmax


def cross_entrophy(inputs: Float[torch.Tensor, "batch_size ... vocab_size"],
                   targets: Float[torch.Tensor, "batch_size ..."]):
    """
    计算批次上的平均交叉熵损失
    输入：
        inputs: (batch_size, ..., vocab_size) → 未归一化的logits
        targets: (batch_size, ...) → 真实token的索引
    输出：
        标量张量 → 批次平均损失
    """
    # 使用log-softmax log(softmax(x)) = x - log(sum(exp(x)))
    batch_size = inputs.shape[0]

    max_vals = torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_vals), dim=-1))
    log_probs = -(inputs - max_vals)[torch.arange(batch_size), targets] + log_sum_exp

    return log_probs.mean()

def learning_rate_schedule(type_sche: str = "cos",
                           t: int = 1,
                           alpha_max: float = 1,
                           alpha_min: float = 1e-9,
                           T_w: int = 1e3,
                           T_c: int = 1e6) -> float:
    if(type_sche == "cos" or type_sche == "cosine"):
        if(t < T_w):
            return t / T_w * alpha_max
        elif(T_w <= t and t <= T_c):
            return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
        else:
            return alpha_min
        
    else:
        raise NotImplementedError

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return  # 无梯度可裁剪
    
    eps = 1e-6
    
    l2_norm_p = 0.0
    for g in grads:
        l2_norm_p += torch.sum(g**2)
    l2_norm_p = torch.sqrt(l2_norm_p)

    clipping_ef = min(1.0, max_l2_norm / (l2_norm_p + eps))

    for g in grads:
        g *= clipping_ef

def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    if max_start <= 0:
        raise ValueError("数据集长度小于指定的context_length")
    
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    x_batch = []
    y_batch = []

    for s in starts:
        seq = dataset[s: s + context_length + 1] # 长度 context_length + 1
        x_batch.append(seq[:-1]) # 往前偏移
        y_batch.append(seq[1:]) # 往后偏移

    x = torch.tensor(x_batch, device=device)
    y = torch.tensor(y_batch, device=device)

    return (x, y)

def save_checkpoint(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]):

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer) -> int:

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data-= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    

class adamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr : float = 1e-3,
                 betas : Tuple[float, float] = (0.9, 0.99),
                 eps : float = 1e-5,
                 weight_decay : float = 1e-2
                 ):
        # 参数合法性检查
        if lr < 0.0:
            raise ValueError(f"无效学习率: {lr}")
        if eps < 0.0:
            raise ValueError(f"无效epsilon值: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"无效weight_decay值: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效betas参数: {betas}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                # 初始化状态（首次更新时）
                t = state.get("t", 1)  # 迭代次数（初始为1）
                m = state.get("m", torch.zeros_like(grad))  # 一阶矩估计（动量）
                v = state.get("v", torch.zeros_like(grad))  # 二阶矩估计
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                # print(t, lr, beta1, beta2)
                lr_t = lr * (1 - beta2**t)**0.5 / (1 - beta1**t)
                # print(v)
                p.data-= lr_t * m / (v**0.5 + eps)
                p.data-= lr * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss



if __name__ == "__main__":
    input_t = torch.tensor(
        [[2.0, 1.0, 0.1, 0.5, 1.5],   # 样本0: 5个类别的logits
        [0.5, 2.5, 1.0, 0.2, 0.1],   # 样本1
        [1.0, 0.5, 2.0, 1.5, 0.2]]
        )   # 样本2
    
    targets = torch.tensor([0, 1, 2])


    print(cross_entrophy(input_t, targets))

    # 2 4 5
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )

    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    print(f"inputs.shape: {inputs.shape}\n"
          f"targets.shape: {targets.shape}\n"
          f"inputs.view(-1, inputs.size(-1)): {inputs.view(-1, inputs.size(-1))}\n"
          f"targets.view(-1): {targets.view(-1)}")
    print(cross_entrophy(inputs.view(-1, inputs.size(-1)), targets.view(-1)))

    
    