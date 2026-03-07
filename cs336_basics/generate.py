import torch
from typing import Optional
from jaxtyping import Float, Int

def top_p_sampling(probs: Float[torch.Tensor, "vocab_size"], top_p: float) -> Float[torch.Tensor, "..."]:
    if not (0 < top_p and top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    cum = torch.cumsum(sorted_probs, dim=-1)

    keep = cum <= top_p
    keep[..., 0] = True # 保证至少有一个

    filtered_sorted_probs = sorted_probs * keep.to(sorted_probs.dtype)
    filtered_sorted_probs = filtered_sorted_probs / torch.sum(filtered_sorted_probs, dim=-1, keepdim=True)
    
    out = torch.zeros_like(probs)
    out.scatter_(dim=-1, index=sorted_idx, src=filtered_sorted_probs) # 按原顺序输出

    return out

@torch.no_grad
def generate(
    model: torch.nn.Module,
    prompt_ids: Int[torch.Tensor, "sequence_length"],
    *,
    end_token_id: int,
    max_new_tokens: int = 128,
    temperature: float,
    top_p: float
) -> torch.Tensor:
    if(prompt_ids.dim() != 1):
        raise ValueError(f"prompts_id must be 1D (sequence_length, ), got {prompt_ids.shape}")
    
    if(prompt_ids.dtype != torch.long):
        prompt_ids.to(torch.long)

    if(max_new_tokens < 0):
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    
    model_was_training = model.train
    model.eval()

    device = next(model.parameters()).device
    out = prompt_ids.to(device)

    context_length: Optional[int] = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        if context_length is not None and out.numel() > context_length:
            inp = out[-context_length:]
        else:
            inp = out
        
        logits = model(inp.unsqueeze(0))  # (1, S, V)
        next_logits = logits[0, -1, :]    # (V,)

        # Greedy decoding if temperature == 0
        if temperature == 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            if temperature < 0.0:
                raise ValueError(f"temperature must be >= 0, got {temperature}")
            
            scaled = next_logits / float(temperature)
            probs = torch.softmax(scaled, dim=-1)

            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)
            
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        
        out = torch.cat([out, torch.tensor([next_id], device=device, dtype=torch.long)], dim=0)

        if next_id == int(end_token_id):
            break
    
    if model_was_training:
        model.train()

    return out



if __name__ == "__main__":
    print("===== test top_p =====")
    probs = torch.tensor([[0.7, 0.1, 0.1, 0.05, 0.03, 0.02],
                          [0.15, 0.15, 0.6, 0.1, 0.06, 0.03]])

    print(top_p_sampling(probs, 0.95))

    print("===== test generate =====")
    
