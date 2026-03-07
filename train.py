import torch
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import argparse

from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_llm import (
    cross_entrophy,
    learning_rate_schedule,
    gradient_clipping,
    data_loading,
    save_checkpoint,
    load_checkpoint,
    adamW
)


def load_and_tokenize_data(data_path: str, tokenizer: Tokenizer, max_tokens: int = None) -> np.ndarray:
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text length: {len(text)} characters")
    
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}")
    
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"Truncated to {max_tokens} tokens")
    
    return np.array(tokens, dtype=np.int32)


def train_epoch(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                data: np.ndarray,
                batch_size: int,
                context_length: int,
                device: str,
                max_grad_norm: float,
                iteration: int,
                lr_schedule_type: str = "cos",
                alpha_max: float = 1e-3,
                alpha_min: float = 1e-9,
                T_w: int = 1000,
                T_c: int = 1000000) -> tuple[float, int]:
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    max_start = len(data) - context_length - 1
    num_possible_batches = max_start // batch_size
    
    pbar = tqdm(range(num_possible_batches), desc="Training")
    
    for _ in pbar:
        x, y = data_loading(data, batch_size, context_length, device)
        
        logits = model(x)
        loss = cross_entrophy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        gradient_clipping(model.parameters(), max_grad_norm)
        
        lr = learning_rate_schedule(lr_schedule_type, iteration, alpha_max, alpha_min, T_w, T_c)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        iteration += 1
        
        if num_batches % 100 == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.6f}', 'iter': iteration})
    
    avg_loss = total_loss / num_batches
    return avg_loss, iteration


def evaluate(model: torch.nn.Module,
             data: np.ndarray,
             batch_size: int,
             context_length: int,
             device: str,
             num_batches: int = 100) -> float:
    
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data_loading(data, batch_size, context_length, device)
            
            logits = model(x)
            loss = cross_entrophy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--train_file', type=str, default='TinyStoriesV2-GPT4-train.txt', help='Training data file')
    parser.add_argument('--valid_file', type=str, default='TinyStoriesV2-GPT4-valid.txt', help='Validation data file')
    parser.add_argument('--vocab_file', type=str, default='vocab_test.tsv', help='Vocabulary file')
    parser.add_argument('--merges_file', type=str, default='merges_test.txt', help='Merges file')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    
    parser.add_argument('--vocab_size', type=int, default=5030, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=128, help='Context length')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--max_train_tokens', type=int, default=10000000, help='Maximum training tokens')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    
    parser.add_argument('--lr_schedule', type=str, default='cos', help='Learning rate schedule type')
    parser.add_argument('--alpha_max', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--alpha_min', type=float, default=1e-9, help='Minimum learning rate')
    parser.add_argument('--T_w', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--T_c', type=int, default=1000000, help='Cosine decay total steps')
    
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluation interval (in epochs)')
    parser.add_argument('--save_interval', type=int, default=1, help='Checkpoint save interval (in epochs)')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading tokenizer...")
    special_tokens = ["<pad>"]
    tokenizer = Tokenizer.from_files(
        str(data_dir / args.vocab_file),
        str(data_dir / args.merges_file),
        special_tokens
    )
    
    print("Loading and tokenizing training data...")
    train_data = load_and_tokenize_data(
        str(data_dir / args.train_file),
        tokenizer,
        args.max_train_tokens
    )
    
    print("Loading and tokenizing validation data...")
    valid_data = load_and_tokenize_data(
        str(data_dir / args.valid_file),
        tokenizer
    )
    
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    if args.optimizer == 'adamw':
        optimizer = adamW(
            model.parameters(),
            lr=args.alpha_max,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.alpha_max)
    
    iteration = 0
    start_epoch = 0
    
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
        start_epoch = iteration // (len(train_data) // args.batch_size)
        print(f"Resuming from iteration {iteration}, epoch {start_epoch}")
    
    print("\nStarting training...")
    print(f"Training data size: {len(train_data)} tokens")
    print(f"Validation data size: {len(valid_data)} tokens")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Number of epochs: {args.num_epochs}")
    
    best_valid_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)
        
        epoch_start_time = time.time()
        
        train_loss, iteration = train_epoch(
            model=model,
            optimizer=optimizer,
            data=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            max_grad_norm=args.max_grad_norm,
            iteration=iteration,
            lr_schedule_type=args.lr_schedule,
            alpha_max=args.alpha_max,
            alpha_min=args.alpha_min,
            T_w=args.T_w,
            T_c=args.T_c
        )
        
        epoch_time = time.time() - epoch_start_time
        tokens_per_sec = (len(train_data) * args.batch_size) / epoch_time
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        print(f"Tokens/sec: {tokens_per_sec:.0f}")
        
        if (epoch + 1) % args.eval_interval == 0:
            print("\nEvaluating on validation set...")
            valid_loss = evaluate(
                model=model,
                data=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device
            )
            print(f"Validation loss: {valid_loss:.4f}")
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print(f"New best validation loss: {best_valid_loss:.4f}")
                
                best_checkpoint_path = output_dir / "best_model.pt"
                save_checkpoint(model, optimizer, iteration, best_checkpoint_path)
                print(f"Saved best model to {best_checkpoint_path}")
        
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
