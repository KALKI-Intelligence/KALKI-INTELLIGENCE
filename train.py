"""Train Kalki Mini on plain-text files in a corpus directory."""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import tiktoken
import torch

from kalki import KalkiConfig, KalkiMini, parameter_count


def load_tokens(corpus: Path, device: torch.device) -> torch.Tensor:
    files = sorted(corpus.rglob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {corpus}. Add licensed, clean training text first.")
    text = "\n\n".join(path.read_text(encoding="utf-8") for path in files)
    ids = tiktoken.get_encoding("gpt2").encode(text, allowed_special=set(), disallowed_special=())
    if len(ids) < 2_000:
        raise SystemExit(f"Corpus is only {len(ids):,} tokens. Add substantially more text to train.")
    print(f"Loaded {len(files)} files, {len(ids):,} tokens")
    return torch.tensor(ids, dtype=torch.long, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data"))
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/kalki-mini.pt"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    config = KalkiConfig(context_length=args.context)
    model = KalkiMini(config).to(device)
    print(f"Kalki Mini parameters: {parameter_count(model):,}")
    tokens = load_tokens(args.corpus, device)
    split = max(args.context + 2, int(len(tokens) * 0.98))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:] if len(tokens) - split > args.context + 1 else tokens[:max(args.context + 2, len(tokens) // 50)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    model.train()

    def batch(source: torch.Tensor):
        starts = torch.randint(0, len(source) - args.context - 1, (args.batch_size,), device=device)
        x = torch.stack([source[s:s + args.context] for s in starts.tolist()])
        y = torch.stack([source[s + 1:s + args.context + 1] for s in starts.tolist()])
        return x, y

    for step in range(1, args.steps + 1):
        x, y = batch(train_tokens)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                vx, vy = batch(val_tokens)
                _, val_loss = model(vx, vy)
            model.train()
            print(f"step {step:>6}/{args.steps} | train {loss.item():.3f} | val {val_loss.item():.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": args.steps}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
