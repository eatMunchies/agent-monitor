"""Train the Tier 1 encoder in two stages. Explained in docs/finetuning.md.

Stage 1 (jepa): self-supervised. On safe rows only, learn to predict the embedding of
each step from its context (goal, earlier steps, channel), with SIGReg keeping the
embedding space healthy. No labels needed, so any volume of normal agent traces helps.

Stage 2 (heads): supervised. Freeze the encoder and predictor, then train the
mixture-of-experts heads on labeled safe/unsafe rows to output a risk category. The
prediction error from stage 1 is one of the heads' inputs.

  python training/train_encoder.py --train data/rows/train.jsonl --val data/rows/val.jsonl \
      --stage both --out models/encoder

Writes models/encoder/model.pt and config.json, which the gateway loads when built
with WITH_ENCODER=1.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from schema import read_rows
from monitor.encoder_model import CHANNELS, MonitorModel, context_text, jepa_loss, target_text
from monitor.types import CATEGORIES


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(rows, size, shuffle=True):
    rows = list(rows)
    if shuffle:
        random.shuffle(rows)
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def housekeeping(dev, step, n_steps, loss, every=200):
    """Every `every` steps: free cached GPU blocks on Apple silicon and print progress.

    MPS keeps freed blocks cached by size and rarely hands them back. On a 16 GB Mac that
    cache grew from 9 to 17 GB in ten minutes and pushed training into swap.
    """
    if step % every:
        return
    mem = ""
    if dev.type == "mps":
        torch.mps.empty_cache()
        mem = f"  gpu memory {torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
    print(f"    step {step}/{n_steps}  loss {loss:.4f}{mem}")


def encode(tok, rows, max_len, dev):
    # Pad every batch to max_len, not to its longest row, so tensor shapes never change.
    # Changing shapes are what let the MPS allocator's cache keep growing.
    ev = [r.to_event() for r in rows]
    t = lambda texts: {k: v.to(dev) for k, v in tok(texts, truncation=True, max_length=max_len,
                                                   padding="max_length", return_tensors="pt").items()
                       if k in ("input_ids", "attention_mask")}
    ctx = t([context_text(e) for e in ev])
    tgt = t([target_text(e) for e in ev])
    chan = torch.tensor([CHANNELS.index(r.channel) for r in rows], device=dev)
    cat = torch.tensor([CATEGORIES.index(r.category) if r.label >= 0.5 else 0 for r in rows], device=dev)
    return ctx, tgt, chan, cat


def stage_jepa(model, tok, train, args, dev):
    safe = [r for r in train if r.label == 0]
    print(f"stage 1 (jepa): {len(safe)} safe rows")
    params = [p for n, p in model.named_parameters() if not n.startswith(("router", "experts"))]
    opt = torch.optim.AdamW(params, lr=args.lr)
    model.train()
    n_steps = -(-len(safe) // args.batch)
    for ep in range(args.jepa_epochs):
        total = spread = 0.0
        for step, b in enumerate(batches(safe, args.batch), 1):
            ctx, tgt, chan, _ = encode(tok, b, args.max_len, dev)
            out = model(ctx, tgt, chan)
            loss = jepa_loss(out, args.sigreg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            spread += out["zt"].detach().std(0).mean().item() * len(b)
            housekeeping(dev, step, n_steps, loss.item())
        # SIGReg pulls each embedding dimension toward std 1, so a spread near 0 means collapse.
        n = max(len(safe), 1)
        print(f"  epoch {ep + 1}: loss {total / n:.4f}  embedding spread {spread / n:.3f} (healthy is near 1)")


@torch.no_grad()
def surprise_stats(model, tok, rows, args, dev):
    model.eval()
    vals = []
    for b in batches([r for r in rows if r.label == 0], args.batch, shuffle=False):
        ctx, tgt, chan, _ = encode(tok, b, args.max_len, dev)
        vals += model(ctx, tgt, chan)["surprise"].tolist()
    t = torch.tensor(vals or [0.0, 1.0])
    return float(t.mean()), float(t.std())


def stage_heads(model, tok, train, val, args, dev):
    rows = [r for r in train if r.label != 0.5 or args.keep_controversial]
    print(f"stage 2 (heads): {len(rows)} labeled rows, {sum(r.label >= 0.5 for r in rows)} unsafe")
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(("router", "experts"))
    # Unsafe rows are rarer and matter more, so weight the non-"none" classes up.
    weights = torch.ones(len(CATEGORIES), device=dev)
    weights[1:] = args.unsafe_weight
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.head_lr)
    # Validation scores swing between epochs, so keep the best heads rather than the last.
    # "Best" is the caught rate minus the false-flag rate (Youden's J).
    best_score, best_ep, best_state = -1.0, 0, None
    n_steps = -(-len(rows) // args.batch)
    for ep in range(args.head_epochs):
        model.train()
        total = 0.0
        for step, b in enumerate(batches(rows, args.batch), 1):
            ctx, tgt, chan, cat = encode(tok, b, args.max_len, dev)
            out = model(ctx, tgt, chan)
            loss = F.cross_entropy(out["logits"], cat, weight=weights)
            # Load balancing: keep the router from sending everything to one expert.
            usage = out["gate"].mean(0)
            loss = loss + args.balance * (usage * usage.numel()).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            housekeeping(dev, step, n_steps, loss.item())
        line, score = evaluate(model, tok, val, args, dev)
        print(f"  epoch {ep + 1}: loss {total / max(len(rows), 1):.4f}  {line}")
        if score > best_score:
            best_score, best_ep = score, ep + 1
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                          if n.startswith(("router", "experts"))}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print(f"  keeping epoch {best_ep} (caught minus false flags = {best_score:.2f})")


@torch.no_grad()
def evaluate(model, tok, rows, args, dev, threshold=0.5) -> tuple[str, float]:
    """Return a readable validation line and the caught rate minus the false-flag rate."""
    rows = [r for r in rows if r.label != 0.5]
    if not rows:
        return "(no val rows)", 0.0
    model.eval()
    tp = fp = fn = tn = 0
    for b in batches(rows, args.batch, shuffle=False):
        ctx, tgt, chan, _ = encode(tok, b, args.max_len, dev)
        risk = 1 - torch.softmax(model(ctx, tgt, chan)["logits"], -1)[:, 0]
        for r, k in zip(b, risk.tolist()):
            bad, flagged = r.label >= 0.5, k >= threshold
            tp += bad and flagged
            fn += bad and not flagged
            fp += (not bad) and flagged
            tn += (not bad) and not flagged
    caught, false_flags = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
    return (f"val: caught {tp}/{tp + fn} unsafe ({caught:.0%}), "
            f"false flags {fp}/{fp + tn} safe ({false_flags:.0%})"), caught - false_flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="models/encoder")
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--stage", choices=["jepa", "heads", "both"], default="both")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--jepa-epochs", type=int, default=2)
    ap.add_argument("--head-epochs", type=int, default=5)
    ap.add_argument("--sigreg", type=float, default=0.5, help="SIGReg weight; 0.05 let embeddings collapse at batch 8")
    ap.add_argument("--balance", type=float, default=0.01, help="expert load-balancing weight")
    ap.add_argument("--unsafe-weight", type=float, default=3.0)
    ap.add_argument("--keep-controversial", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = device()
    print(f"device: {dev}")
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = MonitorModel(a.base, a.dim, a.experts).to(dev)
    if a.stage == "heads" and (out / "model.pt").exists():
        model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    train, val = read_rows(a.train), read_rows(a.val)

    if a.stage in ("jepa", "both"):
        stage_jepa(model, tok, train, a, dev)
    mean, std = surprise_stats(model, tok, val, a, dev)
    print(f"surprise on safe val rows: mean {mean:.4f}, std {std:.4f}")
    if a.stage in ("heads", "both"):
        stage_heads(model, tok, train, val, a, dev)

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps({
        "base": a.base, "dim": a.dim, "n_experts": a.experts, "max_len": a.max_len,
        "surprise_mean": mean, "surprise_std": std, "categories": CATEGORIES,
    }, indent=2))
    print(f"saved {out}/model.pt")


if __name__ == "__main__":
    main()
