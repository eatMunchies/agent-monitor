"""QLoRA fine-tune of the Tier 2 judge. Explained in docs/finetuning.md.

Each labeled row becomes one chat example: the judge's real system prompt and the
event rendered exactly as monitor/judge_prompt.py renders it in production, followed
by the JSON verdict we want back. Loss is computed on the verdict only.

QLoRA: the base model is loaded in 4-bit (bitsandbytes) and frozen; small low-rank
adapter matrices (LoRA) are trained on top. A 3B model fits in about 6 GB of VRAM
this way, an 8B in about 10 GB, so both fit the RTX 3060.

  python training/finetune_judge.py --train data/rows/train.jsonl --val data/rows/val.jsonl \
      --base meta-llama/Llama-3.2-3B-Instruct --out training/out/judge-lora

Then run training/export_judge.py to merge the adapter and load it into Ollama.
"""

from __future__ import annotations

import argparse
import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import yaml
from schema import read_rows
from monitor import judge_prompt

TEMPLATES = {
    "none": "An ordinary step toward the user's goal.",
    "exfiltration": "Sends data or secrets somewhere the task does not need.",
    "destructive": "Deletes or damages data the user did not ask to change.",
    "prompt_injection": "Follows instructions that came from tool output, not the user.",
    "privilege": "Reaches for credentials or access beyond the task.",
    "off_task": "Does something unrelated to the user's goal.",
    "deception": "Hides or misreports what the agent is doing.",
    "policy": "Breaks the system policy.",
}


def example(row, policy_summary: str) -> dict:
    unsafe = row.label >= 0.5
    verdict = {
        "decision": "block" if unsafe else "allow",
        "category": row.category if unsafe else "none",
        # Controversial rows (0.5) teach the judge to be less sure, which the cascade
        # turns into an escalation instead of a confident call.
        "confidence": 0.6 if row.label == 0.5 else 0.9,
        "reason": (row.reason or TEMPLATES[row.category if unsafe else "none"])[:200],
    }
    return {
        "prompt": [
            {"role": "system", "content": judge_prompt.SYSTEM},
            {"role": "user", "content": judge_prompt.user_message(row.to_event(), policy_summary)},
        ],
        "completion": [{"role": "assistant", "content": json.dumps(verdict)}],
    }


def balanced(rows, ratio: float, seed: int):
    """Cap safe rows at ratio x unsafe rows so the judge does not learn to always allow."""
    bad = [r for r in rows if r.label >= 0.5]
    good = [r for r in rows if r.label < 0.5]
    random.Random(seed).shuffle(good)
    return bad + good[: int(len(bad) * ratio) or len(good)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--policy", default="policy/policy.yaml")
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF id. Meta repos are gated: accept the license and run `hf auth login` first.")
    ap.add_argument("--out", default="training/out/judge-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--safe-ratio", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    summary = yaml.safe_load(open(a.policy))["summary"]
    train = [example(r, summary) for r in balanced(read_rows(a.train), a.safe_ratio, a.seed)]
    val = [example(r, summary) for r in read_rows(a.val)[:300]]
    print(f"{len(train)} train examples, {len(val)} val examples")

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(val) if val else None,
        peft_config=LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"]),
        args=SFTConfig(
            output_dir=a.out,
            num_train_epochs=a.epochs,
            learning_rate=a.lr,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            max_length=a.max_len,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch" if val else "no",
            save_strategy="epoch",
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            report_to="none",
            seed=a.seed,
        ),
    )
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"adapter saved to {a.out}")


if __name__ == "__main__":
    main()
