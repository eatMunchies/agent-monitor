"""Merge the judge's LoRA adapter into its base model and write an Ollama Modelfile.

  python training/export_judge.py --adapter training/out/judge-lora --out models/agent-judge
  docker compose exec ollama ollama create agent-judge --quantize q4_K_M -f /models/agent-judge/Modelfile

Ollama imports the merged safetensors and quantizes them to 4-bit itself, so llama.cpp
is not needed. Then set JUDGE_MODEL=agent-judge in .env and restart the gateway.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODELFILE = """FROM {path}
PARAMETER temperature 0
PARAMETER num_ctx 4096
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="models/agent-judge")
    ap.add_argument("--ollama-path", default="/models/agent-judge",
                    help="where the Ollama container sees --out (compose mounts ./models at /models)")
    a = ap.parse_args()

    # Load in bf16 (not 4-bit) so the merged weights are full precision before Ollama quantizes.
    model = AutoPeftModelForCausalLM.from_pretrained(a.adapter, torch_dtype=torch.bfloat16)
    merged = model.merge_and_unload()
    out = Path(a.out)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(out)
    (out / "Modelfile").write_text(MODELFILE.format(path=a.ollama_path))
    print(f"merged model and Modelfile written to {out}")


if __name__ == "__main__":
    main()
