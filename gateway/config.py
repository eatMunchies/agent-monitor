"""Gateway settings, all from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://ollama:11434")
    agent_model: str = os.getenv("AGENT_MODEL", "qwen3:8b")
    judge_model: str = os.getenv("JUDGE_MODEL", "llama3.2:3b")
    sandbox_url: str = os.getenv("SANDBOX_URL", "http://sandbox:9000")
    policy_path: str = os.getenv("POLICY_PATH", "/app/policy/policy.yaml")
    encoder_dir: str = os.getenv("ENCODER_DIR", "/app/models/encoder")
    audit_log: str = os.getenv("AUDIT_LOG", "/logs/audit.jsonl")
    mode: str = os.getenv("MONITOR_MODE", "cascade")
    max_steps: int = int(os.getenv("MAX_STEPS", "12"))
    # Seconds to wait for one model turn. CPU-only Ollama (Docker on a Mac) needs minutes.
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "600"))
    judge_timeout: float = float(os.getenv("JUDGE_TIMEOUT", "60"))
    # Cap on tokens per agent turn (reasoning + answer), so a runaway thought cannot stall a run.
    agent_num_predict: int = int(os.getenv("AGENT_NUM_PREDICT", "4096"))
    # Bearer token for /v1/* (you, eval scripts, chat clients). The sandbox never gets it.
    gateway_token: str = os.getenv("GATEWAY_TOKEN", "")
    # Shared secret between the egress proxy and /check/connection.
    check_token: str = os.getenv("CHECK_TOKEN", "")


settings = Settings()
