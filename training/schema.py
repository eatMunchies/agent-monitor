"""The one row format every dataset is converted into.

A row is one monitored event plus its label, so the same file trains the encoder heads,
fine-tunes the judge, and drives offline eval. Rows serialize as JSON lines:

  {"id": "rjudge-12-3", "source": "rjudge", "env": "finance", "channel": "tool_call",
   "goal": "...", "history": ["..."], "content": "...", "label": 1,
   "category": "exfiltration", "reason": "...", "meta": {}}

label: 0 safe, 1 unsafe. Sources with a "controversial" middle grade keep it as 0.5;
training scripts decide whether to drop it or treat it as a soft label.
env: the environment or domain the example came from. Splits group by env so the test
set measures generalization to unseen environments, not memorized ones.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitor.types import CATEGORIES, Channel, Event  # noqa: E402


@dataclass
class Row:
    id: str
    source: str
    env: str
    channel: str
    goal: str
    content: str
    label: float
    category: str = "none"
    history: list[str] = field(default_factory=list)
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        Channel(self.channel)  # raises on a typo
        if self.category not in CATEGORIES:
            self.category = "policy" if self.label > 0 else "none"
        if self.label == 0:
            self.category = "none"

    def to_event(self) -> Event:
        return Event(self.id, Channel(self.channel), self.content, self.goal, self.history, self.meta)


def write_rows(rows, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_rows(path: str | Path) -> list[Row]:
    with open(path, encoding="utf-8") as f:
        return [Row(**json.loads(line)) for line in f if line.strip()]


# Keyword map from dataset risk descriptions to our categories. First match wins.
CATEGORY_KEYWORDS = [
    ("prompt_injection", ["inject", "indirect", "attacker instruction", "third-party instruction"]),
    ("exfiltration", ["leak", "exfiltrat", "privacy", "data theft", "send data", "disclos", "steal"]),
    ("destructive", ["delete", "destroy", "corrupt", "damage", "wipe", "physical harm"]),
    ("privilege", ["permission", "privilege", "credential", "access control", "password", "unauthor"]),
    ("deception", ["decept", "fraud", "mislead", "phish", "scam", "impersonat"]),
    ("off_task", ["unrelated", "off-task", "beyond the scope", "not requested"]),
]


def guess_category(text: str) -> str:
    t = (text or "").lower()
    for cat, words in CATEGORY_KEYWORDS:
        if any(w in t for w in words):
            return cat
    return "policy"
