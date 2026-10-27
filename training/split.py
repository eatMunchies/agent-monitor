"""Split rows into train / val / test by environment, not by row.

Rows from one environment (a tool suite, a website, a task family) look alike. A random
split puts near-duplicates on both sides and inflates scores. Grouping by env makes the
test set a set of environments the models never saw.

  python training/split.py data/rows/all.jsonl --out data/rows --test 0.2 --val 0.1
"""

import argparse
import random
from collections import defaultdict

from schema import read_rows, write_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--out", default="data/rows")
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    by_env = defaultdict(list)
    for r in read_rows(a.rows):
        by_env[f"{r.source}:{r.env}"].append(r)
    envs = sorted(by_env)
    random.Random(a.seed).shuffle(envs)
    total = sum(len(v) for v in by_env.values())

    split, filled = {}, {"test": 0, "val": 0}
    for env in envs:
        n = len(by_env[env])
        # An env goes to test or val only if it does not overshoot the target by more than
        # 20%, so one huge env cannot swallow the test set.
        if filled["test"] + n <= 1.2 * a.test * total:
            split[env] = "test"
        elif filled["val"] + n <= 1.2 * a.val * total:
            split[env] = "val"
        else:
            split[env] = "train"
        filled[split[env]] = filled.get(split[env], 0) + n

    for name in ("train", "val", "test"):
        rows = [r for env in envs if split[env] == name for r in by_env[env]]
        n = write_rows(rows, f"{a.out}/{name}.jsonl")
        unsafe = sum(1 for r in rows if r.label >= 0.5)
        print(f"{name:5} {n:6} rows  {unsafe:6} unsafe  {sum(1 for e in envs if split[e] == name)} envs")


if __name__ == "__main__":
    main()
