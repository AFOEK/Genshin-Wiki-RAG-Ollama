import json
import random
from pathlib import Path

src = Path("genshin_lora_candidates.jsonl")
train = Path("genshin_lora_train.jsonl")
val = Path("genshin_lora_val.jsonl")

rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()]
random.Random(1337).shuffle(rows)

n_val = max(1, int(len(rows) * 0.05))

val_rows = rows[:n_val]
train_rows = rows[n_val:]

train.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n", encoding="utf-8")

val.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + "\n", encoding="utf-8")

print("train", len(train_rows), "val", len(val_rows))