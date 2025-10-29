import os, re, json, hashlib, random
from typing import List, Dict, Optional, Tuple, Callable
from datasets import load_dataset
from build_row import BUILDER_REGISTRY

def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


gsm8k_train = BUILDER_REGISTRY["gsm8k"](split="train")
gsm8k_val = BUILDER_REGISTRY["gsm8k"](split="test")
write_jsonl("all_data/gsm8k_train.jsonl", gsm8k_train)
write_jsonl("all_data/gsm8k_val.jsonl", gsm8k_val)



