import json
import torch
import warnings
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Iterable, Callable


class SimpleDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

class DataProcessor:
    def __init__(
        self,
        tokenizer,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        response_template: str = "### 助手\n",
        max_seq_len: int = 1024,
        include_system: bool = False,
        truncate: str = "response_tail",  # or "prompt_tail"
        warn_overlength: bool = True,
        warn_limit: int = 20,             # 最多打印的详细 warning 条数（避免刷屏）
    ):
        assert truncate in {"response_tail", "prompt_tail"}
        self.tok = tokenizer
        self.train_path = train_path
        self.val_path = val_path
        self.response_template = response_template
        self.max_seq_len = int(max_seq_len)
        self.include_system = bool(include_system)
        self.truncate = truncate
        self.warn_overlength = bool(warn_overlength)
        self.warn_limit = int(warn_limit)

        # 统计与限流
        self._warn_count = 0
        self.stats = {
            "total": 0,
            "from_messages": 0,
            "from_pair": 0,
            "prompt_overlength": 0,      # prompt 已超过上限
            "truncate_response": 0,      # 回答被截断
            "truncate_prompt": 0,        # prompt 被裁剪（prompt_tail）
            "no_supervised_samples": 0,  # 该样本 labels 全为 -100
            "avg_prompt_len": 0.0,
            "avg_seq_len": 0.0,
        }
    def get_datasets(self):
        train_ds = None
        val_ds = None
        if self.train_path:
            train_examples = self._process_file(self.train_path)
            train_ds = SimpleDataset(train_examples)
        if self.val_path:
            val_examples = self._process_file(self.val_path)
            val_ds = SimpleDataset(val_examples)
        return train_ds, val_ds

    def report(self):
        total = max(self.stats["total"], 1)
        no_sup = self.stats["no_supervised_samples"]
        msg = [
            f"[DataProcessor] total={self.stats['total']}",
            f"  - from_messages={self.stats['from_messages']}, from_pair={self.stats['from_pair']}",
            f"  - prompt_overlength={self.stats['prompt_overlength']} (warned)",
            f"  - truncate_response={self.stats['truncate_response']}, truncate_prompt={self.stats['truncate_prompt']}",
            f"  - no_supervised_samples={no_sup} ({no_sup/total:.2%})",
            f"  - avg_prompt_len={self.stats['avg_prompt_len']:.1f}, avg_seq_len={self.stats['avg_seq_len']:.1f}",
        ]
        print("\n".join(msg))

    def _warn(self, text: str):
        if not self.warn_overlength:
            return
        if self._warn_count < self.warn_limit:
            warnings.warn(text)
            self._warn_count += 1
        elif self._warn_count == self.warn_limit:
            warnings.warn("[DataProcessor] warning limit reached; silencing further detailed warnings.")
            self._warn_count += 1
        else:
            pass

    def _format_role(self, role: str, content: str) -> str:
        if role == "user":
            return f"### 用户\n{content}\n"
        elif role == "assistant":
            return f"### 助手\n{content}\n"
        elif role == "system":
            return f"### 系统\n{content}\n"
        else:
            return f"### {role}\n{content}\n"

    def _from_messages(self, rec: Dict) -> Optional[Dict]:
        msgs = rec.get("messages", None)
        if not msgs:
            return None
        # 仅取最后一轮 user→assistant
        last_ass = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if last_ass is None:
            return None
        last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)

        system_prefix = ""
        if self.include_system:
            sys_msgs = [m["content"] for m in msgs if m.get("role") == "system"]
            if sys_msgs:
                system_prefix = "### 系统\n" + "\n".join(sys_msgs) + "\n"

        user_text = last_user["content"] if last_user else ""
        assistant_text = last_ass["content"]

        prompt_text = f"{system_prefix}### 用户\n{user_text}\n{self.response_template}"
        ex = self._tokenize_pair(prompt_text, assistant_text, rec)
        if ex is not None:
            self.stats["from_messages"] += 1
        return ex

    def _from_pair(self, rec: Dict) -> Optional[Dict]:
        ins = rec.get("instruction", "")
        inp = rec.get("input", "")
        out = rec.get("output", "")
        prompt_text = ""
        if ins:
            prompt_text += f"### 指令\n{ins}\n"
        if inp:
            prompt_text += f"### 输入\n{inp}\n"
        prompt_text += self.response_template
        ex = self._tokenize_pair(prompt_text, out, rec)
        if ex is not None:
            self.stats["from_pair"] += 1
        return ex

    def _make_example(self, p_ids: List[int], r_ids: List[int], rec: Dict) -> Dict:
        input_ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        attention_mask = [1] * len(input_ids)
        prompt_len = len(p_ids)

        if not r_ids or all(l == -100 for l in labels):
            self.stats["no_supervised_samples"] += 1
        self._accu_len(prompt_len, len(input_ids))

        ex = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "prompt_len": prompt_len}
        if "id" in rec: ex["id"] = rec["id"]
        if "meta" in rec: ex["meta"] = rec["meta"]
        return ex

    def _tokenize_pair(self, prompt_text: str, assistant_text: str, rec: Dict) -> Optional[Dict]:
        p_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        r_ids = self.tok(assistant_text + (self.tok.eos_token or ""), add_special_tokens=False).input_ids
        rid = rec.get("id", "<unknown>")

        if len(p_ids) >= self.max_seq_len:
            self.stats["prompt_overlength"] += 1
            self._warn(f"[DataProcessor] prompt too long ({len(p_ids)} >= {self.max_seq_len}) for id={rid}; "
                    f"keep sample with labels all -100.")
            p_ids = p_ids[-self.max_seq_len:]  # 保留尾部以尽量保留 response_template
            return self._make_example(p_ids, [], rec)

        remain = self.max_seq_len - len(p_ids)
        if len(r_ids) > remain:
            if self.truncate == "response_tail":
                r_ids = r_ids[:remain]
                self.stats["truncate_response"] += 1
                self._warn(f"[DataProcessor] truncated response tail to fit max_seq_len (id={rid}).")
            else:
                needed = len(r_ids) - remain
                if needed < len(p_ids):
                    p_ids = p_ids[needed:]
                    self.stats["truncate_prompt"] += 1
                    self._warn(f"[DataProcessor] truncated prompt head to fit response (id={rid}).")
                else:
                    r_ids = r_ids[:remain]
                    self.stats["truncate_response"] += 1
                    self._warn(f"[DataProcessor] truncated response tail to fit max_seq_len (id={rid}).")

        return self._make_example(p_ids, r_ids, rec)

    def _process_file(self, path: str) -> List[Dict]:
        out: List[Dict] = []
        for rec in read_jsonl(path):
            self.stats["total"] += 1
            ex = None
            if "messages" in rec:
                ex = self._from_messages(rec)
            elif ("instruction" in rec) or ("output" in rec):
                ex = self._from_pair(rec)
            if ex is not None:
                out.append(ex)
        return out

    def _accu_len(self, prompt_len: int, seq_len: int):
        n = self.stats["total"]
        self.stats["avg_prompt_len"] += (prompt_len - self.stats["avg_prompt_len"]) / max(n, 1)
        self.stats["avg_seq_len"] += (seq_len - self.stats["avg_seq_len"]) / max(n, 1)

def make_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        maxlen = max(len(ex["input_ids"]) for ex in batch)
        def pad(seq, pad_token, pad_len): return seq + [pad_token] * pad_len

        input_ids = []
        labels = []
        attn = []
        prompt_lens = []
        ids = []
        metas = []
        for ex in batch:
            pad_len = maxlen - len(ex["input_ids"])
            input_ids.append(pad(ex["input_ids"], pad_id, pad_len))
            labels.append(pad(ex["labels"], -100, pad_len))
            attn.append(pad(ex["attention_mask"], 0, pad_len))
            prompt_lens.append(ex["prompt_len"])
            ids.append(ex.get("id", None))
            metas.append(ex.get("meta", None))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_lens, dtype=torch.long),
            "id": ids,
            "meta": metas,
        }

    return collate_fn
