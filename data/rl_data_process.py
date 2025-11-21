import json
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Iterable, Callable
from utils import SimpleDataset, read_jsonl


class PPODataProcessor:
    def __init__(
        self,
        tokenizer,
        data_path: str,
        response_template: str = "### 助手\n",
        max_prompt_len: int = 512,
        include_system: bool = False,
    ):
        self.tok = tokenizer
        self.data_path = data_path
        self.response_template = response_template
        self.max_prompt_len = int(max_prompt_len)
        self.include_system = bool(include_system)
        
        self.stats = {"total": 0, "processed": 0, "prompt_overlength": 0}

    def get_dataset(self) -> SimpleDataset:
        examples = self._process_file(self.data_path)
        return SimpleDataset(examples)

    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        dataset = self.get_dataset()
        collate_fn = make_ppo_collate_fn(self.tok)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def report(self):
        """打印处理统计信息。"""
        msg = [
            f"[PPODataProcessor] Report for: {self.data_path}",
            f"  - Total records read: {self.stats['total']}",
            f"  - Records processed (valid prompts): {self.stats['processed']}",
            f"  - Prompts truncated due to overlength: {self.stats['prompt_overlength']}",
        ]
        print("\n".join(msg))

    def _process_file(self, path: str) -> List[Dict]:
        out: List[Dict] = []
        for rec in read_jsonl(path):
            self.stats["total"] += 1
            processed_rec = self._process_record(rec)
            if processed_rec:
                self.stats["processed"] += 1
                out.append(processed_rec)
        return out

    def _process_record(self, rec: Dict) -> Optional[Dict]:
        prompt_text = None
        
        if "messages" in rec:
            msgs = rec.get("messages", [])
            last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
            system_prefix = ""
            if self.include_system:
                sys_msgs = [m["content"] for m in msgs if m.get("role") == "system"]
                if sys_msgs:
                    system_prefix = "### 系统\n" + "\n".join(sys_msgs) + "\n"
            user_text = last_user["content"] if last_user else ""
            prompt_text = f"{system_prefix}### 用户\n{user_text}\n{self.response_template}"

        elif ("instruction" in rec) or ("output" in rec):
            ins = rec.get("instruction", "")
            inp = rec.get("input", "")
            prompt_text = f"### 指令\n{ins}\n" if ins else ""
            if inp: prompt_text += f"### 输入\n{inp}\n"
            prompt_text += self.response_template
        
        if prompt_text is None:
            return None

        tokenized_prompt = self.tok(prompt_text, add_special_tokens=False).input_ids
        if len(tokenized_prompt) > self.max_prompt_len:
            self.stats["prompt_overlength"] += 1
            start_index = len(tokenized_prompt) - self.max_prompt_len
            tokenized_prompt = tokenized_prompt[start_index:]
            prompt_text = self.tok.decode(tokenized_prompt)
            warnings.warn(f"Prompt (id={rec.get('id', 'N/A')}) was truncated to {self.max_prompt_len} tokens.")

        return {
            "prompt_text": prompt_text,
            "id": rec.get("id"),
            "meta": rec.get("meta"),
        }

def make_ppo_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn("tokenizer.pad_token is None. Using tokenizer.eos_token as pad_token.")

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [ex["prompt_text"] for ex in batch]
        tokenized_batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        tokenized_batch["raw_prompts"] = prompts
        tokenized_batch["ids"] = [ex.get("id") for ex in batch]
        return tokenized_batch

    return collate_fn
