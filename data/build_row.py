import re
from datasets import load_dataset
from typing import List, Dict, Optional, Callable


ANSWER_TAG = "Answer: "

def build_gsm8k_rows(split: str = "train", max_n: Optional[int] = None) -> List[Dict]:
    dataset = load_dataset("gsm8k", "main", split=split)
    rows = []
    for index, example in enumerate(dataset):
        if max_n is not None and index >= max_n:
            break
        question = example["question"].strip()
        answer = example["answer"].strip()
        message = re.findall(r"####\s*([\-]?\d+)", answer)

        target = message[-1] if message else None
        reasoning = answer.split("####")[0].strip() if "####" in answer else answer
        
        assistant = (reasoning + f"\n{ANSWER_TAG}{target}") if target else f"{ANSWER_TAG}None"
        user = f"Q: {question}\nPlease think step-by-step, and end with “{ANSWER_TAG}<value>”."
        rows.append({
            "id": f"gsm8k-{index}-{split}",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "meta": {"task": "gsm8k", "target": target, "trap": False}
        })
    return rows

def build_hans_rows(split: str = "train", max_n: Optional[int] = None) -> List[Dict]:
    dataset = load_dataset("hans", split=split)
    label_map = {0: "entailment", 1: "non-entailment"}  # 核对你本地版本
    rows = []
    for index, example in enumerate(dataset):
        if max_n is not None and index >= max_n:
            break
        prem, hypo = example["premise"], example["hypothesis"]
        lab = label_map.get(int(example["label"]), "non-entailment")
        heuristic = example.get("heuristic", "")
        trap = heuristic in {"lexical_overlap", "subsequence", "constituent"}
        user = (
            f"Premise: {prem}\nHypothesis: {hypo}\n"
            f'Please judge the relation, and answer with “{ANSWER_TAG}entailment/non-entailment”.'
        )
        assistant = f"{ANSWER_TAG}{lab}"
        rows.append({
            "id": f"hans-{index:06d}",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "meta": {"task": "hans", "target": lab, "trap": trap, "subcase": heuristic}
        })
    return rows

BUILDER_REGISTRY: Dict[str, Callable[..., List[Dict]]] = {
    "gsm8k": build_gsm8k_rows,
    "hans": build_hans_rows,
}
