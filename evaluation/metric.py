from typing import List, Dict, Callable, Protocol
import re

class Metric(Protocol):
    """评估指标的接口 (Protocol)。"""
    name: str

    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算并返回指标分数。"""
        ...

class Accuracy(Metric):
    name = "accuracy"

    def __init__(self, postprocess_func: Callable[[str], str]):
        self.postprocess_func = postprocess_func

    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        correct_count = 0
        for pred, ref in zip(predictions, references):
            pred_answer = self.postprocess_func(pred)
            ref_answer = self.postprocess_func(ref)
            if pred_answer and ref_answer and pred_answer == ref_answer:
                correct_count += 1
        
        total_count = len(predictions)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            self.name: accuracy,
            "correct_count": float(correct_count),
            "total_count": float(total_count)
        }


def postprocess_gsm8k_answer(text: str) -> str:
    """从 GSM8K 格式的文本中提取最终数值答案。"""
    try:
        # 查找 "####" 标记
        marker = "####"
        last_marker_idx = text.rfind(marker)
        if last_marker_idx != -1:
            answer_part = text[last_marker_idx + len(marker):].strip()
            # 清理答案，移除逗号等
            return "".join(c for c in answer_part if c.isdigit() or c in ['.', '-'])
        
        # 如果没有 "####"，尝试提取文本中最后一个数字
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        return numbers[-1] if numbers else ""
    except Exception:
        return ""

