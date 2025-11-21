# ppo/reward_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

class RewardModel:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化奖励模型。
        
        Args:
            model_name (str): Hugging Face Hub 上的奖励模型名称。
            device (str): 模型所在的设备。
        """
        print(f"Loading Reward Model: {model_name}...")
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Reward Model loaded.")

    @torch.no_grad()
    def get_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        为一批 (prompt, response) 对计算奖励分数。

        Args:
            prompts (List[str]): prompt 列表。
            responses (List[str]): 模型生成的 response 列表。

        Returns:
            torch.Tensor: 包含每个样本奖励分数的张量，形状为 [batch_size]。
        """
        texts = [f"Question: {p}\n\nAnswer: {r}" for p, r in zip(prompts, responses)]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024, # 根据奖励模型调整
        ).to(self.device)

        # 获取 logits 并返回分数
        outputs = self.model(**inputs)
        rewards = outputs.logits.squeeze(-1) # 形状从 [batch_size, 1] 变为 [batch_size]
        
        return rewards
