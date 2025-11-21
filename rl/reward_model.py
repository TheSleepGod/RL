import torch
from evaluation.metric import postprocess_gsm8k_answer

class HeuristicGSM8KRewardModel:
    def __init__(self, device):
        self.device = device

    def get_reward(self, prompts: list, responses: list, references: list) -> torch.Tensor:
        rewards = []
        for response, reference in zip(responses, references):
            pred_answer = postprocess_gsm8k_answer(response)
            ref_answer = postprocess_gsm8k_answer(reference)
            if pred_answer and ref_answer and pred_answer == ref_answer:
                reward = 2.0  # 正确答案
            else:
                reward = -1.0 # 错误答案
            
            rewards.append(reward)
            
        return torch.tensor(rewards, dtype=torch.float32).to(self.device)
