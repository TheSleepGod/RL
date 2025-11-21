# losses.py
import torch
import torch.nn.functional as F
from typing import Optional

def vanilla_loss(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    beta: float,
    **_
) -> torch.Tensor:
    """标准 PPO KL 惩罚"""
    kl = logps - ref_logps
    return -rewards + beta * kl


def calibrated_loss(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    beta: float,
    prior_probs: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    **_
) -> torch.Tensor:
    """
    针对“先验自信错误”的 KL 加权版本
    prior_probs: 模型对错误动作的置信度 (batch,)
    """
    kl = logps - ref_logps
    if prior_probs is not None:
        confidence = torch.clamp(prior_probs, 1e-6, 1 - 1e-6)
        kl_scale = alpha * (-torch.log(confidence))  # 置信度越高惩罚越大
        kl = kl * kl_scale
    return -rewards + beta * kl


LOSS_MAP = {
    "vanilla": vanilla_loss,
    "calibrated": calibrated_loss,
}
