import torch
import wandb
from typing import Dict, Any

def get_loss_fn(name: str):
    loss_map = {
        "vanilla": vanilla_reward_loss,
        "calibrated": calibrated_reward_loss,
    }
    if name not in loss_map:
        raise ValueError(f"Unknown loss function name: '{name}'. Available: {list(loss_map.keys())}")
    return loss_map[name]


def vanilla_reward_loss(
    logps: torch.Tensor,        # 当前策略 π 对生成序列的对数概率 (per-sample)
    ref_logps: torch.Tensor,    # 参考策略 π₀ 对生成序列的对数概率 (per-sample)
    rewards: torch.Tensor,      # 来自外部奖励模型的分数 R (per-sample)
    beta: float,                # 当前的 KL 惩罚系数 β
    **kwargs                    # 吸收其他不用的参数
) -> torch.Tensor:
    # 1. 计算 KL 散度的逐样本估计
    # KL(π || π₀) ≈ log(π(a|s)) - log(π₀(a|s))
    kl = logps - ref_logps
    # 2. 计算最终的奖励信号 R'
    # R' = R - β * KL
    final_reward = rewards - beta * kl
    return final_reward

def calibrated_reward_loss(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,      # 来自外部奖励模型的分数 R
    beta: float,                # 当前的 KL 系数 β
    prior_probs: torch.Tensor,  # (关键输入) 模型对生成序列的先验置信度
    alpha: float = 0.5,         # (来自YAML) 控制自适应强度的超参数
    **kwargs
) -> torch.Tensor:
    kl = logps - ref_logps

    # 2. 设计自适应缩放因子 (scale_factor)
    #    我们的目标是识别 "自信但错误" 的情况。
    #    - "自信" -> prior_probs 很高 (接近 1)
    #    - "错误" -> rewards 很低 (例如 < 0 或接近 0)
    
    # 创建一个 "错误信号"，当 reward 低时，该信号变强。
    # 使用 torch.exp(-rewards) 是一个不错的选择，因为当 reward 变低（甚至为负）时，它的值会显著增大。
    # 你也可以设计其他函数，如 1 - sigmoid(rewards)。
    error_signal = torch.exp(-rewards)
    
    # 结合 "自信" 和 "错误" 来构建调节因子
    # 核心思想：只有当模型又自信、又犯错时，才给予强烈的调节。
    # confidence_term 可以直接用 prior_probs。
    adjustment_factor = prior_probs * error_signal
    
    # 计算最终的缩放因子
    # 基础值为 1.0，在此之上增加调节量。alpha 控制调节的强度。
    scale_factor = 1.0 + alpha * adjustment_factor
    final_reward = rewards - beta * (kl * scale_factor)

    if wandb.run:
        wandb.log({
            "loss/kl_original_mean": kl.mean().item(),
            "loss/rewards_mean": rewards.mean().item(),
            "loss/prior_probs_mean": prior_probs.mean().item(),
            "loss/scale_factor_mean": scale_factor.mean().item(),
            "loss/kl_scaled_mean": (kl * scale_factor).mean().item(),
            "loss/final_reward_mean": final_reward.mean().item(),
        }, commit=False)
    return final_reward


