import os
import torch
import wandb
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from model.model_utils import ModelManager
from loss import LOSS_MAP

@dataclass
class MyPPOConfig(PPOConfig):
    loss_name: str = "calibrated"
    loss_kwargs: dict = field(default_factory=lambda: {"alpha": 1.0})

config = MyPPOConfig(
    model_name="placeholder",
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=8,
    gradient_accumulation_steps=1,
    init_kl_coef=0.1,
    num_epochs=1,
)


def load_models(base_path: str, lora_path: str = None, device="auto"):
    mgr = ModelManager(base_path)
    if lora_path and os.path.exists(lora_path):
        model, tokenizer = mgr.merge_lora_to_model(lora_path, device_map=device)
    else:
        model, tokenizer = ModelManager.load_model_from_path(base_path, device_map=device)
    return model, tokenizer


def build_dataset(tokenizer, data_path="data/dummy.json"):
    ds = Dataset.from_json(data_path)
    def tok(x):
        x["input_ids"] = tokenizer(x["prompt"], truncation=True)["input_ids"]
        return x
    return ds.map(tok)


###############################################################################
# 主函数
###############################################################################
def main():
    # 1. wandb
    wandb.init(project="rlhf-kl-debug", name=config.loss_name, config=config)

    # 2. 加载模型 & tokenizer
    base_path = "your_base_model"
    lora_path = "your_sft_lora"           # 可选
    device = "auto"
    model, tokenizer = load_models(base_path, lora_path, device)

    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    ref_model, _ = load_models(base_path, lora_path, device)
    for p in ref_model.parameters():
        p.requires_grad = False

    # 3. 数据 & loss
    dataset = build_dataset(tokenizer)
    loss_fn = LOSS_MAP[config.loss_name]

    # 4. PPOTrainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # 5. 训练循环
    for epoch in range(config.num_epochs):
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id
            )
            # dummy reward
            rewards = [1.0] * len(response_tensors)

            # 计算 loss（这里仅示例，实际可传 prior_probs）
            logps, ref_logps, _, _ = ppo_trainer.batched_forward_pass(
                query_tensors, response_tensors
            )
            loss = loss_fn(
                logps, ref_logps,
                torch.tensor(rewards, device=logps.device),
                beta=config.init_kl_coef,
                **config.loss_kwargs
            )
            rewards_tensor = -loss  # 转成 reward 形式

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
            ppo_trainer.log_stats(stats, batch, rewards_tensor)
            wandb.log(stats)

    # 6. 保存
    ppo_trainer.save_pretrained("rlhf_ckpt")
    wandb.finish()

if __name__ == "__main__":
    main()
