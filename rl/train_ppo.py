import os, argparse, yaml, torch, wandb
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead 

from model.model_utils import ModelManager
from data.rl_data_process import PPODataProcessor
from rl.loss import get_loss_fn
from rl.reward_model import RewardModel

os.environ['WANDB_API_KEY'] = 'f4d7b19a6f19dc9375f58c933e3da27637b3c9b6'

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_ppo_models_from_manager(base_path, lora_path=None):
    mgr = ModelManager(base_path)
    if lora_path and os.path.exists(lora_path):
        print(f"--- Loading base model '{base_path}' and merging LoRA '{lora_path}' ---")
        base_model, tokenizer = mgr.merge_lora_to_model(lora_adapter_path=lora_path)
    else:
        print(f"--- Loading full model from '{base_path}' ---")
        base_model, tokenizer = ModelManager.load_model_from_path(model_path=base_path, use_cache=False)

    print("Wrapping loaded models with Value Head for PPO...")
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    print("✅ Policy and Reference models wrapped successfully.")
    
    for param in ref_model.parameters():
        param.requires_grad = False
        
    return ppo_model, ref_model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="PPO 配置文件路径")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))

    print("="*20 + " Loading Models & Tokenizer " + "="*20)
    ppo_model, ref_model, tokenizer = load_ppo_models_from_manager(
        cfg["base_model_path"], cfg.get("sft_adapter_path")
    )
    # 设置 tokenizer 的 padding 属性
    if tokenizer.pad_token is None:
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Models & Tokenizer loaded.")
    # 2. 创建 PPOConfig 对象
    print("\nCreating PPOConfig...")
    ppo_config = PPOConfig(**cfg["ppo"])
    print("✅ PPOConfig created.")
    # 3. 初始化 PPOTrainer，并直接传入所有必需的对象
    print("\nInitializing PPOTrainer...")
    ppo_trainer = PPOTrainer(
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer, # <-- 直接在这里传入加载好的 tokenizer
        config=ppo_config,
        dataset=None
    )
    print("✅ PPOTrainer initialized successfully.")

    if tokenizer.pad_token is None:
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

    reward_model = RewardModel(cfg["reward_model_path"], device=ppo_trainer.accelerator.device)

    print("\n" + "="*20 + " Preparing Data " + "="*20)
    processor = PPODataProcessor(
        tokenizer=tokenizer,
        data_path=cfg["data"]["path"],
        max_prompt_len=cfg["data"]["max_prompt_len"],
        response_template=cfg["data"]["response_template"],
    )
    dataloader = processor.get_dataloader(batch_size=cfg["ppo"]["batch_size"], shuffle=True)

    loss_fn = get_loss_fn(cfg["loss"]["name"])
    loss_kwargs = cfg["loss"].get("kwargs", {})

    print("\n" + "="*20 + " Starting PPO Training " + "="*20)
    if cfg["train"]["report_to"] == "wandb":
        wandb.init(project=cfg["train"]["wandb_project"], config=cfg, name=f"ppo_{cfg['loss']['name']}")

    for epoch in range(cfg["train"]["num_epochs"]):
        print(f"--- Starting Epoch {epoch + 1}/{cfg['train']['num_epochs']} ---")
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            query_tensors = batch["input_ids"].to(ppo_trainer.accelerator.device)
            
            # a. 生成 response
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=cfg["train"]["generation_max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
            )
            # 从 query+response 中提取 response 部分
            response_only_tensors = [r[q.size(0):] for q, r in zip(query_tensors, response_tensors)]
            
            # 解码文本
            prompts_text = batch["raw_prompts"]
            responses_text = tokenizer.batch_decode(response_only_tensors, skip_special_tokens=True)

            # b. 获取真实奖励
            rewards = reward_model.get_rewards(prompts_text, responses_text)

            # c. 计算 logprobs
            logprobs, ref_logprobs, values, masks = ppo_trainer.batched_forward_pass(
                query_tensors, response_only_tensors
            )
            
            # d. 计算自定义损失/最终奖励
            with torch.no_grad():
                prior_probs = torch.exp(logprobs).mean(dim=1)
            
            loss_input = {
                "logps": logprobs, "ref_logps": ref_logprobs, "rewards": rewards,
                "beta": ppo_trainer.current_kl_coef, "prior_probs": prior_probs,
                **loss_kwargs
            }
            ppo_rewards = loss_fn(**loss_input)

            # e. 执行 PPO 步骤
            stats = ppo_trainer.step(query_tensors, response_only_tensors, ppo_rewards)
            
            # f. 记录日志
            stats["env/reward_mean"] = rewards.mean().item()
            ppo_trainer.log_stats(stats, batch, ppo_rewards)

    # 7. 保存最终模型
    print("\n" + "="*20 + " Saving Final Model " + "="*20)
    final_dir = os.path.join(cfg["train"]["output_dir"], "final")
    # 保存时，我们只保存策略模型的核心部分，而不是整个 ValueHead 包装器
    unwrapped_model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    unwrapped_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[OK] PPO training finished. Core model saved to {final_dir}")

if __name__ == "__main__":
    main()
