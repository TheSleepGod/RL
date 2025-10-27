import os, json, argparse
from typing import Dict, Any, List
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from trl.trainer.utils import DataCollatorForCompletionOnlyLM
from peft import LoraConfig, TaskType
import yaml
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def str2bool(v: str) -> bool:
    return v.lower() in {"1","true","yes","y","t"}

def cast_val(v: str):
    # 简易类型推断（int/float/bool/str）
    if v.isdigit(): return int(v)
    try:
        fv = float(v); return fv
    except:
        pass
    if v.lower() in {"true","false","1","0","yes","no","y","n"}:
        return str2bool(v)
    return v

def apply_overrides(cfg: Dict[str, Any], overrides: List[str]):
    # 支持 --set a.b.c=val 的覆盖
    for it in overrides:
        if "=" not in it:
            print(f"[WARN] override '{it}' ignored (expected key=value).")
            continue
        k, v = it.split("=", 1)
        keys = k.split(".")
        cur = cfg
        for kk in keys[:-1]:
            if kk not in cur or not isinstance(cur[kk], dict):
                cur[kk] = {}
            cur = cur[kk]
        cur[keys[-1]] = cast_val(v)
    return cfg

def build_text_example(example: Dict, tokenizer, response_template: str, max_len: int) -> Dict[str, Any]:
    """
    统一使用简单模板，确保 response_template 唯一出现，便于只训 assistant 段：
      ### 用户
      {user}
      ### 助手
      {assistant}<eos>
    """
    msgs = example.get("messages", None)
    if msgs is None:
        # 已有 'text' 的数据直接截断
        text = example["text"]
        return {"text": text[:max_len]}
    # 取最后一轮 user->assistant
    last_ass_idx = None
    for i in range(len(msgs)-1, -1, -1):
        if msgs[i]["role"] == "assistant":
            last_ass_idx = i; break
    if last_ass_idx is None:
        return {"text": ""}  # 跳过
    last_user_idx = None
    for j in range(last_ass_idx-1, -1, -1):
        if msgs[j]["role"] == "user":
            last_user_idx = j; break
    user_text = msgs[last_user_idx]["content"] if last_user_idx is not None else ""
    ass_text  = msgs[last_ass_idx]["content"]
    text = f"### 用户\n{user_text}\n{response_template}{ass_text}{tokenizer.eos_token}"
    return {"text": text[:max_len]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--set", nargs="*", default=[], help="覆盖配置，格式 a.b.c=value")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.__dict__.get("set", []))

    model_name = cfg["model"]["name"]
    out_dir    = cfg["train"]["output_dir"]
    max_len    = int(cfg["data"].get("max_seq_length", 2048))
    response_template = cfg["data"].get("response_template", "### 助手\n")
    packing    = bool(cfg.get("packing", False))
    seed       = int(cfg.get("seed", 42))

    set_seed(seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 数据
    train_path = cfg["data"]["train_path"]
    val_path   = cfg["data"]["val_path"]
    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds   = load_dataset("json", data_files=val_path,   split="train")

    def map_fn(ex): return build_text_example(ex, tokenizer, response_template, max_len)
    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names, desc="format train")
    val_ds   = val_ds.map(map_fn,   remove_columns=val_ds.column_names,   desc="format val")

    # 模型
    torch_dtype = torch.bfloat16 if str(cfg["model"].get("torch_dtype","bf16")).lower()=="bf16" else torch.float16
    attn_impl = cfg["model"].get("attn_impl", "sdpa")

    quant = cfg.get("quantization", {})
    load_in_4bit = bool(quant.get("load_in_4bit", False))
    bnb_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype==torch.bfloat16 else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        use_cache=False,
        quantization_config=bnb_config if load_in_4bit else None
    )

    # LoRA
    lcfg = cfg["lora"]
    target_modules = lcfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    lora_cfg = LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("alpha", 32)),
        lora_dropout=float(lcfg.get("dropout", 0.05)),
        bias=lcfg.get("bias","none"),
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM
    )

    # 训练参数
    tcfg = cfg["train"]
    args_tr = TrainingArguments(
        output_dir=tcfg.get("output_dir", "out/sft_lora"),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 2)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(tcfg.get("warmup_steps", 500)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        logging_steps=int(tcfg.get("logging_steps", 20)),
        evaluation_strategy=tcfg.get("evaluation_strategy", "steps"),
        eval_steps=int(tcfg.get("eval_steps", 500)),
        save_steps=int(tcfg.get("save_steps", 500)),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        bf16=bool(tcfg.get("bf16", True)),
        fp16=bool(tcfg.get("fp16", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        ddp_find_unused_parameters=bool(tcfg.get("ddp_find_unused_parameters", False)),
        report_to=[tcfg.get("report_to","none")] if isinstance(tcfg.get("report_to","none"), str) else tcfg.get("report_to"),
    )

    # 仅监督 assistant 的 collator
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        max_seq_length=max_len,
        packing=packing,
        dataset_text_field="text",
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(os.path.join(out_dir, "final"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final"))
    print("[OK] SFT+LoRA training finished.")

if __name__ == "__main__":
    main()
