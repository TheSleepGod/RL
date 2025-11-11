import os, argparse, json, random, numpy as np, torch, yaml
from typing import Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType
from data.sft_data_process import DataProcessor, make_collate_fn
from transformers import Trainer
from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cast_val(v: str):
    # 简易类型推断（int/float/bool/str）
    if v.isdigit(): return int(v)
    try:
        return float(v)
    except:
        pass
    if v.lower() in {"true","false","1","0","yes","no","y","n"}:
        return v.lower() in {"true","1","yes","y"}
    return v

def apply_overrides(cfg: Dict[str, Any], overrides):
    # 支持 --set a.b.c=val
    for it in overrides or []:
        if "=" not in it:
            print(f"[WARN] override '{it}' ignored (expected key=value)."); continue
        k, v = it.split("=", 1)
        cur = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            if kk not in cur or not isinstance(cur[kk], dict): cur[kk] = {}
            cur = cur[kk]
        cur[ks[-1]] = cast_val(v)
    return cfg

def ensure_schema(ds, required=("input_ids","labels","attention_mask")):
    sample = ds[0]
    miss = [k for k in required if k not in sample]
    if miss:
        raise ValueError(f"Dataset schema missing keys: {miss}. "
                         f"Expected tokenized JSONL with fields {required}. "
                         f"Got keys={list(sample.keys())[:10]} ...")

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件")
    parser.add_argument("--set", nargs="*", default=[], help="覆盖配置，格式 a.b.c=value")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)

    model_name = cfg["model"]["name"]
    out_dir    = cfg["train"]["output_dir"]
    seed       = int(cfg.get("seed", 42))
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_path = cfg["data"]["train_path"]
    val_path   = cfg["data"]["val_path"]

    train_ds, val_ds = DataProcessor(tokenizer, train_path = train_path, val_path = val_path).get_datasets()

    ensure_schema(train_ds); ensure_schema(val_ds)
    collate_fn = make_collate_fn(tokenizer)

    torch_dtype = torch.bfloat16 if str(cfg["model"].get("torch_dtype","bf16")).lower() == "bf16" else torch.float16
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
        dtype=torch_dtype,
        attn_implementation=attn_impl,
        use_cache=False,
        quantization_config=bnb_config if load_in_4bit else None
    )

    lcfg = cfg["lora"]
    lora_cfg = LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("alpha", 32)),
        lora_dropout=float(lcfg.get("dropout", 0.05)),
        bias=lcfg.get("bias","none"),
        target_modules=lcfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tcfg = cfg["train"]
    train_args = TrainingArguments(
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
        # evaluation_strategy=tcfg.get("evaluation_strategy", "steps"),
        eval_steps=int(tcfg.get("eval_steps", 500)),
        save_steps=int(tcfg.get("save_steps", 500)),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        bf16=bool(tcfg.get("bf16", False)),
        fp16=bool(tcfg.get("fp16", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        ddp_find_unused_parameters=bool(tcfg.get("ddp_find_unused_parameters", False)),
        report_to=[tcfg.get("report_to","none")] if isinstance(tcfg.get("report_to","none"), str) else tcfg.get("report_to"),
    )
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn
    )

    trainer.train()
    os.makedirs(os.path.join(out_dir, "final"), exist_ok=True)
    trainer.save_model(os.path.join(out_dir, "final"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final"))
    print("[OK] SFT+LoRA training finished.")

if __name__ == "__main__":
    main()
