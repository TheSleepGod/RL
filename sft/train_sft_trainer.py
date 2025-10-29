import os
from trl import SFTTrainer
from peft import LoraConfig, TaskType
from data.sft_data_process import DataProcessor, collate_fn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

def main():
    model_name = "Qwen/Qwen2-7B"
    train_path = "data/train.jsonl"
    val_path = "data/val.jsonl"
    out_dir = "out/sft_lora_simple"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    proc = DataProcessor(tokenizer, train_path=train_path, val_path=val_path)
    train_ds, val_ds = proc.get_datasets()

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", use_cache=False)

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type=TaskType.CAUSAL_LM)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=500,
    )

    # SFTTrainer 接受 data_collator（这里我们用刚才的 collate_fn）
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        max_seq_length=2048,
        packing=False,
        dataset_text_field=None,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.save_model(os.path.join(out_dir, "final"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final"))

if __name__ == "__main__":
    main()
