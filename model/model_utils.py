import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Tuple

class ModelManager:
    def __init__(self, base_model_path: str):
        if not os.path.exists(base_model_path):
            print(f"[Warning] 基础模型路径 '{base_model_path}' 不存在。请确保这是一个有效的 Hugging Face 模型标识符或本地路径。")
        self.base_model_path = base_model_path
        print(f"ModelManager initialized with base model path: {self.base_model_path}")

    def merge_lora_to_model(
        self, 
        lora_adapter_path: str, 
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu"
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        print(f"--- Method 1: Merging LoRA to model in memory ---")
        print(f"Loading base model from: {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch_dtype,
            device_map=device
        )

        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path, device_map=device)

        print("Merging LoRA weights into the base model...")
        merged_model = lora_model.merge_and_unload()
        print("✅ Merge complete.")
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)

        return merged_model, tokenizer

    def merge_and_save_model(
        self, 
        lora_adapter_path: str, 
        save_path: str,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        print(f"--- Method 2: Merging and saving model to disk ---")
        merged_model, tokenizer = self.merge_lora_to_model(
            lora_adapter_path=lora_adapter_path,
            torch_dtype=torch_dtype,
            device="cpu"
        )

        print(f"Saving merged model to: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ Full merged model and tokenizer saved successfully to {save_path}.")

    @staticmethod
    def load_model_from_path(
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        use_cache: bool = True
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        print(f"--- Method 3: Loading a full model from path ---")
        print(f"Loading model and tokenizer from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_cache=use_cache
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"✅ Model and tokenizer loaded successfully from {model_path}.")
        return model, tokenizer

