# run_evaluation.py

import json
from model_utils import ModelManager
from data.sft_data_process import DataProcessor
from evaluation.evaluator import Evaluator
from evaluation.metrics import Accuracy, postprocess_gsm8k_answer

BASE_MODEL_PATH = "/opt/tiger/model/Qwen2-7B"

MODELS_TO_EVAL = {
    "sft_final": "out/sft_qwen2_7b/final",
    "rl_final": "out/rl_qwen2_7b/final", # 假设的 RL 模型
}

DATASETS_TO_EVAL = {
    "gsm8k_val": "data/all_data/gsm8k_val.jsonl",
    # "another_dataset": "data/another_dataset.jsonl",
}

TASK_METRICS = {
    "gsm8k_val": [Accuracy(postprocess_func=postprocess_gsm8k_answer)],
    # "another_dataset": [AnotherMetric(...)],
}

# --- 2. 执行交叉评估 ---
manager = ModelManager(base_model_path=BASE_MODEL_PATH)
all_results = {}

for model_name, model_path in MODELS_TO_EVAL.items():
    print(f"\n{'='*20} Evaluating Model: {model_name} {'='*20}")
    all_results[model_name] = {}

    # 加载模型
    model, tokenizer = manager.merge_lora_to_model(lora_adapter_path=model_path, device="cuda")
    
    # 初始化评估器
    evaluator = Evaluator(model, tokenizer)

    for dataset_name, dataset_path in DATASETS_TO_EVAL.items():
        print(f"\n--- On Dataset: {dataset_name} ---")
        
        # 加载数据
        data_processor = DataProcessor(tokenizer=tokenizer, val_path=dataset_path, max_seq_len=2048)
        _, dataset = data_processor.get_datasets()
        if not dataset:
            print(f"Warning: Skipping {dataset_name}, failed to load.")
            continue

        # 获取该任务对应的指标
        metrics_for_task = TASK_METRICS.get(dataset_name)
        if not metrics_for_task:
            print(f"Warning: No metrics defined for {dataset_name}. Skipping.")
            continue
            
        # 运行评估
        output_filename = f"results/{model_name}_on_{dataset_name}.json"
        metric_results = evaluator.run(
            dataset=dataset,
            metrics=metrics_for_task,
            batch_size=8,
            output_file=output_filename
        )
        
        all_results[model_name][dataset_name] = metric_results
        print(f"Results for {model_name} on {dataset_name}: {metric_results}")

print(f"\n\n{'='*30} FINAL EVALUATION SUMMARY {'='*30}")
print(json.dumps(all_results, indent=2))