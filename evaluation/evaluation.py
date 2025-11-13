# evaluation/evaluator.py

import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from .metrics import Metric  # 从同级目录的 metrics.py 导入

class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'

    def run(
        self,
        dataset,
        metrics: List[Metric],
        batch_size: int = 8,
        generation_config: Dict[str, Any] = None,
        output_file: str = None
    ) -> Dict[str, float]:
        """
        执行完整的评估流程。

        Args:
            dataset: 待评估的 Hugging Face Dataset 对象。
            metrics: 一个包含 Metric 对象的列表。
            batch_size: 批量大小。
            generation_config: 用于 .generate() 的参数。
            output_file: (可选) 保存详细预测结果的 JSON 文件路径。

        Returns:
            一个包含所有指标结果的字典。
        """
        # 1. 生成预测
        predictions = self._generate(dataset, batch_size, generation_config)
        
        # 2. 准备参考答案 (假设统一格式的数据集有 'output' 字段)
        references = [item['output'] for item in dataset]

        # 3. 计算所有指标
        all_metric_results = {}
        for metric in metrics:
            results = metric.compute(predictions, references)
            all_metric_results.update(results)

        # 4. (可选) 保存详细的对比结果
        if output_file:
            self._save_details(output_file, dataset, predictions, references, metrics)
        
        return all_metric_results

    def _generate(self, dataset, batch_size, generation_config) -> List[str]:
        """私有方法，负责生成预测。"""
        if generation_config is None:
            generation_config = {"max_new_tokens": 512, "do_sample": False}
        
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        for batch in tqdm(dataloader, desc=f"Generating predictions"):
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            
            input_len = input_ids.shape[1]
            batch_preds = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            predictions.extend(batch_preds)
            
        return predictions

    def _save_details(self, output_file, dataset, predictions, references, metrics):
        """私有方法，负责保存详细的 JSON 结果。"""
        # 假设第一个 metric 是主要的分类指标 (如 Accuracy)
        main_metric = metrics[0] if metrics else None
        
        details = []
        for i in range(len(predictions)):
            is_correct = None
            if main_metric and hasattr(main_metric, 'postprocess_func'):
                pred_ans = main_metric.postprocess_func(predictions[i])
                ref_ans = main_metric.postprocess_func(references[i])
                is_correct = pred_ans and ref_ans and pred_ans == ref_ans

            details.append({
                "prompt": dataset[i].get('instruction') or dataset[i].get('input'),
                "prediction_raw": predictions[i],
                "reference_raw": references[i],
                "is_correct": is_correct
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
