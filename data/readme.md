## 数据处理模块
### pipe_line_sft
处理数据为本地文件，直接运行即可

### sft_data_process
处理本地文件为训练格式
#### DataProcessor（SFT 数据管线）说明
本文档描述 SFT 数据处理模块的输入格式、核心行为、截断与告警策略、输出字段定义与使用示例。
#### 简介
- 目标：将上游 JSONL 数据统一转换为 SFT 可直接训练的样本，且只在 assistant 段计算损失。
- 特性：
  - 支持两类输入格式：Chat 多轮、Instruction/Pair。
  - 统一模板：“### 用户”“### 助手”等分隔，保证掩码唯一。
  - 对过长样本仅抛出 warning，不丢弃；必要时对回答或提示进行截断。
  - 输出包含 `prompt_len` 等辅助字段，便于后续诊断与校准分析。
#### 适用文件类型
- 输入文件类型：JSONL（每行一个 JSON 对象）。
- 通过模块参数 `train_path`、`val_path` 指定路径。
#### 可接受的输入格式
##### Chat 多轮格式（推荐）
```
{
  "id": "sample-0001",
  "messages": [
    {"role": "system", "content": "你是一个有礼貌的助手。"},
    {"role": "user", "content": "Q: 47 + 35。请一步步推理，并在最后一行以“答案：<数字>”给出结果。"},
    {"role": "assistant", "content": "先从个位开始...\\n答案：82"},
    {"role": "user", "content": "再来一个 58+24"},
    {"role": "assistant", "content": "...\\n答案：82"}
  ],
  "meta": {"task": "math_add_2d", "target": "82", "trap": false}
}
```
- 仅取“最后一轮”的 user→assistant 作为训练样本。
- 如 `include_system=True`，会把所有 system 内容拼接到 prompt 前。
##### Instruction / Pair 格式
```
{
  "id": "sample-0002",
  "instruction": "计算两位数加法，最后一行以“答案：<数字>”。",
  "input": "58 + 24",
  "output": "先对齐位数...\\n答案：82",
  "meta": {"task":"add2","target":"82"}
}
```
- prompt 由 “### 指令”“### 输入” 构成；回答取自 `output`。
#### 行为约定
- 统一模板：
  - prompt = 可选的 system 前缀 + "### 用户\n{user}\n" + `response_template`（默认 "### 助手\n"）
  - assistant_text = 最后一条 assistant 的内容（或 pair 的 `output`）
- 训练掩码：
  - labels = [-100] × len(prompt) + response_token_ids（仅监督回答段）
- 结尾处理：
  - 在回答末尾自动追加 `tokenizer.eos_token`（若存在），帮助学习停止。
#### 长度与截断策略
- `max_seq_len` 为 prompt + response 的最大 token 数：
  - 若 `len(prompt_tokens) >= max_seq_len`：
    - 不丢弃样本；抛出 warning（限流）。
    - 保留样本但 labels 全为 -100（无监督），统计为 `no_supervised_samples`。
    - prompt 保留尾部以适配长度（通常能保留 `response_template` 标记）。
  - 若 `len(prompt_tokens) + len(response_tokens) > max_seq_len`：
    - `truncate="response_tail"`（默认）：截断回答尾部以适配长度（统计+告警）。
    - `truncate="prompt_tail"`：从 prompt 头部裁剪出空间以容纳完整回答（统计+告警）。
#### 输出样本字段定义
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| input_ids | List[int] | prompt + response 的 token 序列 |
| labels | List[int] | prompt 段为 -100；response 段为真实 token id |
| attention_mask | List[int] | 1 为有效 token；pad 为 0 |
| prompt_len | int | prompt 段的 token 数，用于回答区间的度量/诊断 |
| id | Optional[str] | 透传自输入（若存在） |
| meta | Optional[Dict] | 透传自输入（如 task/target/trap 等） |
#### collate_fn（批量对齐）
- 使用 `tokenizer.pad_token_id`（无则回退 `eos_token_id`，再回退 0）进行右侧 padding。
- 对 labels 的 padding 值为 -100；对 attention_mask 的 padding 值为 0。
- 返回的 batch 字段：
  - `input_ids`: torch.LongTensor [B, T_max]
  - `labels`: torch.LongTensor [B, T_max]
  - `attention_mask`: torch.LongTensor [B, T_max]
  - `prompt_len`: torch.LongTensor [B]
  - `id`: List[str or None]
  - `meta`: List[Dict or None]
#### 配置参数一览
| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| response_template | "### 助手\n" | 回答段落的固定前缀 |
| max_seq_len | 1024 | 最大序列长度（prompt+response） |
| include_system | False | 是否拼接所有 system 内容到 prompt 前缀 |
| truncate | "response_tail" | 长度超限时的截断策略 |
| warn_overlength | True | 是否对超长样本打印 warning |
| warn_limit | 20 | warning 的详细打印上限（超过后静默） |
#### 统计与告警
- 处理器内部维护 `stats`：
  - `total`, `from_messages`, `from_pair`
  - `prompt_overlength`（prompt 超限警告次数）
  - `truncate_response`, `truncate_prompt`
  - `no_supervised_samples`（labels 全为 -100 的样本数）
  - `avg_prompt_len`, `avg_seq_len`（在线平均）
- 通过 `processor.report()` 打印汇总。
- warning 限流：最多打印 `warn_limit` 条详细 warning；随后静默。