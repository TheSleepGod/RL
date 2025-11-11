from huggingface_hub import snapshot_download
import os

# 1. 定义模型的仓库 ID (Repo ID)
# 这是 Qwen2-7B 在 Hugging Face Hub 上的官方名称
repo_id = "Qwen/Qwen2-7B"

# 2. 定义你想要保存模型的本地路径
local_dir = "/opt/tiger/model/Qwen2-7B"

# 3. 确保目标目录存在
# os.makedirs(local_dir, exist_ok=True) # snapshot_download 会自动创建目录，但这行代码是个好习惯

print(f"开始下载模型 {repo_id} 到 {local_dir} ...")

# 4. 执行下载
# snapshot_download 会下载仓库中的所有文件
# local_dir_use_symlinks=False 可以确保文件是物理复制过来的，而不是符号链接
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    # 如果你的网络不稳定，可以加上 resume_download=True
    # resume_download=True, 
)

print("模型下载完成！")
