# Envs Create
1. Create envs
```bash
conda create -n <env_name> python=3.11.2
conda activate <env_name>
bash boot.sh
```

2. Test envs
```bash
python envs/env_test.py
```
If envs is activated, should print:

Torch: 2.4.0+cu121
CUDA available: True GPU count: 4
GPU 0: NVIDIA A100-SXM4-80GB
Transformers: 4.44.2

3. Test ddp_allreduce
```bash
bash envs/tests/ddp_allreduce_test.sh
```
If you use 4 GPUs, should print:

[Rank 0/4] ... reduced_x=10.0
[Rank 1/4] ... reduced_x=10.0
[Rank 2/4] ... reduced_x=10.0
[Rank 3/4] ... reduced_x=10.0
Other GPUs should change GPU set in ddp_allreduce_test.sh.
