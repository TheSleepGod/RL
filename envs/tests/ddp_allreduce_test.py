import os, torch, torch.distributed as dist

def main():
    # 1) DDP 初始化
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 2) 构造张量并 all-reduce
    x = torch.ones(1, device=torch.cuda.current_device()) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)  # 期望结果 = 1+2+...+world_size = world_size*(world_size+1)/2

    # 3) 打印结果
    print(f"[Rank {rank}/{world_size}] local_rank={local_rank}, device={torch.cuda.get_device_name(local_rank)}, reduced_x={x.item():.1f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
