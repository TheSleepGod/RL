export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=12345 tests/ddp_allreduce_test.py
