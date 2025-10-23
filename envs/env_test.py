import torch, transformers
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "GPU count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Transformers:", transformers.__version__)