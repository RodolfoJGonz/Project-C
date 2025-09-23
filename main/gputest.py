import torch

print("Torch version")
print(torch.__version__)

print("Is CUDA Reading GPU?")
print(torch.cuda.is_available())


print("How many GPUs?")
print(torch.cuda.device_count())
