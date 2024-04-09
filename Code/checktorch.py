import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())     # The ID of the current GPU.
print(torch.cuda.get_device_name(id))  # The name of the specified GPU, where id is an integer.
print(torch.cuda.device(id))           # The memory address of the specified GPU, where id is an integer.
print(torch.cuda.device_count())       # The amount of GPUs that are accessible.