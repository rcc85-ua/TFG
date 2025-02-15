import torch

if torch.cuda.is_available():
    print(f"CUDA está disponible. Usando: {torch.cuda.get_device_name(0)}")
    print(f"Cantidad de GPUs disponibles: {torch.cuda.device_count()}")
    print(f"GPU actual: {torch.cuda.current_device()}")
else:
    print("CUDA no está disponible. Usando CPU.")
