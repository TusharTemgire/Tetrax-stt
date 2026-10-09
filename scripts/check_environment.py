import sys
import torch
import transformers
import datasets

def check_env():
    print("========================================")
    print(" Tetrax STT Environment Diagnostic")
    print("========================================")
    print("Python version      :", sys.version.split()[0])
    print("PyTorch version     :", torch.__version__)
    print("Transformers version:", transformers.__version__)
    print("Datasets version    :", datasets.__version__)
    print("CUDA Available      :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU Device Name     :", torch.cuda.get_device_name(0))
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        print(f"VRAM                : {vram} GiB")
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU!")
    print("========================================")

if __name__ == "__main__":
    check_env()
