import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Multi-processor count: {props.multi_processor_count}")
    device = torch.device('cuda:0')
    print(f"Using device: {device}")
    # Quick tensor test
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    z = torch.matmul(x, y)
    print(f"Test matmul sum: {z.sum().item():.4f}")
    print("GPU is working")
else:
    print("CUDA not available, falling back to CPU")
    device = torch.device('cpu')
    print(f"Using device: {device}")
