import torch

# 查看是否有GPU
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 设备名:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# 如果你有 GPU，这行代码会在 GPU 上跑测试
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    print("✅ 成功在 GPU 上创建张量！")
else:
    print("❌ 仅在 CPU 上运行")