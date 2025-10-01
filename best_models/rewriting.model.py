import torch

path = "./best_model_20250827_123421_acc0.9516.pth"
weight = torch.load(path, map_location="cpu")

new_path = path.replace(".pth", "_cpu.pth")
torch.save(weight, new_path)

