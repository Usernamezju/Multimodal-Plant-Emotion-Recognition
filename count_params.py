import torch
import torch.nn as nn

# 加载你的模型文件
ckpt = torch.load("plant_fusion_train_by_synth_data.pth", map_location="cpu")

# 如果是直接保存的 model.state_dict()
if isinstance(ckpt, dict) and "state_dict" not in ckpt:
    state_dict = ckpt
else:
    state_dict = ckpt.get("state_dict", ckpt)

# 计算总参数
total_params = 0
for k, v in state_dict.items():
    total_params += v.numel()

# 输出结果
print(f"\n🔥 模型总参数：{total_params:,}")
