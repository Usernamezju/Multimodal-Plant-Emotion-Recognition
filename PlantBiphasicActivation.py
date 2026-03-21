import torch
import torch.nn as nn
import torch.nn.functional as F

class PlantBiphasicActivation(nn.Module):
    def __init__(self, num_classes):
        super(PlantBiphasicActivation, self).__init__()
        # 1. 响应增益 k (Gain): 初始设为 1.0
        # 物理意义：植物对电信号特征的放大倍数
        self.k_fast = nn.Parameter(torch.ones(num_classes))
        self.k_slow = nn.Parameter(torch.ones(num_classes))
        
        # 2. 响应阈值 theta (Threshold): 初始设为 0
        # 物理意义：触发应激反应的最小信号强度
        self.theta_fast = nn.Parameter(torch.zeros(num_classes))
        self.theta_slow = nn.Parameter(torch.zeros(num_classes))
        
        # 3. 快慢配比 alpha (Biphasic Ratio): 初始设为 0.5 (中立)
        # 物理意义：判定该情绪是由“快相(AP)”还是“慢相(VP)”主导
        self.alpha = nn.Parameter(torch.full((num_classes,), 0.5))

    def forward(self, z):
        # 【修改点】：换用 softplus 解决“梯度坏死”
        # 生理学意义：模拟植物离子通道渐进式的开启特性，而不是生硬的 0/1 切断
        f_fast = F.softplus(self.k_fast * (z - self.theta_fast))
        f_slow = F.softplus(self.k_slow * (z - self.theta_slow))
        
        # 确保 alpha 在 [0, 1] 之间
        alpha_bounded = torch.sigmoid(self.alpha) 
        
        # 最终组合输出
        phi = alpha_bounded * f_fast + (1 - alpha_bounded) * f_slow
        
        # 竞争性线性归一化
        phi_norm = phi / (torch.sum(phi, dim=1, keepdim=True) + 1e-8)
        
        return phi_norm, f_fast, f_slow
    